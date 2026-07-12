#!/usr/bin/env python3
"""Pure SPR continual-learning protocol on ISRUC EEG.

The subject split, pretrained backbone and old/new evaluation protocol match
BrainUICL. The online method is SPR: delayed data, expert/base NT-Xent,
Self-Centered filtering, a fixed Purified Buffer and supervised evaluation
fine-tuning. No guiding model, pseudo-label update or confidence gate is used.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.spr_eeg import (  # noqa: E402
    EpochMemoryRecord,
    NTXentLoss,
    PurifiedEpochBuffer,
    self_centered_clean_probabilities,
    symmetric_label_noise,
)
from rttdp_brainuicl_full import (  # noqa: E402
    SequenceDataset,
    clone_blocks,
    discover_subjects,
    evaluate,
    flat_labels,
    flat_logits,
    forward_blocks,
    load_pretrained,
    make_loader,
    merge_subject_paths,
    split_subjects,
    subject_paths,
)
from utils.config import ModelConfig  # noqa: E402
from utils.util import compute_aaf1, compute_aaa, compute_forget, fix_randomness  # noqa: E402


class RawSequenceDataset(Dataset):
    def __init__(self, data_paths: list[Path]):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        x = torch.from_numpy(np.load(self.data_paths[index]).astype(np.float32))
        return x[:, :2, :], x[:, 2:, :]


class MemorySequenceDataset(Dataset):
    """Load full sequence context and supervise only retained epoch indices."""

    def __init__(self, records: list[EpochMemoryRecord], sequence_length: int):
        grouped: dict[str, dict[int, int]] = defaultdict(dict)
        for record in records:
            grouped[record.data_path][record.epoch_index] = record.observed_label
        self.items = sorted(grouped.items())
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, retained = self.items[index]
        x = torch.from_numpy(np.load(path).astype(np.float32))
        labels = torch.full((self.sequence_length,), -100, dtype=torch.long)
        for epoch_index, label in retained.items():
            labels[epoch_index] = label
        return x[:, :2, :], x[:, 2:, :], labels


class EEGSimCLR(nn.Module):
    def __init__(self, feature_extractor, feature_encoder, embedding_dim: int = 512):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_encoder = feature_encoder
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

    def epoch_embeddings(self, eog, eeg, args):
        batch = eeg.shape[0]
        eog = eog.reshape(-1, args.model_param.EogNum, args.model_param.EpochLength)
        eeg = eeg.reshape(-1, args.model_param.EegNum, args.model_param.EpochLength)
        features = self.feature_extractor(eeg, eog)
        features = self.feature_encoder(features)
        return features.reshape(batch, args.model_param.SeqLength, -1)

    def forward(self, eog, eeg, args):
        sequence_embedding = self.epoch_embeddings(eog, eeg, args).mean(dim=1)
        return self.projector(sequence_embedding)


def augment_eeg(eog, eeg, *, jitter: float, mask_ratio: float, channel_drop: float):
    """Apply EEG-compatible SimCLR augmentations without changing labels."""

    eog_aug, eeg_aug = eog.clone(), eeg.clone()
    for signal in (eog_aug, eeg_aug):
        batch = signal.shape[0]
        scale = torch.empty((batch, 1, 1, 1), device=signal.device).uniform_(0.9, 1.1)
        signal.mul_(scale)
        if jitter > 0:
            amplitude = signal.std(dim=-1, keepdim=True).clamp_min(1e-6)
            signal.add_(torch.randn_like(signal) * amplitude * jitter)
        if channel_drop > 0:
            keep = torch.rand((batch, 1, signal.shape[2], 1), device=signal.device) >= channel_drop
            signal.mul_(keep)
        width = int(signal.shape[-1] * mask_ratio)
        if width > 0:
            starts = torch.randint(0, signal.shape[-1] - width + 1, (batch,), device=signal.device)
            for row, start in enumerate(starts.tolist()):
                signal[row, :, :, start : start + width] = 0
    return eog_aug, eeg_aug


def unique_memory_paths(memory: PurifiedEpochBuffer) -> list[Path]:
    return [Path(path) for path in sorted({record.data_path for record in memory.records})]


def next_replay_batch(iterator, loader):
    if loader is None:
        return None, iterator
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def ssl_step(model, criterion, optimizer, eog, eeg, args):
    first_eog, first_eeg = augment_eeg(
        eog, eeg, jitter=args.jitter, mask_ratio=args.mask_ratio, channel_drop=args.channel_drop
    )
    second_eog, second_eeg = augment_eeg(
        eog, eeg, jitter=args.jitter, mask_ratio=args.mask_ratio, channel_drop=args.channel_drop
    )
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(first_eog, first_eeg, args), model(second_eog, second_eeg, args))
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def train_expert(expert, loader, args):
    expert.train()
    optimizer = torch.optim.Adam(expert.parameters(), lr=args.ssl_lr, weight_decay=args.weight_decay)
    criterion = NTXentLoss(args.temperature)
    losses = []
    for _ in range(args.expert_epochs):
        for batch_index, (eog, eeg, _labels) in enumerate(loader):
            if args.max_ssl_batches and batch_index >= args.max_ssl_batches:
                break
            if eog.shape[0] < 2:
                continue
            eog, eeg = eog.to(args.device), eeg.to(args.device)
            losses.append(ssl_step(expert, criterion, optimizer, eog, eeg, args))
    return float(np.mean(losses)) if losses else math.nan


def train_base(base, optimizer, current_loader, memory, args):
    base.train()
    criterion = NTXentLoss(args.temperature)
    replay_paths = unique_memory_paths(memory)
    replay_loader = None
    if replay_paths:
        replay_loader = DataLoader(
            RawSequenceDataset(replay_paths),
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_worker,
            drop_last=False,
        )
    losses = []
    replay_iterator = iter(replay_loader) if replay_loader is not None else None
    for _ in range(args.base_epochs):
        for batch_index, (eog, eeg, _labels) in enumerate(current_loader):
            if args.max_ssl_batches and batch_index >= args.max_ssl_batches:
                break
            replay, replay_iterator = next_replay_batch(replay_iterator, replay_loader)
            if replay is not None:
                replay_eog, replay_eeg = replay
                eog = torch.cat((eog, replay_eog), dim=0)
                eeg = torch.cat((eeg, replay_eeg), dim=0)
            if eog.shape[0] < 2:
                continue
            eog, eeg = eog.to(args.device), eeg.to(args.device)
            losses.append(ssl_step(base, criterion, optimizer, eog, eeg, args))
    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def expert_embeddings(expert, loader, args):
    expert.eval()
    output = []
    for eog, eeg, _labels in loader:
        eog, eeg = eog.to(args.device), eeg.to(args.device)
        output.append(expert.epoch_embeddings(eog, eeg, args).cpu())
    return torch.cat(output, dim=0).numpy()


def current_observations(paths, noise_rate, args, step):
    true_labels = np.stack([np.load(path).astype(np.int64) for path in paths[1]])
    observed, noisy_mask = symmetric_label_noise(
        true_labels,
        noise_rate,
        args.model_param.NumClasses,
        seed=args.seed + 1000 * step,
    )
    return true_labels, observed, noisy_mask


def filter_delayed_buffer(expert, loader, paths, noise_rate, args, step):
    features = expert_embeddings(expert, loader, args)
    true_labels, observed, noisy_mask = current_observations(paths, noise_rate, args, step)
    clean_p = self_centered_clean_probabilities(
        features.reshape(-1, features.shape[-1]),
        observed.reshape(-1),
        ensembles=args.ensembles,
        bmm_iters=args.bmm_iters,
        seed=args.seed + 10000 * step,
    ).reshape(observed.shape)
    rng = np.random.default_rng(args.seed + 20000 * step)
    accepted = clean_p > rng.random(clean_p.shape)
    records = []
    for sequence_index, data_path in enumerate(paths[0]):
        for epoch_index in np.flatnonzero(accepted[sequence_index]):
            records.append(
                EpochMemoryRecord(
                    data_path=str(data_path),
                    epoch_index=int(epoch_index),
                    observed_label=int(observed[sequence_index, epoch_index]),
                    clean_probability=float(clean_p[sequence_index, epoch_index]),
                    true_label=int(true_labels[sequence_index, epoch_index]),
                )
            )
    accepted_count = int(accepted.sum())
    accepted_clean = int(((observed == true_labels) & accepted).sum())
    return records, {
        "delayed_epochs": int(observed.size),
        "injected_noisy_epochs": int(noisy_mask.sum()),
        "accepted_epochs": accepted_count,
        "acceptance_rate": float(accepted.mean()),
        "accepted_purity": float(accepted_clean / accepted_count) if accepted_count else math.nan,
        "mean_clean_probability": float(clean_p.mean()),
    }


def source_records(data_root, subjects):
    data_paths, label_paths = merge_subject_paths(data_root, list(subjects))
    records = []
    for data_path, label_path in zip(data_paths, label_paths):
        labels = np.load(label_path).astype(np.int64)
        records.extend(
            EpochMemoryRecord(str(data_path), index, int(label), 1.0, int(label))
            for index, label in enumerate(labels)
        )
    return records


def finetune_for_evaluation(base, initial_classifier, memory, args):
    blocks = (
        copy.deepcopy(base.feature_extractor).to(args.device),
        copy.deepcopy(base.feature_encoder).to(args.device),
        copy.deepcopy(initial_classifier).to(args.device),
    )
    dataset = MemorySequenceDataset(memory.records, args.model_param.SeqLength)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_worker,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(
        [parameter for block in blocks for parameter in block.parameters()],
        lr=args.ft_lr,
        weight_decay=args.weight_decay,
    )
    for block in blocks:
        block.train()
    losses = []
    for _ in range(args.ft_epochs):
        for batch_index, (eog, eeg, labels) in enumerate(loader):
            if args.max_ft_batches and batch_index >= args.max_ft_batches:
                break
            eog, eeg, labels = eog.to(args.device), eeg.to(args.device), labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_blocks(blocks, eog, eeg, args)
            loss = F.cross_entropy(flat_logits(logits), flat_labels(labels), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for block in blocks for parameter in block.parameters()], 5.0
            )
            optimizer.step()
            losses.append(float(loss.detach()))
    return blocks, float(np.mean(losses)) if losses else math.nan


def buffer_stats(memory):
    records = memory.records
    known = [record for record in records if record.true_label >= 0]
    correct = sum(record.true_label == record.observed_label for record in known)
    dynamic_known = [record for record in memory.dynamic if record.true_label >= 0]
    dynamic_correct = sum(record.true_label == record.observed_label for record in dynamic_known)
    return {
        "epochs": len(records),
        "source_epochs": len(memory.source),
        "dynamic_epochs": len(memory.dynamic),
        "unique_sequences": len({record.data_path for record in records}),
        "class_counts": memory.class_counts(),
        "purity": float(correct / len(known)) if known else math.nan,
        "dynamic_purity": (
            float(dynamic_correct / len(dynamic_known)) if dynamic_known else math.nan
        ),
    }


def plasticity_summary(performance):
    names = ("initial_acc", "before_acc", "after_acc", "initial_mf1", "before_mf1", "after_mf1")
    values = {name: [] for name in names}
    for metrics in performance["plasticity"].values():
        if len(metrics["ACC"]) < 3 or len(metrics["MF1"]) < 3:
            continue
        values["initial_acc"].append(metrics["ACC"][0])
        values["before_acc"].append(metrics["ACC"][1])
        values["after_acc"].append(metrics["ACC"][2])
        values["initial_mf1"].append(metrics["MF1"][0])
        values["before_mf1"].append(metrics["MF1"][1])
        values["after_mf1"].append(metrics["MF1"][2])
    return {name: float(np.mean(metric)) for name, metric in values.items()}


def save_progress(path, performance, config):
    summary = plasticity_summary(performance) if performance["completed_subjects"] else {}
    payload = {"config": config, "performance": performance, "summary": summary}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def run_noise_variant(args, noise_rate, train_idx, old_idx, new_order, config):
    fix_randomness(args.seed)
    variant_dir = args.output_root / f"noise_{noise_rate:.2f}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    initial_blocks = load_pretrained(args)
    base = EEGSimCLR(
        copy.deepcopy(initial_blocks[0]).to(args.device),
        copy.deepcopy(initial_blocks[1]).to(args.device),
    ).to(args.device)
    base_optimizer = torch.optim.Adam(base.parameters(), lr=args.ssl_lr, weight_decay=args.weight_decay)
    memory = PurifiedEpochBuffer(args.memory_capacity, args.source_capacity, args.model_param.NumClasses)
    memory.seed_source(source_records(args.data_root, train_idx), seed=args.seed)
    old_loader = make_loader(args.data_root, sorted(old_idx), args.eval_batch, False, args.num_worker)
    performance = {
        "method": "pure_spr_eeg",
        "noise_rate": noise_rate,
        "order": [int(subject) for subject in new_order],
        "completed_subjects": 0,
        "stability": {"ACC": [], "MF1": [], "AAA": [], "AAF1": [], "FR": []},
        "plasticity": {str(int(subject)): {"ACC": [], "MF1": []} for subject in new_order},
        "steps": [],
    }
    old_initial = evaluate(initial_blocks, old_loader, args)
    performance["stability"]["ACC"].append(old_initial["acc"])
    performance["stability"]["MF1"].append(old_initial["mf1"])
    performance["stability"]["AAA"].append(float(compute_aaa(performance["stability"]["ACC"])))
    performance["stability"]["AAF1"] = compute_aaf1(performance["stability"]["MF1"])
    performance["stability"]["FR"].append(float(compute_forget(performance["stability"]["ACC"])))
    inference_blocks = clone_blocks(initial_blocks, args)

    for step, subject in enumerate(new_order, start=1):
        print(
            f"[pure SPR noise={noise_rate:.2f}] subject {step}/{len(new_order)} id={subject}",
            flush=True,
        )
        previous_blocks = clone_blocks(inference_blocks, args)
        paths = subject_paths(args.data_root, int(subject))
        current_loader = DataLoader(
            SequenceDataset(paths),
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_worker,
            drop_last=False,
        )
        ordered_loader = DataLoader(
            SequenceDataset(paths),
            batch_size=args.eval_batch,
            shuffle=False,
            num_workers=args.num_worker,
            drop_last=False,
        )
        expert = EEGSimCLR(
            copy.deepcopy(initial_blocks[0]).to(args.device),
            copy.deepcopy(initial_blocks[1]).to(args.device),
        ).to(args.device)
        expert_loss = train_expert(expert, current_loader, args)
        base_loss = train_base(base, base_optimizer, current_loader, memory, args)
        accepted_records, filter_metrics = filter_delayed_buffer(
            expert, ordered_loader, paths, noise_rate, args, step
        )
        memory.update(accepted_records)
        inference_blocks, ft_loss = finetune_for_evaluation(
            base, initial_blocks[2], memory, args
        )

        new_loader = DataLoader(
            SequenceDataset(paths),
            batch_size=args.eval_batch,
            shuffle=False,
            num_workers=args.num_worker,
        )
        initial_result = evaluate(initial_blocks, new_loader, args)
        before_result = evaluate(previous_blocks, new_loader, args)
        after_result = evaluate(inference_blocks, new_loader, args)
        key = str(int(subject))
        performance["plasticity"][key]["ACC"] = [
            initial_result["acc"], before_result["acc"], after_result["acc"]
        ]
        performance["plasticity"][key]["MF1"] = [
            initial_result["mf1"], before_result["mf1"], after_result["mf1"]
        ]
        old_result = evaluate(inference_blocks, old_loader, args)
        stability = performance["stability"]
        stability["ACC"].append(old_result["acc"])
        stability["MF1"].append(old_result["mf1"])
        stability["AAA"].append(float(compute_aaa(stability["ACC"])))
        stability["AAF1"] = compute_aaf1(stability["MF1"])
        stability["FR"].append(float(compute_forget(stability["ACC"])))
        step_metrics = {
            "step": step,
            "subject": int(subject),
            "expert_nt_xent": expert_loss,
            "base_nt_xent": base_loss,
            "finetune_loss": ft_loss,
            "filter": filter_metrics,
            "buffer": buffer_stats(memory),
            "old_acc": old_result["acc"],
            "old_mf1": old_result["mf1"],
            "new_initial_acc": initial_result["acc"],
            "new_before_acc": before_result["acc"],
            "new_after_acc": after_result["acc"],
        }
        performance["steps"].append(step_metrics)
        performance["completed_subjects"] = step
        save_progress(variant_dir / "metrics.json", performance, config)
        print(
            f"  old ACC={old_result['acc']:.4f} MF1={old_result['mf1']:.4f}; "
            f"new before/after={before_result['acc']:.4f}/{after_result['acc']:.4f}; "
            f"accepted={filter_metrics['accepted_epochs']} P={len(memory)} "
            f"purity={step_metrics['buffer']['purity']:.4f}",
            flush=True,
        )

    summary = plasticity_summary(performance)
    final = {
        "noise_rate": noise_rate,
        "final_old_acc": performance["stability"]["ACC"][-1],
        "final_old_mf1": performance["stability"]["MF1"][-1],
        "aaa": performance["stability"]["AAA"][-1],
        "aaf1": performance["stability"]["AAF1"][-1],
        "fr": performance["stability"]["FR"][-1],
        "plasticity": summary,
        "buffer": performance["steps"][-1]["buffer"],
    }
    (variant_dir / "summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False))
    return final


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32"))
    parser.add_argument("--input-checkpoint-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/model_parameter"))
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments" / "rttdp_brainuicl_runs" / "pure_spr_eeg")
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=10)
    parser.add_argument("--noise-rates", type=float, nargs="+", default=[0.0, 0.4])
    parser.add_argument("--expert-epochs", type=int, default=10)
    parser.add_argument("--base-epochs", type=int, default=10)
    parser.add_argument("--ft-epochs", type=int, default=10)
    parser.add_argument("--max-ssl-batches", type=int, default=0)
    parser.add_argument("--max-ft-batches", type=int, default=0)
    parser.add_argument("--ssl-lr", type=float, default=1e-6)
    parser.add_argument("--ft-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--jitter", type=float, default=0.01)
    parser.add_argument("--mask-ratio", type=float, default=0.1)
    parser.add_argument("--channel-drop", type=float, default=0.1)
    parser.add_argument("--ensembles", type=int, default=5)
    parser.add_argument("--bmm-iters", type=int, default=10)
    parser.add_argument("--memory-capacity", type=int, default=5000)
    parser.add_argument("--source-capacity", type=int, default=3000)
    return parser.parse_args()


def main():
    args = parse_args()
    fix_randomness(args.seed)
    args.dataset = "ISRUC"
    args.model_param = ModelConfig(args.dataset)
    args.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    subjects = discover_subjects(args.data_root)
    train_idx, val_idx, old_idx, full_new_order = split_subjects(subjects, args.seed)
    new_order = list(full_new_order)
    if args.max_subjects > 0:
        new_order = new_order[: args.max_subjects]
    args.output_root.mkdir(parents=True, exist_ok=True)
    split = {
        "train": sorted(int(subject) for subject in train_idx),
        "val": sorted(int(subject) for subject in val_idx),
        "old_generalization": sorted(int(subject) for subject in old_idx),
        "new_order": [int(subject) for subject in new_order],
        "full_new_order": [int(subject) for subject in full_new_order],
    }
    config = {
        "method": "pure_spr_eeg",
        "device": str(args.device),
        "seed": args.seed,
        "expert_epochs": args.expert_epochs,
        "base_epochs": args.base_epochs,
        "ft_epochs": args.ft_epochs,
        "temperature": args.temperature,
        "memory_capacity_epochs": args.memory_capacity,
        "source_capacity_epochs": args.source_capacity,
        "dynamic_capacity_epochs": args.memory_capacity - args.source_capacity,
        "confidence_gate": False,
        "guiding_model": False,
        "split": split,
    }
    (args.output_root / "protocol.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)
    results = []
    for noise_rate in args.noise_rates:
        results.append(run_noise_variant(args, noise_rate, train_idx, old_idx, new_order, config))
    (args.output_root / "comparison.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
