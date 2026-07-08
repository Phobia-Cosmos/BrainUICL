#!/usr/bin/env python3
"""
White-box proxy degradation probe for BrainUICL ISRUC.

This script does not modify the original BrainUICL continual-learning code.
It loads the pretrained ISRUC model, simulates short proxy updates on candidate
new individuals, and measures whether a surrogate/NHE-style objective can
degrade the old/generalization set.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.pretrain_net import FeatureExtractor, SleepMLP, TransformerEncoder  # noqa: E402
from utils.util import fix_randomness  # noqa: E402


class SequenceDataset(Dataset):
    def __init__(self, paths: tuple[list[Path], list[Path]]):
        self.data_paths, self.label_paths = paths

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int):
        x = np.load(self.data_paths[index]).astype(np.float32)
        y = np.load(self.label_paths[index]).astype(np.int64)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x[:, :2, :], x[:, 2:, :], y


def subject_paths(data_root: Path, subject: int) -> tuple[list[Path], list[Path]]:
    data_dir = data_root / str(subject) / "data"
    label_dir = data_root / str(subject) / "label"
    data_paths: list[Path] = []
    label_paths: list[Path] = []
    idx = 0
    while (data_dir / f"{idx}.npy").exists():
        data_paths.append(data_dir / f"{idx}.npy")
        label_paths.append(label_dir / f"{idx}.npy")
        idx += 1
    return data_paths, label_paths


def merge_subject_paths(data_root: Path, subjects: list[int]) -> tuple[list[Path], list[Path]]:
    all_data: list[Path] = []
    all_label: list[Path] = []
    for subject in subjects:
        data_paths, label_paths = subject_paths(data_root, subject)
        all_data.extend(data_paths)
        all_label.extend(label_paths)
    return all_data, all_label


def discover_subjects(data_root: Path) -> list[int]:
    subjects = []
    for sid in range(1, 101):
        if sid in (8, 40):
            continue
        if (data_root / str(sid) / "data" / "0.npy").exists():
            subjects.append(sid)
    if len(subjects) < 5:
        raise RuntimeError(f"Need at least 5 processed subjects under {data_root}, found {len(subjects)}")
    return subjects


def split_subjects(subjects: list[int], seed: int):
    fix_randomness(seed)
    idx = list(subjects)
    path_len = len(idx)
    old_count = max(1, int(path_len * 0.2))
    new_count = max(1, int(path_len * 0.5))
    new_count = min(new_count, path_len - old_count - 2)

    old_task_idx = list(np.random.choice(idx, old_count, replace=False))
    remaining_idx = sorted(set(idx) - set(old_task_idx))
    new_task_idx = list(np.random.choice(remaining_idx, new_count, replace=False))
    train_val_idx = sorted(set(idx) - set(old_task_idx) - set(new_task_idx))
    train_count = max(1, int(len(train_val_idx) * 0.8))
    train_count = min(train_count, len(train_val_idx) - 1)
    train_idx = list(np.random.choice(train_val_idx, train_count, replace=False))
    val_idx = [i for i in train_val_idx if i not in train_idx]
    return train_idx, val_idx, old_task_idx, new_task_idx


def build_blocks(args, device):
    feature_extractor = FeatureExtractor(args).float().to(device)
    feature_encoder = TransformerEncoder(args).float().to(device)
    classifier = SleepMLP(args).float().to(device)
    return feature_extractor, feature_encoder, classifier


def load_pretrained(args, checkpoint_root: Path, device):
    blocks = build_blocks(args, device)
    ckpt_dir = checkpoint_root / "ISRUC" / "Pretrain"
    blocks[0].load_state_dict(torch.load(ckpt_dir / f"feature_extractor_parameter_{args.seed}.pkl", map_location=device))
    blocks[1].load_state_dict(torch.load(ckpt_dir / f"feature_encoder_parameter_{args.seed}.pkl", map_location=device))
    blocks[2].load_state_dict(torch.load(ckpt_dir / f"sleep_classifier_parameter_{args.seed}.pkl", map_location=device))
    return blocks


def clone_blocks(blocks, device):
    return tuple(copy.deepcopy(block).to(device) for block in blocks)


def set_train(blocks, train: bool):
    for block in blocks:
        block.train(train)


def forward_blocks(blocks, eog, eeg):
    batch = eeg.shape[0]
    eog = eog.reshape(-1, 2, 3000)
    eeg = eeg.reshape(-1, 6, 3000)
    features = blocks[0](eeg, eog)
    features = blocks[1](features)
    logits = blocks[2](features)
    return logits.reshape(batch, 5, 20)


def flatten_logits(logits):
    return logits.permute(0, 2, 1).reshape(-1, 5)


def flatten_labels(labels):
    return labels.reshape(-1).long()


@torch.no_grad()
def evaluate(blocks, loader, device, max_batches: int = 0):
    set_train(blocks, False)
    y_true: list[int] = []
    y_pred: list[int] = []
    entropies: list[float] = []
    confidences: list[float] = []
    for batch_idx, (eog, eeg, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        eog = eog.to(device)
        eeg = eeg.to(device)
        labels = labels.to(device)
        logits = forward_blocks(blocks, eog, eeg)
        flat_logits = flatten_logits(logits)
        probs = flat_logits.softmax(dim=1)
        pred = probs.argmax(dim=1)
        entropy = -(probs * (probs + 1e-12).log()).sum(dim=1)
        y_true.extend(flatten_labels(labels).detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
        entropies.extend(entropy.detach().cpu().tolist())
        confidences.extend(probs.max(dim=1).values.detach().cpu().tolist())
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "mf1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3, 4], zero_division=0)),
        "entropy": float(np.mean(entropies)) if entropies else math.nan,
        "confidence": float(np.mean(confidences)) if confidences else math.nan,
        "n_epochs": len(y_true),
    }


def nhe_target_from_logits(flat_logits):
    pred = flat_logits.detach().argmax(dim=1)
    target = torch.ones_like(flat_logits)
    target[torch.arange(pred.shape[0], device=flat_logits.device), pred] = 0.0
    target = target / target.sum(dim=1, keepdim=True)
    return target


def proxy_update(
    blocks,
    subject_loader,
    source_iter,
    device,
    mode: str,
    lr: float,
    update_batches: int,
    source_weight: float,
    target_weight: float,
):
    set_train(blocks, True)
    optimizer = torch.optim.Adam(
        list(blocks[0].parameters()) + list(blocks[1].parameters()) + list(blocks[2].parameters()),
        lr=lr,
        betas=(0.5, 0.99),
        weight_decay=3e-4,
    )

    losses: list[float] = []
    subject_iter = cycle(subject_loader)
    for _ in range(update_batches):
        eog, eeg, labels = next(subject_iter)
        eog = eog.to(device)
        eeg = eeg.to(device)
        labels = labels.to(device)

        logits = forward_blocks(blocks, eog, eeg)
        flat_logits = flatten_logits(logits)

        if mode == "benign":
            pseudo = flat_logits.detach().argmax(dim=1)
            subject_loss = F.cross_entropy(flat_logits, pseudo)
        elif mode == "nhe_attack":
            target = nhe_target_from_logits(flat_logits)
            subject_loss = F.kl_div(F.log_softmax(flat_logits, dim=1), target, reduction="batchmean")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        loss = target_weight * subject_loss
        if source_iter is not None and source_weight > 0:
            src_eog, src_eeg, src_labels = next(source_iter)
            src_eog = src_eog.to(device)
            src_eeg = src_eeg.to(device)
            src_labels = src_labels.to(device)
            src_logits = forward_blocks(blocks, src_eog, src_eeg)
            src_loss = F.cross_entropy(src_logits, src_labels.long())
            loss = loss + source_weight * src_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(blocks[0].parameters()) + list(blocks[1].parameters()) + list(blocks[2].parameters()),
            5.0,
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else math.nan


def score_candidates(base_blocks, candidates, data_root, old_loader, source_loader, args, device):
    source_iter = cycle(source_loader) if source_loader is not None else None
    rows = []
    baseline_old = evaluate(base_blocks, old_loader, device, args.eval_max_batches)
    for subject in candidates:
        subject_loader = make_loader(data_root, [subject], args.batch, shuffle=True)
        subject_eval = evaluate(base_blocks, subject_loader, device, args.eval_max_batches)
        proxy_blocks = clone_blocks(base_blocks, device)
        loss = proxy_update(
            proxy_blocks,
            subject_loader,
            source_iter,
            device,
            mode="nhe_attack",
            lr=args.attack_lr,
            update_batches=args.update_batches,
            source_weight=args.source_weight,
            target_weight=args.target_weight,
        )
        attacked_old = evaluate(proxy_blocks, old_loader, device, args.eval_max_batches)
        rows.append(
            {
                "subject": int(subject),
                "subject_acc_m0": subject_eval["acc"],
                "subject_mf1_m0": subject_eval["mf1"],
                "subject_entropy_m0": subject_eval["entropy"],
                "subject_confidence_m0": subject_eval["confidence"],
                "proxy_loss": loss,
                "old_acc_after_proxy": attacked_old["acc"],
                "old_mf1_after_proxy": attacked_old["mf1"],
                "old_acc_drop": baseline_old["acc"] - attacked_old["acc"],
                "old_mf1_drop": baseline_old["mf1"] - attacked_old["mf1"],
            }
        )
        del proxy_blocks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows.sort(key=lambda row: (row["old_mf1_drop"], row["old_acc_drop"]), reverse=True)
    return baseline_old, rows


def run_sequence(base_blocks, order, data_root, old_loader, source_loader, args, device, mode):
    blocks = clone_blocks(base_blocks, device)
    source_iter = cycle(source_loader) if source_loader is not None else None
    curve = [{"step": 0, "subject": None, **evaluate(blocks, old_loader, device, args.eval_max_batches)}]
    for step, subject in enumerate(order, start=1):
        subject_loader = make_loader(data_root, [subject], args.batch, shuffle=True)
        loss = proxy_update(
            blocks,
            subject_loader,
            source_iter,
            device,
            mode=mode,
            lr=args.attack_lr if mode == "nhe_attack" else args.benign_lr,
            update_batches=args.update_batches,
            source_weight=args.source_weight,
            target_weight=args.target_weight,
        )
        metrics = evaluate(blocks, old_loader, device, args.eval_max_batches)
        curve.append({"step": step, "subject": int(subject), "loss": loss, **metrics})
    del blocks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return curve


def make_loader(data_root: Path, subjects: list[int], batch: int, shuffle: bool):
    return DataLoader(SequenceDataset(merge_subject_paths(data_root, subjects)), batch_size=batch, shuffle=shuffle, num_workers=0)


def write_report(output_dir: Path, payload: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "proxy_degradation_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    lines = []
    lines.append("# BrainUICL Proxy Degradation Probe\n")
    lines.append("This is an independent probe script. It does not modify the original BrainUICL CL code.\n")
    lines.append("## Config\n")
    lines.append("```json\n" + json.dumps(payload["config"], indent=2, ensure_ascii=False) + "\n```\n")
    lines.append("## Split\n")
    lines.append("```json\n" + json.dumps(payload["split"], indent=2, ensure_ascii=False) + "\n```\n")
    lines.append("## Baseline Old/Generalization Metrics\n")
    lines.append("```json\n" + json.dumps(payload["baseline_old"], indent=2, ensure_ascii=False) + "\n```\n")
    lines.append("## Candidate Proxy Harmfulness\n")
    lines.append("| rank | subject | old ACC drop | old MF1 drop | M0 subject ACC | M0 subject MF1 | entropy | confidence |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for rank, row in enumerate(payload["candidate_scores"], start=1):
        lines.append(
            f"| {rank} | {row['subject']} | {row['old_acc_drop']:.4f} | {row['old_mf1_drop']:.4f} | "
            f"{row['subject_acc_m0']:.4f} | {row['subject_mf1_m0']:.4f} | "
            f"{row['subject_entropy_m0']:.4f} | {row['subject_confidence_m0']:.4f} |\n"
        )
    lines.append("\n## Curves\n")
    for name in ["benign_natural_curve", "benign_selected_curve", "attack_selected_curve"]:
        lines.append(f"### {name}\n")
        lines.append("| step | subject | ACC | MF1 | entropy | confidence |\n")
        lines.append("|---:|---:|---:|---:|---:|---:|\n")
        for row in payload[name]:
            subject = "" if row["subject"] is None else row["subject"]
            lines.append(
                f"| {row['step']} | {subject} | {row['acc']:.4f} | {row['mf1']:.4f} | "
                f"{row['entropy']:.4f} | {row['confidence']:.4f} |\n"
            )
        lines.append("\n")

    attack_final = payload["attack_selected_curve"][-1]
    benign_final = payload["benign_selected_curve"][-1]
    base = payload["baseline_old"]
    lines.append("## Quick Interpretation\n")
    lines.append(
        f"- Attack-selected final old/generalization ACC drop: {base['acc'] - attack_final['acc']:.4f}; "
        f"MF1 drop: {base['mf1'] - attack_final['mf1']:.4f}.\n"
    )
    lines.append(
        f"- Benign updates on the same selected subjects final ACC drop: {base['acc'] - benign_final['acc']:.4f}; "
        f"MF1 drop: {base['mf1'] - benign_final['mf1']:.4f}.\n"
    )
    lines.append("- A larger attack drop than benign drop is preliminary evidence that a proxy objective can drive CL degradation.\n")
    (output_dir / "proxy_degradation_report.md").write_text("".join(lines))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/model_parameter"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments" / "proxy_degradation")
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--sequential-k", type=int, default=3)
    parser.add_argument("--update-batches", type=int, default=10)
    parser.add_argument("--eval-max-batches", type=int, default=0, help="0 means full old/generalization evaluation.")
    parser.add_argument("--attack-lr", type=float, default=5e-5)
    parser.add_argument("--benign-lr", type=float, default=5e-5)
    parser.add_argument("--source-weight", type=float, default=0.2)
    parser.add_argument("--target-weight", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    fix_randomness(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    model_args = SimpleNamespace(dataset="ISRUC", gpu=args.gpu, seed=args.seed)

    subjects = discover_subjects(args.data_root)
    train_idx, val_idx, old_idx, new_idx = split_subjects(subjects, args.seed)
    base_blocks = load_pretrained(model_args, args.checkpoint_root, device)

    old_loader = make_loader(args.data_root, sorted(old_idx), args.batch, shuffle=False)
    source_loader = make_loader(args.data_root, sorted(train_idx), args.batch, shuffle=True)
    candidates = list(new_idx[: args.candidate_count])

    baseline_old, candidate_rows = score_candidates(
        base_blocks, candidates, args.data_root, old_loader, source_loader, args, device
    )
    selected_order = [row["subject"] for row in candidate_rows[: args.sequential_k]]
    natural_order = list(new_idx[: args.sequential_k])

    benign_natural_curve = run_sequence(
        base_blocks, natural_order, args.data_root, old_loader, source_loader, args, device, mode="benign"
    )
    benign_selected_curve = run_sequence(
        base_blocks, selected_order, args.data_root, old_loader, source_loader, args, device, mode="benign"
    )
    attack_selected_curve = run_sequence(
        base_blocks, selected_order, args.data_root, old_loader, source_loader, args, device, mode="nhe_attack"
    )

    payload = {
        "config": {
            "data_root": str(args.data_root),
            "checkpoint_root": str(args.checkpoint_root),
            "seed": args.seed,
            "device": str(device),
            "batch": args.batch,
            "candidate_count": args.candidate_count,
            "sequential_k": args.sequential_k,
            "update_batches": args.update_batches,
            "eval_max_batches": args.eval_max_batches,
            "attack_lr": args.attack_lr,
            "benign_lr": args.benign_lr,
            "source_weight": args.source_weight,
            "target_weight": args.target_weight,
        },
        "split": {
            "train": sorted(int(x) for x in train_idx),
            "val": sorted(int(x) for x in val_idx),
            "old_generalization": sorted(int(x) for x in old_idx),
            "new_order_prefix": [int(x) for x in new_idx[: max(args.candidate_count, args.sequential_k)]],
            "candidate_subjects": [int(x) for x in candidates],
            "selected_proxy_order": [int(x) for x in selected_order],
            "natural_order": [int(x) for x in natural_order],
        },
        "baseline_old": baseline_old,
        "candidate_scores": candidate_rows,
        "benign_natural_curve": benign_natural_curve,
        "benign_selected_curve": benign_selected_curve,
        "attack_selected_curve": attack_selected_curve,
    }
    write_report(args.output_dir, payload)
    print(json.dumps(payload["baseline_old"], indent=2))
    print("selected_proxy_order", selected_order)
    print("report", args.output_dir / "proxy_degradation_report.md")


if __name__ == "__main__":
    main()
