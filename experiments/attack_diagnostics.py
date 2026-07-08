#!/usr/bin/env python3
"""
Diagnostics for BrainUICL RTTDP-style attacks.

This script is read-only with respect to the original BrainUICL code and saved
checkpoints. It creates visualizations for:
1. normal vs PGD-poisoned input distributions,
2. confidence-filter pass rates,
3. gradient direction/norm differences between clean CL and attack objectives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.rttdp_brainuicl_full import (  # noqa: E402
    AttackState,
    SequenceDataset,
    flat_labels,
    flat_logits,
    forward_blocks,
    poison_new_batch,
)
from model.pretrain_net import FeatureExtractor, SleepMLP, TransformerEncoder  # noqa: E402
from utils.config import ModelConfig  # noqa: E402


MODULE_FILES = {
    "feature_extractor": "feature_extractor_parameter_{seed}.pkl",
    "feature_encoder": "feature_encoder_parameter_{seed}.pkl",
    "sleep_classifier": "sleep_classifier_parameter_{seed}.pkl",
}


def subject_paths_limited(data_root: Path, subject: int, limit: int) -> tuple[list[Path], list[Path]]:
    data_dir = data_root / str(subject) / "data"
    label_dir = data_root / str(subject) / "label"
    data_paths, label_paths = [], []
    idx = 0
    while (data_dir / f"{idx}.npy").exists():
        data_paths.append(data_dir / f"{idx}.npy")
        label_paths.append(label_dir / f"{idx}.npy")
        idx += 1
        if limit and idx >= limit:
            break
    return data_paths, label_paths


def merge_subject_paths_limited(data_root: Path, subjects: list[int], per_subject_limit: int) -> tuple[list[Path], list[Path]]:
    data_paths, label_paths = [], []
    for subject in subjects:
        d, l = subject_paths_limited(data_root, subject, per_subject_limit)
        data_paths.extend(d)
        label_paths.extend(l)
    return data_paths, label_paths


def build_blocks(args):
    return (
        FeatureExtractor(args).float().to(args.device),
        TransformerEncoder(args).float().to(args.device),
        SleepMLP(args).float().to(args.device),
    )


def load_blocks(checkpoint_dir: Path, args):
    blocks = build_blocks(args)
    for block, module in zip(blocks, MODULE_FILES):
        state = torch.load(checkpoint_dir / MODULE_FILES[module].format(seed=args.seed), map_location=args.device)
        block.load_state_dict(state)
        block.to(args.device)
    return blocks


def set_train(blocks, train: bool):
    for block in blocks:
        block.train(train)


def zero_grad(blocks):
    for block in blocks:
        block.zero_grad(set_to_none=True)


def model_embeddings(blocks, eog, eeg, args):
    batch = eeg.shape[0]
    eog_f = eog.reshape(-1, args.model_param.EogNum, args.model_param.EpochLength)
    eeg_f = eeg.reshape(-1, args.model_param.EegNum, args.model_param.EpochLength)
    features = blocks[0](eeg_f, eog_f)
    features = blocks[1](features)
    return features.reshape(batch, args.model_param.SeqLength, -1).mean(dim=1)


def signal_stats(eog, eeg):
    # Per sequence statistics over channel and time. This is intentionally simple
    # and model-independent so it can show whether PGD changes raw signal scale.
    x = torch.cat((eog, eeg), dim=2)  # batch, seq, channels, length
    vals = []
    for fn in [torch.mean, torch.std]:
        vals.append(fn(x, dim=(1, 3)))
    q05 = torch.quantile(x, 0.05, dim=3).mean(dim=1)
    q95 = torch.quantile(x, 0.95, dim=3).mean(dim=1)
    vals.extend([q05, q95])
    return torch.cat(vals, dim=1)


def confidence_stats(blocks, eog, eeg, args):
    with torch.no_grad():
        logits = forward_blocks(blocks, eog, eeg, args)
        probs = logits.softmax(dim=1)
        max_prob = probs.max(dim=1).values
        pass_mask = (max_prob >= args.confidence).sum(dim=1) >= args.confident_epoch_n
    return {
        "avg_epoch_conf": max_prob.mean(dim=1).detach().cpu().numpy(),
        "pass_mask": pass_mask.detach().cpu().numpy(),
    }


def collect_distribution(args, blocks, subjects):
    rows = []
    state = AttackState(args.model_param.NumClasses, args.device)
    set_train(blocks, False)
    for subject in subjects:
        paths = merge_subject_paths_limited(args.data_root, [subject], args.max_seq_per_subject)
        loader = DataLoader(SequenceDataset(paths), batch_size=args.batch, shuffle=False, num_workers=0)
        for eog, eeg, labels in loader:
            eog = eog.to(args.device)
            eeg = eeg.to(args.device)
            labels = labels.to(args.device)
            with torch.no_grad():
                emb_clean = model_embeddings(blocks, eog, eeg, args)
                stats_clean = signal_stats(eog, eeg)
            conf_clean = confidence_stats(blocks, eog, eeg, args)

            eog_adv, eeg_adv = poison_new_batch(eog, eeg, blocks, args.pgd_attack_mode, state, args)
            with torch.no_grad():
                emb_adv = model_embeddings(blocks, eog_adv, eeg_adv, args)
                stats_adv = signal_stats(eog_adv, eeg_adv)
            conf_adv = confidence_stats(blocks, eog_adv, eeg_adv, args)

            rel_eog = torch.linalg.norm((eog_adv - eog).reshape(eog.shape[0], -1), dim=1) / (
                torch.linalg.norm(eog.reshape(eog.shape[0], -1), dim=1) + 1e-12
            )
            rel_eeg = torch.linalg.norm((eeg_adv - eeg).reshape(eeg.shape[0], -1), dim=1) / (
                torch.linalg.norm(eeg.reshape(eeg.shape[0], -1), dim=1) + 1e-12
            )

            for idx in range(eog.shape[0]):
                rows.append(
                    {
                        "subject": int(subject),
                        "kind": "clean",
                        "embedding": emb_clean[idx].detach().cpu().numpy(),
                        "signal_stats": stats_clean[idx].detach().cpu().numpy(),
                        "avg_epoch_conf": float(conf_clean["avg_epoch_conf"][idx]),
                        "pass_confidence": bool(conf_clean["pass_mask"][idx]),
                        "rel_eog_delta": 0.0,
                        "rel_eeg_delta": 0.0,
                    }
                )
                rows.append(
                    {
                        "subject": int(subject),
                        "kind": "pgd",
                        "embedding": emb_adv[idx].detach().cpu().numpy(),
                        "signal_stats": stats_adv[idx].detach().cpu().numpy(),
                        "avg_epoch_conf": float(conf_adv["avg_epoch_conf"][idx]),
                        "pass_confidence": bool(conf_adv["pass_mask"][idx]),
                        "rel_eog_delta": float(rel_eog[idx].detach().cpu()),
                        "rel_eeg_delta": float(rel_eeg[idx].detach().cpu()),
                    }
                )
    return rows


def pca_plot(rows, key, out_path, title):
    matrix = np.stack([row[key] for row in rows])
    points = PCA(n_components=2, random_state=0).fit_transform(matrix)
    subjects = sorted({row["subject"] for row in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(subjects), 2)))
    color_map = {subject: colors[i % len(colors)] for i, subject in enumerate(subjects)}
    markers = {"clean": "o", "pgd": "x"}

    plt.figure(figsize=(8, 6))
    for subject in subjects:
        for kind in ["clean", "pgd"]:
            idx = [i for i, row in enumerate(rows) if row["subject"] == subject and row["kind"] == kind]
            if not idx:
                continue
            plt.scatter(
                points[idx, 0],
                points[idx, 1],
                s=38,
                marker=markers[kind],
                color=color_map[subject],
                alpha=0.78,
                label=f"S{subject} {kind}",
            )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def tsne_plot(rows, key, out_path, title):
    matrix = np.stack([row[key] for row in rows])
    perplexity = max(5, min(30, (matrix.shape[0] - 1) // 3))
    points = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=0,
    ).fit_transform(matrix)
    subjects = sorted({row["subject"] for row in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(subjects), 2)))
    color_map = {subject: colors[i % len(colors)] for i, subject in enumerate(subjects)}
    markers = {"clean": "o", "pgd": "x"}

    plt.figure(figsize=(8, 6))
    for subject in subjects:
        for kind in ["clean", "pgd"]:
            idx = [i for i, row in enumerate(rows) if row["subject"] == subject and row["kind"] == kind]
            if not idx:
                continue
            plt.scatter(
                points[idx, 0],
                points[idx, 1],
                s=38,
                marker=markers[kind],
                color=color_map[subject],
                alpha=0.78,
                label=f"S{subject} {kind}",
            )
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def confidence_plot(rows, out_path):
    subjects = sorted({row["subject"] for row in rows})
    clean_rates, pgd_rates, clean_conf, pgd_conf = [], [], [], []
    for subject in subjects:
        clean = [row for row in rows if row["subject"] == subject and row["kind"] == "clean"]
        pgd = [row for row in rows if row["subject"] == subject and row["kind"] == "pgd"]
        clean_rates.append(np.mean([row["pass_confidence"] for row in clean]))
        pgd_rates.append(np.mean([row["pass_confidence"] for row in pgd]))
        clean_conf.append(np.mean([row["avg_epoch_conf"] for row in clean]))
        pgd_conf.append(np.mean([row["avg_epoch_conf"] for row in pgd]))

    x = np.arange(len(subjects))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x - width / 2, clean_rates, width, label="clean")
    axes[0].bar(x + width / 2, pgd_rates, width, label="pgd")
    axes[0].set_title("Confidence-filter pass rate")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(s) for s in subjects])
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("subject")
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, clean_conf, width, label="clean")
    axes[1].bar(x + width / 2, pgd_conf, width, label="pgd")
    axes[1].set_title("Average epoch confidence")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(s) for s in subjects])
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("subject")
    axes[1].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def perturbation_plot(rows, out_path):
    pgd = [row for row in rows if row["kind"] == "pgd"]
    rel_eog = [row["rel_eog_delta"] for row in pgd]
    rel_eeg = [row["rel_eeg_delta"] for row in pgd]
    plt.figure(figsize=(7, 4))
    plt.hist(rel_eog, bins=20, alpha=0.7, label="EOG rel L2")
    plt.hist(rel_eeg, bins=20, alpha=0.7, label="EEG rel L2")
    plt.title("PGD perturbation magnitude")
    plt.xlabel("||adv - clean|| / ||clean||")
    plt.ylabel("sequence count")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def grad_vector(blocks):
    vectors = []
    module_norms = {}
    for module_name, block in zip(MODULE_FILES, blocks):
        module_parts = []
        for param in block.parameters():
            if param.grad is None:
                continue
            g = param.grad.detach().flatten().float().cpu()
            vectors.append(g)
            module_parts.append(g)
        module_norms[module_name] = float(torch.linalg.norm(torch.cat(module_parts))) if module_parts else 0.0
    return torch.cat(vectors), module_norms


def backward_and_collect(blocks, loss):
    zero_grad(blocks)
    loss.backward()
    return grad_vector(blocks)


def pseudo_loss(blocks, teacher_blocks, eog, eeg, args):
    with torch.no_grad():
        teacher_logits = flat_logits(forward_blocks(teacher_blocks, eog, eeg, args))
        probs = teacher_logits.softmax(dim=1)
        max_prob, labels = probs.max(dim=1)
    logits = flat_logits(forward_blocks(blocks, eog, eeg, args))
    mask = max_prob > args.confidence
    if int(mask.sum()) == 0:
        return F.cross_entropy(logits, labels)
    return F.cross_entropy(logits[mask], labels[mask])


def nhe_loss(blocks, eog, eeg, state, args):
    logits = flat_logits(forward_blocks(blocks, eog, eeg, args))
    target = state.nhe_target(logits)
    return F.kl_div(F.log_softmax(logits, dim=1), target, reduction="batchmean")


def gradient_diagnostics(args, checkpoint_dir, split, subject):
    blocks = load_blocks(checkpoint_dir, args)
    teacher_blocks = load_blocks(checkpoint_dir, args)
    set_train(blocks, True)
    set_train(teacher_blocks, False)

    train_paths = merge_subject_paths_limited(args.data_root, split["train"], args.max_source_seq)
    source_loader = DataLoader(SequenceDataset(train_paths), batch_size=args.batch, shuffle=False, num_workers=0)
    new_paths = merge_subject_paths_limited(args.data_root, [subject], args.max_seq_per_subject)
    new_loader = DataLoader(SequenceDataset(new_paths), batch_size=args.batch, shuffle=False, num_workers=0)
    source_eog, source_eeg, source_label = next(iter(source_loader))
    new_eog, new_eeg, _ = next(iter(new_loader))
    source_eog = source_eog.to(args.device)
    source_eeg = source_eeg.to(args.device)
    source_label = source_label.to(args.device)
    new_eog = new_eog.to(args.device)
    new_eeg = new_eeg.to(args.device)

    state = AttackState(args.model_param.NumClasses, args.device)
    gradients = {}
    module_norms = {}

    source_logits = flat_logits(forward_blocks(blocks, source_eog, source_eeg, args))
    source_loss = F.cross_entropy(source_logits, flat_labels(source_label))
    gradients["source_replay_ce"], module_norms["source_replay_ce"] = backward_and_collect(blocks, source_loss)

    clean_loss = pseudo_loss(blocks, teacher_blocks, new_eog, new_eeg, args)
    gradients["new_pseudo_ce"], module_norms["new_pseudo_ce"] = backward_and_collect(blocks, clean_loss)

    model_attack_loss = nhe_loss(blocks, new_eog, new_eeg, state, args)
    gradients["model_nhe_kl"], module_norms["model_nhe_kl"] = backward_and_collect(blocks, model_attack_loss)

    new_eog_adv, new_eeg_adv = poison_new_batch(new_eog, new_eeg, teacher_blocks, args.pgd_attack_mode, state, args)
    pgd_loss = pseudo_loss(blocks, teacher_blocks, new_eog_adv, new_eeg_adv, args)
    gradients["pgd_pseudo_ce"], module_norms["pgd_pseudo_ce"] = backward_and_collect(blocks, pgd_loss)

    names = list(gradients)
    cosine = np.zeros((len(names), len(names)), dtype=np.float64)
    norms = {}
    for i, name_i in enumerate(names):
        gi = gradients[name_i]
        norms[name_i] = float(torch.linalg.norm(gi))
        for j, name_j in enumerate(names):
            gj = gradients[name_j]
            value = float(F.cosine_similarity(gi, gj, dim=0))
            cosine[i, j] = max(-1.0, min(1.0, value))
    return {
        "names": names,
        "cosine": cosine,
        "norms": norms,
        "module_norms": module_norms,
    }


def gradient_plots(grad, output_dir):
    names = grad["names"]
    cosine = grad["cosine"]
    plt.figure(figsize=(6, 5))
    plt.imshow(cosine, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="cosine")
    plt.xticks(range(len(names)), names, rotation=35, ha="right")
    plt.yticks(range(len(names)), names)
    plt.title("Gradient cosine similarity")
    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(j, i, f"{cosine[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "gradient_cosine.png", dpi=180)
    plt.close()

    module_names = list(MODULE_FILES)
    x = np.arange(len(module_names))
    width = 0.18
    plt.figure(figsize=(9, 4.5))
    for idx, name in enumerate(names):
        values = [grad["module_norms"][name].get(module, 0.0) for module in module_names]
        plt.bar(x + (idx - 1.5) * width, values, width, label=name)
    plt.yscale("log")
    plt.xticks(x, module_names)
    plt.ylabel("gradient norm (log scale)")
    plt.title("Gradient norm by module")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "gradient_module_norms.png", dpi=180)
    plt.close()


def summarize_distribution(rows):
    summary = {}
    for subject in sorted({row["subject"] for row in rows}):
        summary[str(subject)] = {}
        for kind in ["clean", "pgd"]:
            selected = [row for row in rows if row["subject"] == subject and row["kind"] == kind]
            summary[str(subject)][kind] = {
                "count": len(selected),
                "confidence_pass_rate": float(np.mean([row["pass_confidence"] for row in selected])) if selected else None,
                "avg_epoch_conf": float(np.mean([row["avg_epoch_conf"] for row in selected])) if selected else None,
                "rel_eog_delta_mean": float(np.mean([row["rel_eog_delta"] for row in selected])) if selected else None,
                "rel_eeg_delta_mean": float(np.mean([row["rel_eeg_delta"] for row in selected])) if selected else None,
            }
    return summary


def write_report(output_dir, payload):
    lines = [
        "# BrainUICL Attack Diagnostics",
        "",
        "生成日期：2026-07-07",
        "",
        "## 关键区分",
        "",
        "- `model_nhe` 是模型/损失级 white-box 攻击，不直接修改输入 EEG/EOG。full run 后期 buffer 不增长，是因为模型输出退化后无法产生足够高置信伪标签，不是因为某个 poisoned input 被过滤。",
        "- `pgd_nhe/pgd_ble` 才是输入级污染。PGD 扰动被限制在 `eps = batch_std * pgd_eps_scale` 内，并通过多步 sign-gradient 让输出靠近攻击目标。",
        "",
        "## 生成文件",
        "",
        "```text",
        "embedding_pca.png",
        "embedding_tsne.png",
        "signal_stats_pca.png",
        "signal_stats_tsne.png",
        "confidence_filter.png",
        "perturbation_magnitude.png",
        "gradient_cosine.png",
        "gradient_module_norms.png",
        "attack_diagnostics.json",
        "```",
        "",
        "## 置信度过滤摘要",
        "",
        "| subject | clean pass | pgd pass | clean avg conf | pgd avg conf | rel EOG delta | rel EEG delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for subject, row in payload["distribution_summary"].items():
        clean = row["clean"]
        pgd = row["pgd"]
        lines.append(
            f"| {subject} | {clean['confidence_pass_rate']:.3f} | {pgd['confidence_pass_rate']:.3f} | "
            f"{clean['avg_epoch_conf']:.3f} | {pgd['avg_epoch_conf']:.3f} | "
            f"{pgd['rel_eog_delta_mean']:.5f} | {pgd['rel_eeg_delta_mean']:.5f} |"
        )
    lines.extend([
        "",
        "## 梯度方向摘要",
        "",
        "梯度项含义：",
        "",
        "- `source_replay_ce`：历史/source replay 样本的监督 CE 梯度。",
        "- `new_pseudo_ce`：new subject 上由 teacher pseudo label 产生的正常 CL 梯度。",
        "- `model_nhe_kl`：当前 full run 使用的 `model_nhe` 攻击目标梯度。",
        "- `pgd_pseudo_ce`：PGD 污染输入进入 pseudo-label 更新后的梯度。",
        "",
        "Cosine 矩阵：",
        "",
        "| gradient | " + " | ".join(payload["gradient"]["names"]) + " |",
        "|---|" + "|".join(["---:" for _ in payload["gradient"]["names"]]) + "|",
    ])
    names = payload["gradient"]["names"]
    cosine = payload["gradient"]["cosine"]
    for i, name in enumerate(names):
        lines.append("| " + name + " | " + " | ".join(f"{cosine[i][j]:.3f}" for j in range(len(names))) + " |")
    lines.extend([
        "",
        "## 解释",
        "",
        "如果 `pgd pass` 接近 clean pass，说明输入级扰动仍然位于模型可接受的分布附近，攻击更隐蔽；如果明显低于 clean pass，则当前 PGD 过强或目标过偏，容易被置信度过滤发现。",
        "",
        "如果 `model_nhe_kl` 与 `new_pseudo_ce/source_replay_ce` 的 cosine 为负或很低，说明攻击更新方向与正常 CL 明显冲突；这可以解释为什么 full `model_nhe` 会导致模型快速退化和 buffer 停止增长。",
    ])
    (output_dir / "attack_diagnostics.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("experiments/rttdp_brainuicl_runs/full49_model_nhe_seed4321"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/attack_diagnostics/full49_model_nhe_seed4321"))
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--dataset", default="ISRUC")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--subjects", type=int, nargs="+", default=[64, 89, 1, 27])
    parser.add_argument("--max-seq-per-subject", type=int, default=16)
    parser.add_argument("--max-source-seq", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--confident-epoch-n", type=int, default=15)
    parser.add_argument("--pgd-attack-mode", choices=["pgd_nhe", "pgd_ble"], default="pgd_nhe")
    parser.add_argument("--pgd-steps", type=int, default=5)
    parser.add_argument("--pgd-eps-scale", type=float, default=0.10)
    parser.add_argument("--pgd-random-start", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    args.model_param = ModelConfig(args.dataset)

    split = json.loads((args.run_dir / "split.json").read_text())
    checkpoint_dir = args.run_dir / "clean" / "checkpoints" / "Pretrain"
    blocks = load_blocks(checkpoint_dir, args)

    rows = collect_distribution(args, blocks, args.subjects)
    pca_plot(rows, "embedding", args.output_dir / "embedding_pca.png", "Model embedding PCA: clean vs PGD")
    pca_plot(rows, "signal_stats", args.output_dir / "signal_stats_pca.png", "Raw signal statistics PCA: clean vs PGD")
    tsne_plot(rows, "embedding", args.output_dir / "embedding_tsne.png", "Model embedding t-SNE: clean vs PGD")
    tsne_plot(rows, "signal_stats", args.output_dir / "signal_stats_tsne.png", "Raw signal statistics t-SNE: clean vs PGD")
    confidence_plot(rows, args.output_dir / "confidence_filter.png")
    perturbation_plot(rows, args.output_dir / "perturbation_magnitude.png")

    grad = gradient_diagnostics(args, checkpoint_dir, split, args.subjects[0])
    gradient_plots(grad, args.output_dir)

    payload = {
        "config": {
            "subjects": args.subjects,
            "checkpoint_dir": str(checkpoint_dir),
            "pgd_attack_mode": args.pgd_attack_mode,
            "pgd_steps": args.pgd_steps,
            "pgd_eps_scale": args.pgd_eps_scale,
            "confidence": args.confidence,
            "confident_epoch_n": args.confident_epoch_n,
        },
        "distribution_summary": summarize_distribution(rows),
        "gradient": {
            "names": grad["names"],
            "cosine": grad["cosine"].tolist(),
            "norms": grad["norms"],
            "module_norms": grad["module_norms"],
        },
    }
    (args.output_dir / "attack_diagnostics.json").write_text(json.dumps(payload, indent=2))
    write_report(args.output_dir, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
