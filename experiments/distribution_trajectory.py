#!/usr/bin/env python3
"""
Visualize BrainUICL feature distribution trajectories across checkpoints.

The same clean EEG/EOG inputs are passed through Pretrain, clean CL, and attack
CL checkpoints. This separates raw input distribution from model-induced feature
distribution drift.
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
from sklearn.manifold import TSNE  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.attack_diagnostics import (  # noqa: E402
    SequenceDataset,
    load_blocks,
    merge_subject_paths_limited,
    model_embeddings,
    signal_stats,
)
from utils.config import ModelConfig  # noqa: E402


def checkpoint_plan(run_dir: Path, clean_run_dir: Path, attack_variant: str):
    candidates = [
        ("pretrain", "base", clean_run_dir / "clean" / "checkpoints" / "Pretrain"),
        ("clean_10", "clean", clean_run_dir / "clean" / "checkpoints" / "individual_10"),
        ("clean_25", "clean", clean_run_dir / "clean" / "checkpoints" / "individual_25"),
        ("clean_49", "clean", clean_run_dir / "clean" / "checkpoints" / "individual_49"),
        ("attack_10", "attack", run_dir / attack_variant / "checkpoints" / "individual_10"),
        ("attack_25", "attack", run_dir / attack_variant / "checkpoints" / "individual_25"),
        ("attack_49", "attack", run_dir / attack_variant / "checkpoints" / "individual_49"),
    ]
    return [(name, branch, path) for name, branch, path in candidates if path.exists()]


def collect_group(paths, group_name, args):
    loader = DataLoader(SequenceDataset(paths), batch_size=args.batch, shuffle=False, num_workers=0)
    rows = []
    for eog, eeg, _ in loader:
        eog = eog.to(args.device)
        eeg = eeg.to(args.device)
        stats = signal_stats(eog, eeg).detach().cpu().numpy()
        for idx in range(stats.shape[0]):
            rows.append({"group": group_name, "raw_stats": stats[idx]})
    return rows


def collect_embeddings(args, split):
    old_subjects = split["old_generalization"][: args.group_subjects]
    new_subjects = split["new_order"][: args.group_subjects]
    source_subjects = split["train"][: args.group_subjects]
    groups = {
        "source_train": merge_subject_paths_limited(args.data_root, source_subjects, args.per_subject_limit),
        "old_generalization": merge_subject_paths_limited(args.data_root, old_subjects, args.per_subject_limit),
        "new_order": merge_subject_paths_limited(args.data_root, new_subjects, args.per_subject_limit),
    }

    raw_rows = []
    for group_name, paths in groups.items():
        raw_rows.extend(collect_group(paths, group_name, args))

    feature_rows = []
    centroids = {}
    for state_name, branch, checkpoint_dir in checkpoint_plan(args.run_dir, args.clean_run_dir, args.attack_variant):
        blocks = load_blocks(checkpoint_dir, args)
        for block in blocks:
            block.eval()
        for group_name, paths in groups.items():
            loader = DataLoader(SequenceDataset(paths), batch_size=args.batch, shuffle=False, num_workers=0)
            embeddings = []
            with torch.no_grad():
                for eog, eeg, _ in loader:
                    eog = eog.to(args.device)
                    eeg = eeg.to(args.device)
                    emb = model_embeddings(blocks, eog, eeg, args).detach().cpu().numpy()
                    embeddings.append(emb)
            matrix = np.concatenate(embeddings, axis=0)
            centroids[f"{state_name}:{group_name}"] = matrix.mean(axis=0)
            for row in matrix:
                feature_rows.append(
                    {
                        "state": state_name,
                        "branch": branch,
                        "group": group_name,
                        "embedding": row,
                    }
                )
    return raw_rows, feature_rows, centroids, old_subjects, new_subjects, source_subjects


def tsne(matrix):
    perplexity = max(5, min(30, (matrix.shape[0] - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=0,
    ).fit_transform(matrix)


def plot_raw_tsne(raw_rows, out_path):
    matrix = np.stack([row["raw_stats"] for row in raw_rows])
    points = tsne(matrix)
    groups = sorted({row["group"] for row in raw_rows})
    colors = {"source_train": "#4C78A8", "old_generalization": "#54A24B", "new_order": "#E45756"}
    plt.figure(figsize=(7, 5.5))
    for group in groups:
        idx = [i for i, row in enumerate(raw_rows) if row["group"] == group]
        plt.scatter(points[idx, 0], points[idx, 1], s=42, alpha=0.78, color=colors.get(group), label=group)
    plt.title("Raw signal-stat t-SNE: input distribution")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_feature_tsne(feature_rows, out_path):
    matrix = np.stack([row["embedding"] for row in feature_rows])
    points = tsne(matrix)
    states = ["pretrain", "clean_10", "clean_25", "clean_49", "attack_10", "attack_25", "attack_49"]
    state_colors = {
        "pretrain": "#222222",
        "clean_10": "#72B7B2",
        "clean_25": "#54A24B",
        "clean_49": "#2E7D32",
        "attack_10": "#FF9D2E",
        "attack_25": "#F58518",
        "attack_49": "#C43C2F",
    }
    group_markers = {"source_train": "o", "old_generalization": "s", "new_order": "^"}
    plt.figure(figsize=(9, 6.5))
    for state in states:
        for group, marker in group_markers.items():
            idx = [
                i
                for i, row in enumerate(feature_rows)
                if row["state"] == state and row["group"] == group
            ]
            if not idx:
                continue
            plt.scatter(
                points[idx, 0],
                points[idx, 1],
                s=32,
                alpha=0.55,
                marker=marker,
                color=state_colors[state],
                label=f"{state}/{group}",
            )
    plt.title("Feature t-SNE across checkpoints: same clean inputs")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(ncol=3, fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_centroid_shift(centroids, out_path):
    states = ["pretrain", "clean_10", "clean_25", "clean_49", "attack_10", "attack_25", "attack_49"]
    groups = ["source_train", "old_generalization", "new_order"]
    state_x = {state: idx for idx, state in enumerate(states)}
    branch_color = {"clean": "#2E7D32", "attack": "#C43C2F", "base": "#222222"}
    plt.figure(figsize=(9, 5))
    table = {}
    for group in groups:
        base = centroids.get(f"pretrain:{group}")
        if base is None:
            continue
        values = []
        xs = []
        labels = []
        for state in states:
            key = f"{state}:{group}"
            if key not in centroids:
                continue
            dist = float(np.linalg.norm(centroids[key] - base))
            table[key] = dist
            values.append(dist)
            xs.append(state_x[state])
            labels.append(state)
        plt.plot(xs, values, marker="o", linewidth=2, label=group)
    plt.xticks(range(len(states)), states, rotation=25, ha="right")
    plt.ylabel("centroid L2 distance from pretrain")
    plt.title("Feature centroid shift from pretrain")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return table


def plot_old_new_distance(centroids, out_path):
    states = ["pretrain", "clean_10", "clean_25", "clean_49", "attack_10", "attack_25", "attack_49"]
    distances = []
    labels = []
    for state in states:
        old_key = f"{state}:old_generalization"
        new_key = f"{state}:new_order"
        if old_key in centroids and new_key in centroids:
            distances.append(float(np.linalg.norm(centroids[old_key] - centroids[new_key])))
            labels.append(state)
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(len(labels)), distances, marker="o", linewidth=2, color="#4C78A8")
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("old/new centroid L2 distance")
    plt.title("Old vs new feature separation across checkpoints")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return dict(zip(labels, distances))


def write_report(args, subjects, centroid_shift, old_new_distance):
    old_subjects, new_subjects, source_subjects = subjects
    lines = [
        "# BrainUICL Distribution Trajectory",
        "",
        "生成日期：2026-07-07",
        "",
        "## 目的",
        "",
        "同一批 clean EEG/EOG 输入分别送入 Pretrain、clean CL、attack CL checkpoints，观察表征空间如何变化。这样可以区分：",
        "",
        "- 原始输入分布是否不同；",
        "- 特征提取后分布是否开始明显变化；",
        "- clean CL 和 attack CL 对同一输入的表征漂移是否不同。",
        "",
        "## Subjects",
        "",
        f"- source_train: {source_subjects}",
        f"- old_generalization: {old_subjects}",
        f"- new_order: {new_subjects}",
        "",
        "## 输出图像",
        "",
        "```text",
        "raw_signal_tsne.png",
        "feature_tsne_by_checkpoint.png",
        "centroid_shift_from_pretrain.png",
        "old_new_distance_by_checkpoint.png",
        "```",
        "",
        "## Centroid shift from pretrain",
        "",
        "| state/group | L2 distance |",
        "|---|---:|",
    ]
    for key, value in sorted(centroid_shift.items()):
        lines.append(f"| {key} | {value:.4f} |")
    lines.extend([
        "",
        "## Old/New distance",
        "",
        "| state | old-new centroid distance |",
        "|---|---:|",
    ])
    for key, value in old_new_distance.items():
        lines.append(f"| {key} | {value:.4f} |")
    lines.extend([
        "",
        "## 如何解读",
        "",
        "- `raw_signal_tsne.png` 只看输入统计，不受 checkpoint 影响；如果 clean/PGD 在这里很近，说明原始数据空间偏移不明显。",
        "- `feature_tsne_by_checkpoint.png` 看模型提取后的 embedding；同一输入在不同 checkpoint 下位置变化，说明变化来自模型表征而不是输入本身。",
        "- `centroid_shift_from_pretrain.png` 量化 CL 过程中表征中心相对 pretrain 的漂移；attack 曲线如果明显更大，说明攻击主要拉偏模型表征。",
        "- `old_new_distance_by_checkpoint.png` 看 old/new subject 表征距离是否被 CL 或 attack 放大。",
    ])
    (args.output_dir / "distribution_trajectory.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("experiments/rttdp_brainuicl_runs/full49_model_nhe_seed4321"))
    parser.add_argument("--clean-run-dir", type=Path, default=None)
    parser.add_argument("--attack-variant", default="attack_model_nhe")
    parser.add_argument("--data-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/distribution_trajectory/full49_model_nhe_seed4321"))
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--dataset", default="ISRUC")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--group-subjects", type=int, default=4)
    parser.add_argument("--per-subject-limit", type=int, default=6)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.clean_run_dir is None:
        args.clean_run_dir = args.run_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    args.model_param = ModelConfig(args.dataset)

    split = json.loads((args.run_dir / "split.json").read_text())
    raw_rows, feature_rows, centroids, old_subjects, new_subjects, source_subjects = collect_embeddings(args, split)
    plot_raw_tsne(raw_rows, args.output_dir / "raw_signal_tsne.png")
    plot_feature_tsne(feature_rows, args.output_dir / "feature_tsne_by_checkpoint.png")
    centroid_shift = plot_centroid_shift(centroids, args.output_dir / "centroid_shift_from_pretrain.png")
    old_new_distance = plot_old_new_distance(centroids, args.output_dir / "old_new_distance_by_checkpoint.png")
    payload = {
        "source_subjects": source_subjects,
        "old_subjects": old_subjects,
        "new_subjects": new_subjects,
        "centroid_shift_from_pretrain": centroid_shift,
        "old_new_distance": old_new_distance,
    }
    (args.output_dir / "distribution_trajectory.json").write_text(json.dumps(payload, indent=2))
    write_report(args, (old_subjects, new_subjects, source_subjects), centroid_shift, old_new_distance)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
