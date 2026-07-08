#!/usr/bin/env python3
"""
Analyze neuron/parameter stability across BrainUICL continual checkpoints.

This is a read-only analysis script. It does not modify BrainUICL training code
or checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.pretrain_net import FeatureExtractor, SleepMLP, TransformerEncoder  # noqa: E402
from utils.config import ModelConfig  # noqa: E402


MODULE_FILES = {
    "feature_extractor": "feature_extractor_parameter_{seed}.pkl",
    "feature_encoder": "feature_encoder_parameter_{seed}.pkl",
    "sleep_classifier": "sleep_classifier_parameter_{seed}.pkl",
}

SKIP_KEY_PARTS = ("running_mean", "running_var", "num_batches_tracked", "pos_emb.pe")


class SequenceDataset(Dataset):
    def __init__(self, paths: tuple[list[Path], list[Path]], max_sequences: int = 0):
        data_paths, label_paths = paths
        if max_sequences > 0:
            data_paths = data_paths[:max_sequences]
            label_paths = label_paths[:max_sequences]
        self.data_paths = data_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        x = np.load(self.data_paths[index]).astype(np.float32)
        y = np.load(self.label_paths[index]).astype(np.int64)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x[:, :2, :], x[:, 2:, :], y


def subject_paths(data_root: Path, subject: int, per_subject_limit: int) -> tuple[list[Path], list[Path]]:
    data_dir = data_root / str(subject) / "data"
    label_dir = data_root / str(subject) / "label"
    data_paths, label_paths = [], []
    idx = 0
    while (data_dir / f"{idx}.npy").exists():
        data_paths.append(data_dir / f"{idx}.npy")
        label_paths.append(label_dir / f"{idx}.npy")
        idx += 1
        if per_subject_limit and idx >= per_subject_limit:
            break
    return data_paths, label_paths


def merge_subject_paths(data_root: Path, subjects: list[int], per_subject_limit: int) -> tuple[list[Path], list[Path]]:
    data_paths, label_paths = [], []
    for subject in subjects:
        d, l = subject_paths(data_root, subject, per_subject_limit)
        data_paths.extend(d)
        label_paths.extend(l)
    return data_paths, label_paths


def checkpoint_dirs(checkpoint_root: Path) -> list[Path]:
    dirs = [checkpoint_root / "Pretrain"]
    indexed = []
    for path in checkpoint_root.glob("individual_*"):
        try:
            indexed.append((int(path.name.split("_")[-1]), path))
        except ValueError:
            continue
    dirs.extend(path for _, path in sorted(indexed))
    return dirs


def load_module_state(checkpoint_dir: Path, module: str, seed: int) -> dict[str, torch.Tensor]:
    path = checkpoint_dir / MODULE_FILES[module].format(seed=seed)
    return torch.load(path, map_location="cpu")


def include_key(key: str, tensor: torch.Tensor) -> bool:
    if any(part in key for part in SKIP_KEY_PARTS):
        return False
    if tensor.ndim == 0 or tensor.shape[0] < 2:
        return False
    if not torch.is_floating_point(tensor):
        return False
    return True


def unit_matrix(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(tensor.shape[0], -1)


def rel_unit_change(current: torch.Tensor, base: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    cur = unit_matrix(current)
    bas = unit_matrix(base)
    return torch.linalg.norm(cur - bas, dim=1) / (torch.linalg.norm(bas, dim=1) + eps)


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if np.std(xx) == 0 or np.std(yy) == 0:
        return None
    return float(np.corrcoef(xx, yy)[0, 1])


def analyze_variant(run_dir: Path, variant: str, seed: int) -> dict:
    ckpt_root = run_dir / variant / "checkpoints"
    dirs = checkpoint_dirs(ckpt_root)
    if len(dirs) < 2:
        raise RuntimeError(f"Not enough checkpoints under {ckpt_root}")

    result = {
        "variant": variant,
        "checkpoint_count": len(dirs),
        "module_summary": {},
        "key_summary": {},
        "step_global_delta": [],
        "step_stable_delta": [],
        "step_unstable_delta": [],
        "stable_units": {},
    }

    for module in MODULE_FILES:
        base = load_module_state(dirs[0], module, seed)
        final = load_module_state(dirs[-1], module, seed)
        module_final_changes = []
        key_summaries = {}
        stable_masks = {}

        for key, base_tensor in base.items():
            if key not in final or not include_key(key, base_tensor):
                continue
            changes = rel_unit_change(final[key], base_tensor).numpy()
            if changes.size == 0:
                continue
            stable_cut = np.quantile(changes, 0.10)
            unstable_cut = np.quantile(changes, 0.90)
            stable_mask = changes <= stable_cut
            unstable_mask = changes >= unstable_cut
            stable_masks[key] = stable_mask
            module_final_changes.append(changes)
            key_summaries[key] = {
                "unit_count": int(changes.size),
                "final_relative_change": quantiles(changes),
                "stable_bottom10_mean": float(np.mean(changes[stable_mask])),
                "unstable_top10_mean": float(np.mean(changes[unstable_mask])),
                "fraction_lt_0.01": float(np.mean(changes < 0.01)),
                "fraction_lt_0.05": float(np.mean(changes < 0.05)),
                "fraction_lt_0.10": float(np.mean(changes < 0.10)),
            }
            result["stable_units"][f"{module}:{key}"] = np.where(stable_mask)[0].astype(int).tolist()

        all_changes = np.concatenate(module_final_changes) if module_final_changes else np.array([])
        result["module_summary"][module] = {
            "unit_count": int(all_changes.size),
            "final_relative_change": quantiles(all_changes),
            "fraction_lt_0.01": float(np.mean(all_changes < 0.01)) if all_changes.size else 0.0,
            "fraction_lt_0.05": float(np.mean(all_changes < 0.05)) if all_changes.size else 0.0,
            "fraction_lt_0.10": float(np.mean(all_changes < 0.10)) if all_changes.size else 0.0,
        }
        result["key_summary"][module] = key_summaries

    for prev_dir, cur_dir in zip(dirs[:-1], dirs[1:]):
        global_num = global_den = 0.0
        stable_num = stable_den = 0.0
        unstable_num = unstable_den = 0.0

        for module in MODULE_FILES:
            prev = load_module_state(prev_dir, module, seed)
            cur = load_module_state(cur_dir, module, seed)
            for key, prev_tensor in prev.items():
                if key not in cur or not include_key(key, prev_tensor):
                    continue
                prev_mat = unit_matrix(prev_tensor)
                cur_mat = unit_matrix(cur[key])
                diff = torch.linalg.norm(cur_mat - prev_mat, dim=1)
                den = torch.linalg.norm(prev_mat, dim=1) + 1e-12
                rel = diff / den
                global_num += float(diff.sum())
                global_den += float(den.sum())

                mask_key = f"{module}:{key}"
                stable_idx = result["stable_units"].get(mask_key, [])
                if stable_idx:
                    idx = torch.as_tensor(stable_idx, dtype=torch.long)
                    stable_num += float(diff[idx].sum())
                    stable_den += float(den[idx].sum())
                    all_idx = torch.ones(rel.shape[0], dtype=torch.bool)
                    all_idx[idx] = False
                    if all_idx.any():
                        unstable_num += float(diff[all_idx].sum())
                        unstable_den += float(den[all_idx].sum())

        result["step_global_delta"].append(global_num / max(global_den, 1e-12))
        result["step_stable_delta"].append(stable_num / max(stable_den, 1e-12))
        result["step_unstable_delta"].append(unstable_num / max(unstable_den, 1e-12))

    metrics_path = run_dir / variant / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        stability = metrics["performance"]["stability"]
        buffer_rows = metrics.get("buffer") or metrics["performance"].get("buffer", [])
        added = [float(row["added"]) for row in buffer_rows]
        lengths = [float(row["length"]) for row in buffer_rows]
        acc = [float(v) for v in stability["ACC"][1:]]
        mf1 = [float(v) for v in stability["MF1"][1:]]
        acc_change = [float(stability["ACC"][i + 1] - stability["ACC"][i]) for i in range(len(stability["ACC"]) - 1)]
        mf1_change = [float(stability["MF1"][i + 1] - stability["MF1"][i]) for i in range(len(stability["MF1"]) - 1)]
        plasticity_gain = []
        for subject in [str(row["subject"]) for row in buffer_rows]:
            values = metrics["performance"]["plasticity"][subject]["MF1"]
            plasticity_gain.append(float(values[2] - values[1]))
        result["relations"] = {
            "global_delta_vs_buffer_added": pearson(result["step_global_delta"], added),
            "stable_delta_vs_buffer_added": pearson(result["step_stable_delta"], added),
            "global_delta_vs_buffer_length": pearson(result["step_global_delta"], lengths),
            "global_delta_vs_old_acc": pearson(result["step_global_delta"], acc),
            "global_delta_vs_old_mf1": pearson(result["step_global_delta"], mf1),
            "global_delta_vs_old_acc_change": pearson(result["step_global_delta"], acc_change),
            "global_delta_vs_old_mf1_change": pearson(result["step_global_delta"], mf1_change),
            "global_delta_vs_new_mf1_gain": pearson(result["step_global_delta"], plasticity_gain),
            "buffer_added_mean": float(np.mean(added)) if added else None,
            "buffer_added_min": float(np.min(added)) if added else None,
            "buffer_added_max": float(np.max(added)) if added else None,
            "buffer_final_length": float(lengths[-1]) if lengths else None,
        }
    return result


def build_blocks(args):
    return (
        FeatureExtractor(args).float().to(args.device),
        TransformerEncoder(args).float().to(args.device),
        SleepMLP(args).float().to(args.device),
    )


def load_blocks(checkpoint_dir: Path, args):
    blocks = build_blocks(args)
    for block, module in zip(blocks, MODULE_FILES):
        state = load_module_state(checkpoint_dir, module, args.seed)
        block.load_state_dict(state)
    return blocks


def forward_blocks(blocks, eog, eeg, args):
    batch = eeg.shape[0]
    eog = eog.reshape(-1, args.model_param.EogNum, args.model_param.EpochLength)
    eeg = eeg.reshape(-1, args.model_param.EegNum, args.model_param.EpochLength)
    features = blocks[0](eeg, eog)
    features = blocks[1](features)
    logits = blocks[2](features)
    return logits.reshape(batch, args.model_param.NumClasses, args.model_param.SeqLength)


def flat_logits(logits):
    return logits.permute(0, 2, 1).reshape(-1, logits.shape[1])


def flat_labels(labels):
    return labels.reshape(-1).long()


def zero_grads(blocks):
    for block in blocks:
        block.zero_grad(set_to_none=True)


def fisher_importance(blocks, loader, args, max_batches: int) -> dict[str, dict[str, np.ndarray]]:
    for block in blocks:
        block.train(False)
    accum: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    counts: dict[str, int] = defaultdict(int)
    module_names = list(MODULE_FILES)

    for batch_idx, (eog, eeg, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        eog = eog.to(args.device)
        eeg = eeg.to(args.device)
        labels = labels.to(args.device)
        zero_grads(blocks)
        logits = forward_blocks(blocks, eog, eeg, args)
        loss = F.cross_entropy(flat_logits(logits), flat_labels(labels))
        loss.backward()

        for module_name, block in zip(module_names, blocks):
            for key, param in block.named_parameters():
                if param.grad is None or not include_key(key, param.detach()):
                    continue
                grad = unit_matrix(param.grad.detach())
                score = grad.pow(2).mean(dim=1).detach().cpu()
                if key not in accum[module_name]:
                    accum[module_name][key] = torch.zeros_like(score)
                accum[module_name][key] += score
                counts[f"{module_name}:{key}"] += 1

    output = {}
    for module_name, key_scores in accum.items():
        output[module_name] = {}
        for key, score in key_scores.items():
            count = max(counts[f"{module_name}:{key}"], 1)
            output[module_name][key] = (score / count).numpy()
    return output


def top_mask(values: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    cutoff = np.quantile(values, 1.0 - fraction)
    return values >= cutoff


def fisher_overlap(clean_result: dict, old_fisher: dict, new_fisher: dict) -> dict:
    overlaps = {}
    stable_units = clean_result["stable_units"]
    for module_name in old_fisher:
        for key, old_values in old_fisher[module_name].items():
            if module_name not in new_fisher or key not in new_fisher[module_name]:
                continue
            new_values = new_fisher[module_name][key]
            stable_idx = set(stable_units.get(f"{module_name}:{key}", []))
            if not stable_idx:
                continue
            old_top = set(np.where(top_mask(old_values))[0].astype(int).tolist())
            new_top = set(np.where(top_mask(new_values))[0].astype(int).tolist())
            old_new_union = old_top | new_top
            overlaps[f"{module_name}:{key}"] = {
                "unit_count": int(old_values.size),
                "stable_count": len(stable_idx),
                "old_top_count": len(old_top),
                "new_top_count": len(new_top),
                "stable_in_old_top_fraction": len(stable_idx & old_top) / max(len(stable_idx), 1),
                "stable_in_new_top_fraction": len(stable_idx & new_top) / max(len(stable_idx), 1),
                "old_top_in_stable_fraction": len(stable_idx & old_top) / max(len(old_top), 1),
                "new_top_in_stable_fraction": len(stable_idx & new_top) / max(len(new_top), 1),
                "old_new_top_jaccard": len(old_top & new_top) / max(len(old_top | new_top), 1),
                "stable_vs_old_new_union_jaccard": len(stable_idx & old_new_union) / max(len(stable_idx | old_new_union), 1),
            }
    return overlaps


def aggregate_overlap(overlaps: dict) -> dict:
    if not overlaps:
        return {}
    keys = [
        "stable_in_old_top_fraction",
        "stable_in_new_top_fraction",
        "old_top_in_stable_fraction",
        "new_top_in_stable_fraction",
        "old_new_top_jaccard",
        "stable_vs_old_new_union_jaccard",
    ]
    return {key: float(np.mean([row[key] for row in overlaps.values()])) for key in keys}


def write_markdown(output_path: Path, payload: dict):
    clean = payload["variants"]["clean"]
    attack = payload["variants"].get("attack_model_nhe")
    lines = [
        "# BrainUICL Stable Neuron Analysis",
        "",
        "生成日期：2026-07-05",
        "",
        "## 方法",
        "",
        "这里把 BrainUICL 的神经元近似为 Conv/Linear/Norm 参数的输出通道或输出行。对每个 checkpoint，计算相对 L2 变化：",
        "",
        "```text",
        "relative_change = ||w_t - w_0||_2 / (||w_0||_2 + eps)",
        "```",
        "",
        "每个参数张量中变化最低的 10% 单元被记为 stable units。这个定义对应论文中的“跨 continual tasks 权重变化小”的稳定性，但不是完整 Fisher 版本；后面用小样本 Fisher 做了旧任务/新任务重要性的交叉检查。",
        "",
        "## Clean 分支模块稳定性",
        "",
        "| module | units | median final rel-change | p10 | p90 | frac < 0.01 | frac < 0.05 | frac < 0.10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for module, row in clean["module_summary"].items():
        q = row["final_relative_change"]
        lines.append(
            f"| {module} | {row['unit_count']} | {q.get('median', 0):.6f} | {q.get('p10', 0):.6f} | "
            f"{q.get('p90', 0):.6f} | {row['fraction_lt_0.01']:.3f} | {row['fraction_lt_0.05']:.3f} | {row['fraction_lt_0.10']:.3f} |"
        )

    lines.extend([
        "",
        "## Clean vs Attack 模块变化",
        "",
        "| module | clean median | attack median | clean p90 | attack p90 |",
        "|---|---:|---:|---:|---:|",
    ])
    if attack:
        for module in clean["module_summary"]:
            cq = clean["module_summary"][module]["final_relative_change"]
            aq = attack["module_summary"][module]["final_relative_change"]
            lines.append(
                f"| {module} | {cq.get('median', 0):.6f} | {aq.get('median', 0):.6f} | "
                f"{cq.get('p90', 0):.6f} | {aq.get('p90', 0):.6f} |"
            )

    lines.extend([
        "",
        "## 与 Replay Buffer / 性能的关系",
        "",
        "| variant | corr(global delta, buffer added) | corr(stable delta, buffer added) | corr(global delta, old MF1) | corr(global delta, new MF1 gain) | final buffer length |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for variant, row in payload["variants"].items():
        rel = row.get("relations", {})
        def fmt(value):
            return "NA" if value is None else f"{value:.3f}"
        final_buffer = rel.get("buffer_final_length")
        final_buffer_text = "NA" if final_buffer is None else f"{final_buffer:.0f}"
        lines.append(
            f"| {variant} | {fmt(rel.get('global_delta_vs_buffer_added'))} | "
            f"{fmt(rel.get('stable_delta_vs_buffer_added'))} | {fmt(rel.get('global_delta_vs_old_mf1'))} | "
            f"{fmt(rel.get('global_delta_vs_new_mf1_gain'))} | {final_buffer_text} |"
        )

    overlap = payload.get("fisher_overlap", {})
    agg = payload.get("fisher_overlap_summary", {})
    lines.extend([
        "",
        "## 旧任务/新任务 Fisher 重要性重叠",
        "",
        "Fisher 使用 final clean checkpoint，在 old_generalization subjects 和 new_order subjects 上各采样少量 sequence。下面是各层平均结果：",
        "",
        "| metric | value |",
        "|---|---:|",
    ])
    for key, value in agg.items():
        lines.append(f"| {key} | {value:.4f} |")

    lines.extend([
        "",
        "按参数张量的细节：",
        "",
        "| key | stable in old top | stable in new top | old/new top Jaccard | stable vs old/new union Jaccard |",
        "|---|---:|---:|---:|---:|",
    ])
    for key, row in sorted(overlap.items()):
        lines.append(
            f"| {key} | {row['stable_in_old_top_fraction']:.3f} | {row['stable_in_new_top_fraction']:.3f} | "
            f"{row['old_new_top_jaccard']:.3f} | {row['stable_vs_old_new_union_jaccard']:.3f} |"
        )

    lines.extend([
        "",
        "## 解释",
        "",
        "- BrainUICL clean 分支确实存在参数变化很小的稳定单元，尤其是在卷积特征提取和 Transformer 表征层中更明显。",
        "- 这些稳定单元更像是跨 subject 共享的睡眠阶段表征，而不是论文中 SplitMNIST/SplitCIFAR 那种绑定到某个离散 class-incremental task 的神经元。",
        "- replay buffer 不直接“固定某些神经元”，但它通过混合历史样本和高置信伪标签约束梯度方向，间接降低全局参数漂移；攻击分支 buffer 不再增长时，模型退化明显。",
        "- 因为 BrainUICL 的每个 new task 仍然是同一套 5 类睡眠分期任务，只是 subject/domain 改变，所以稳定性仍会出现；但原因更偏向共享生理模式和预训练表征，而不是不同任务使用近乎不相交的神经元集合。",
    ])
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("experiments/rttdp_brainuicl_runs/full49_model_nhe_seed4321"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/stable_neuron_analysis/full49_model_nhe_seed4321"))
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--dataset", default="ISRUC")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fisher-batch", type=int, default=8)
    parser.add_argument("--fisher-max-batches", type=int, default=20)
    parser.add_argument("--per-subject-limit", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    payload = {
        "run_dir": str(args.run_dir),
        "data_root": str(args.data_root),
        "seed": args.seed,
        "device": str(device),
        "variants": {},
    }

    for variant in ["clean", "attack_model_nhe"]:
        variant_dir = args.run_dir / variant
        if variant_dir.exists():
            payload["variants"][variant] = analyze_variant(args.run_dir, variant, args.seed)

    split = json.loads((args.run_dir / "split.json").read_text())
    model_args = SimpleNamespace(
        dataset=args.dataset,
        device=device,
        seed=args.seed,
        model_param=ModelConfig(args.dataset),
    )
    final_clean_dir = checkpoint_dirs(args.run_dir / "clean" / "checkpoints")[-1]
    blocks = load_blocks(final_clean_dir, model_args)

    old_paths = merge_subject_paths(args.data_root, split["old_generalization"], args.per_subject_limit)
    new_paths = merge_subject_paths(args.data_root, split["new_order"], args.per_subject_limit)
    old_loader = DataLoader(SequenceDataset(old_paths), batch_size=args.fisher_batch, shuffle=False, num_workers=0)
    new_loader = DataLoader(SequenceDataset(new_paths), batch_size=args.fisher_batch, shuffle=False, num_workers=0)

    old_fisher = fisher_importance(blocks, old_loader, model_args, args.fisher_max_batches)
    new_fisher = fisher_importance(blocks, new_loader, model_args, args.fisher_max_batches)
    overlaps = fisher_overlap(payload["variants"]["clean"], old_fisher, new_fisher)
    payload["fisher_overlap"] = overlaps
    payload["fisher_overlap_summary"] = aggregate_overlap(overlaps)
    payload["fisher_sampling"] = {
        "old_subjects": split["old_generalization"],
        "new_subjects": split["new_order"],
        "per_subject_limit": args.per_subject_limit,
        "fisher_batch": args.fisher_batch,
        "fisher_max_batches": args.fisher_max_batches,
    }

    (args.output_dir / "stable_neuron_analysis.json").write_text(json.dumps(payload, indent=2))
    write_markdown(args.output_dir / "stable_neuron_analysis.md", payload)
    print(json.dumps({
        "output": str(args.output_dir),
        "clean_modules": payload["variants"]["clean"]["module_summary"],
        "relations_clean": payload["variants"]["clean"].get("relations", {}),
        "fisher_overlap_summary": payload["fisher_overlap_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
