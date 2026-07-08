#!/usr/bin/env python3
"""Finalize an attack-only BrainUICL run against an existing clean baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.rttdp_brainuicl_full import build_blocks, evaluate, make_loader  # noqa: E402
from utils.config import ModelConfig  # noqa: E402


MODULE_FILES = {
    "feature_extractor": "feature_extractor_parameter_{seed}.pkl",
    "feature_encoder": "feature_encoder_parameter_{seed}.pkl",
    "sleep_classifier": "sleep_classifier_parameter_{seed}.pkl",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def load_blocks_from(checkpoint_dir: Path, args):
    blocks = build_blocks(args)
    for block, module in zip(blocks, MODULE_FILES):
        file_name = MODULE_FILES[module].format(seed=args.seed)
        block.load_state_dict(torch.load(checkpoint_dir / file_name, map_location=args.device))
        block.eval()
    return blocks


def final_from_metrics(metrics):
    stability = metrics["performance"]["stability"]
    return {
        "acc": stability["ACC"][-1],
        "mf1": stability["MF1"][-1],
        "aaa": stability["AAA"][-1],
        "aaf1": stability["AAF1"][-1],
        "fr": stability["FR"][-1],
    }


def low_point(metrics):
    stability = metrics["performance"]["stability"]
    acc = np.asarray(stability["ACC"], dtype=float)
    mf1 = np.asarray(stability["MF1"], dtype=float)
    return {
        "min_acc": float(acc.min()),
        "min_acc_step": int(acc.argmin()),
        "min_mf1": float(mf1.min()),
        "min_mf1_step": int(mf1.argmin()),
    }


def buffer_summary(metrics):
    rows = metrics["performance"].get("buffer", [])
    if not rows:
        return {"final_length": 0, "total_added": 0, "total_biased": 0, "mean_added": 0.0}
    return {
        "final_length": int(rows[-1]["length"]),
        "total_added": int(sum(row.get("added", 0) for row in rows)),
        "total_biased": int(sum(row.get("biased", 0) for row in rows)),
        "mean_added": float(np.mean([row.get("added", 0) for row in rows])),
    }


def diagnostics_summary(metrics):
    rows = metrics["performance"].get("attack_diagnostics", [])
    keys = [
        "clean_pass_rate",
        "adv_pass_rate",
        "accepted_rate",
        "mean_rel_eog",
        "mean_rel_eeg",
        "mean_feature_shift",
    ]
    summary = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return summary


def evaluate_groups(args, split, clean_checkpoint: Path, attack_checkpoint: Path):
    clean_blocks = load_blocks_from(clean_checkpoint, args)
    attack_blocks = load_blocks_from(attack_checkpoint, args)
    groups = {
        "old_generalization": split["old_generalization"],
        "new_order_all": split["new_order"],
        "source_train": split["train"],
        "validation": split["val"],
    }
    output = {}
    for group, subjects in groups.items():
        loader = make_loader(args.data_root, subjects, args.batch, shuffle=False, num_workers=args.num_worker)
        clean_result = evaluate(clean_blocks, loader, args)
        attack_result = evaluate(attack_blocks, loader, args)
        output[group] = {
            "subjects": [int(x) for x in subjects],
            "clean": {
                "acc": clean_result["acc"],
                "mf1": clean_result["mf1"],
                "n_epochs": clean_result["n_epochs"],
            },
            "attack": {
                "acc": attack_result["acc"],
                "mf1": attack_result["mf1"],
                "n_epochs": attack_result["n_epochs"],
            },
            "delta_attack_minus_clean": {
                "acc": attack_result["acc"] - clean_result["acc"],
                "mf1": attack_result["mf1"] - clean_result["mf1"],
            },
        }
    return output


def write_markdown(report, path: Path):
    lines = [
        "# BrainUICL Attack-only Finalization",
        "",
        "This report compares an attack-only run against an existing clean baseline with the same seed and new-individual order.",
        "",
        "## Final Stability From CL Metrics",
        "",
        "| variant | ACC | MF1 | AAA | AAF1 | FR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["clean", "attack"]:
        row = report[f"{name}_final"]
        lines.append(f"| {name} | {row['acc']:.4f} | {row['mf1']:.4f} | {row['aaa']:.4f} | {row['aaf1']:.4f} | {row['fr']:.4f} |")
    delta = report["final_delta_attack_minus_clean"]
    lines.extend(
        [
            f"| attack-clean | {delta['acc']:.4f} | {delta['mf1']:.4f} | {delta['aaa']:.4f} | {delta['aaf1']:.4f} | {delta['fr']:.4f} |",
            "",
            "## Final Checkpoint Evaluation",
            "",
            "| group | clean ACC | attack ACC | delta ACC | clean MF1 | attack MF1 | delta MF1 | epochs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group, row in report["final_checkpoint_eval"].items():
        lines.append(
            f"| {group} | {row['clean']['acc']:.4f} | {row['attack']['acc']:.4f} | "
            f"{row['delta_attack_minus_clean']['acc']:.4f} | {row['clean']['mf1']:.4f} | "
            f"{row['attack']['mf1']:.4f} | {row['delta_attack_minus_clean']['mf1']:.4f} | "
            f"{row['clean']['n_epochs']} |"
        )
    lines.extend(
        [
            "",
            "## Buffer",
            "",
            "| variant | final length | total added | total biased | mean added/subject |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ["clean", "attack"]:
        row = report[f"{name}_buffer"]
        lines.append(
            f"| {name} | {row['final_length']} | {row['total_added']} | "
            f"{row['total_biased']} | {row['mean_added']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Lowest Old-generalization Points",
            "",
            "| variant | min ACC | ACC step | min MF1 | MF1 step |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ["clean", "attack"]:
        row = report[f"{name}_low_point"]
        lines.append(f"| {name} | {row['min_acc']:.4f} | {row['min_acc_step']} | {row['min_mf1']:.4f} | {row['min_mf1_step']} |")
    lines.extend(["", "## Attack Diagnostics", "", "| metric | mean | min | max |", "|---|---:|---:|---:|"])
    for key, row in report["attack_diagnostics"].items():
        lines.append(f"| {key} | {row['mean']:.4f} | {row['min']:.4f} | {row['max']:.4f} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-run-dir", type=Path, required=True)
    parser.add_argument("--attack-run-dir", type=Path, required=True)
    parser.add_argument("--attack-variant", default="attack_stealth_loss_drift")
    parser.add_argument("--data-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32"))
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num-worker", type=int, default=4)
    parser.add_argument("--dataset", default="ISRUC")
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    args.model_param = ModelConfig(args.dataset)

    split = load_json(args.attack_run_dir / "split.json")
    clean_metrics = load_json(args.clean_run_dir / "clean" / "metrics.json")
    attack_metrics = load_json(args.attack_run_dir / args.attack_variant / "metrics.json")
    final_step = len(split["new_order"])
    clean_checkpoint = args.clean_run_dir / "clean" / "checkpoints" / f"individual_{final_step}"
    attack_checkpoint = args.attack_run_dir / args.attack_variant / "checkpoints" / f"individual_{final_step}"

    clean_final = final_from_metrics(clean_metrics)
    attack_final = final_from_metrics(attack_metrics)
    report = {
        "config": {
            "clean_run_dir": str(args.clean_run_dir),
            "attack_run_dir": str(args.attack_run_dir),
            "attack_variant": args.attack_variant,
            "seed": args.seed,
            "device": str(args.device),
            "final_step": final_step,
        },
        "clean_summary": clean_metrics["summary"],
        "attack_summary": attack_metrics["summary"],
        "clean_final": clean_final,
        "attack_final": attack_final,
        "final_delta_attack_minus_clean": {
            key: attack_final[key] - clean_final[key] for key in clean_final
        },
        "clean_low_point": low_point(clean_metrics),
        "attack_low_point": low_point(attack_metrics),
        "clean_buffer": buffer_summary(clean_metrics),
        "attack_buffer": buffer_summary(attack_metrics),
        "attack_diagnostics": diagnostics_summary(attack_metrics),
        "final_checkpoint_eval": evaluate_groups(args, split, clean_checkpoint, attack_checkpoint),
    }
    (args.attack_run_dir / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_markdown(report, args.attack_run_dir / "comparison_report.md")
    print(json.dumps(report["final_delta_attack_minus_clean"], indent=2))
    print(json.dumps(report["final_checkpoint_eval"], indent=2))


if __name__ == "__main__":
    main()
