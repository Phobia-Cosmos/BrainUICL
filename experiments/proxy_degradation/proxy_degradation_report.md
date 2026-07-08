# BrainUICL Proxy Degradation Probe
This is an independent probe script. It does not modify the original BrainUICL CL code.
## Config
```json
{
  "data_root": "/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32",
  "checkpoint_root": "/home/undefined/Disk/ai-storage/BrainUICL/model_parameter",
  "seed": 4321,
  "device": "cpu",
  "batch": 4,
  "candidate_count": 6,
  "sequential_k": 2,
  "update_batches": 4,
  "eval_max_batches": 0,
  "attack_lr": 8e-05,
  "benign_lr": 8e-05,
  "source_weight": 0.2,
  "target_weight": 1.0
}
```
## Split
```json
{
  "train": [
    7,
    16,
    18,
    23,
    24,
    28,
    30,
    34,
    35,
    37,
    38,
    41,
    45,
    48,
    50,
    53,
    69,
    71,
    74,
    78,
    79,
    82,
    93,
    94
  ],
  "val": [
    12,
    21,
    29,
    58,
    76,
    77
  ],
  "old_generalization": [
    6,
    14,
    32,
    39,
    43,
    44,
    51,
    56,
    59,
    62,
    67,
    68,
    72,
    73,
    75,
    88,
    90,
    92,
    100
  ],
  "new_order_prefix": [
    64,
    89,
    1,
    27,
    60,
    5
  ],
  "candidate_subjects": [
    64,
    89,
    1,
    27,
    60,
    5
  ],
  "selected_proxy_order": [
    64,
    5
  ],
  "natural_order": [
    64,
    89
  ]
}
```
## Baseline Old/Generalization Metrics
```json
{
  "acc": 0.7023952095808383,
  "mf1": 0.6879558256933758,
  "entropy": 0.2166995297871628,
  "confidence": 0.9176514566372969,
  "n_epochs": 16700
}
```
## Candidate Proxy Harmfulness
| rank | subject | old ACC drop | old MF1 drop | M0 subject ACC | M0 subject MF1 | entropy | confidence |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.1387 | 0.1684 | 0.7291 | 0.7631 | 0.2574 | 0.8911 |
| 2 | 5 | 0.1163 | 0.1386 | 0.3107 | 0.1130 | 0.7258 | 0.7454 |
| 3 | 27 | 0.0972 | 0.1118 | 0.7170 | 0.6478 | 0.1161 | 0.9557 |
| 4 | 89 | 0.0826 | 0.0942 | 0.8010 | 0.7326 | 0.1504 | 0.9412 |
| 5 | 1 | 0.0573 | 0.0743 | 0.2000 | 0.0918 | 0.6887 | 0.7365 |
| 6 | 60 | 0.0420 | 0.0588 | 0.7068 | 0.6199 | 0.1704 | 0.9349 |

## Curves
### benign_natural_curve
| step | subject | ACC | MF1 | entropy | confidence |
|---:|---:|---:|---:|---:|---:|
| 0 |  | 0.7024 | 0.6880 | 0.2167 | 0.9177 |
| 1 | 64 | 0.6815 | 0.6694 | 0.2055 | 0.9217 |
| 2 | 89 | 0.6912 | 0.6697 | 0.2320 | 0.9127 |

### benign_selected_curve
| step | subject | ACC | MF1 | entropy | confidence |
|---:|---:|---:|---:|---:|---:|
| 0 |  | 0.7024 | 0.6880 | 0.2167 | 0.9177 |
| 1 | 64 | 0.6916 | 0.6575 | 0.1860 | 0.9309 |
| 2 | 5 | 0.6468 | 0.5941 | 0.2118 | 0.9175 |

### attack_selected_curve
| step | subject | ACC | MF1 | entropy | confidence |
|---:|---:|---:|---:|---:|---:|
| 0 |  | 0.7024 | 0.6880 | 0.2167 | 0.9177 |
| 1 | 64 | 0.5443 | 0.4914 | 0.5234 | 0.8003 |
| 2 | 5 | 0.3786 | 0.2739 | 0.9563 | 0.6101 |

## Quick Interpretation
- Attack-selected final old/generalization ACC drop: 0.3238; MF1 drop: 0.4141.
- Benign updates on the same selected subjects final ACC drop: 0.0556; MF1 drop: 0.0939.
- A larger attack drop than benign drop is preliminary evidence that a proxy objective can drive CL degradation.
