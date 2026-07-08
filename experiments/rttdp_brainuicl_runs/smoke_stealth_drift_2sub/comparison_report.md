# BrainUICL RTTDP-style CL Comparison

This run uses the same new-individual order for clean and attacked variants.

## Config

```json
{
  "seed": 4321,
  "data_root": "/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32",
  "input_checkpoint_root": "/home/undefined/Disk/ai-storage/BrainUICL/model_parameter",
  "new_order": [
    64,
    89
  ],
  "max_subjects": 2,
  "ssl_epoch": 1,
  "incremental_epoch": 1,
  "attack_mode": "stealth_drift"
}
```

## Final Stability

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.7139 | 0.7017 | 0.7110 | 0.6971 | 0.0163 |
| attack | 0.6599 | 0.6311 | 0.6838 | 0.6618 | 0.0605 |

## Final Plasticity

| variant | initial ACC | before ACC | after ACC | initial MF1 | before MF1 | after MF1 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.7650 | 0.7490 | 0.7340 | 0.7479 | 0.7359 | 0.7191 |
| attack | 0.7650 | 0.7040 | 0.6520 | 0.7479 | 0.6937 | 0.6350 |

## Stability Curves

```text
clean ACC:  [0.7024550898203593, 0.7165868263473054, 0.7138922155688623]
attack ACC: [0.7024550898203593, 0.6889820359281437, 0.6599401197604791]
clean MF1:  [0.6880332943309935, 0.7015982808204975, 0.7016869249580776]
attack MF1: [0.6880332943309935, 0.6661852590283541, 0.6310714824408311]
```
