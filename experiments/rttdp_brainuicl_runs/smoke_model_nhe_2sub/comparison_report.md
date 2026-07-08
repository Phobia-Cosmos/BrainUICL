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
  "attack_mode": "model_nhe"
}
```

## Final Stability

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.7156 | 0.7039 | 0.7108 | 0.6972 | 0.0188 |
| attack | 0.3198 | 0.2789 | 0.4733 | 0.4310 | 0.5447 |

## Final Plasticity

| variant | initial ACC | before ACC | after ACC | initial MF1 | before MF1 | after MF1 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.7650 | 0.7440 | 0.7295 | 0.7479 | 0.7327 | 0.7151 |
| attack | 0.7650 | 0.4885 | 0.2565 | 0.7479 | 0.4656 | 0.1835 |

## Stability Curves

```text
clean ACC:  [0.7023952095808383, 0.7143712574850299, 0.7155688622754491]
attack ACC: [0.7023952095808383, 0.3975449101796407, 0.31982035928143715]
clean MF1:  [0.6879558256933758, 0.6998043769078398, 0.7038879532433076]
attack MF1: [0.6879558256933758, 0.32603284017137046, 0.27890604301704147]
```
