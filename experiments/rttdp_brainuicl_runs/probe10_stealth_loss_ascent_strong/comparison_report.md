# BrainUICL Attack-only Finalization

This report compares an attack-only run against an existing clean baseline with the same seed and new-individual order.

## Final Stability From CL Metrics

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.6502 | 0.6032 | 0.7046 | 0.6828 | 0.0744 |
| attack | 0.6856 | 0.6557 | 0.7094 | 0.6881 | 0.0240 |
| attack-clean | 0.0354 | 0.0525 | 0.0048 | 0.0053 | -0.0504 |

## Final Checkpoint Evaluation

| group | clean ACC | attack ACC | delta ACC | clean MF1 | attack MF1 | delta MF1 | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_generalization | 0.6502 | 0.6856 | 0.0354 | 0.6032 | 0.6557 | 0.0525 | 16700 |
| new_order_all | 0.5255 | 0.5792 | 0.0537 | 0.4824 | 0.5431 | 0.0607 | 8880 |
| source_train | 0.6644 | 0.7046 | 0.0402 | 0.6357 | 0.6874 | 0.0517 | 20600 |
| validation | 0.6004 | 0.6375 | 0.0371 | 0.5567 | 0.6106 | 0.0539 | 5260 |

## Buffer

| variant | final length | total added | total biased | mean added/subject |
|---|---:|---:|---:|---:|
| clean | 1271 | 241 | 0 | 24.10 |
| attack | 1234 | 204 | 62 | 20.40 |

## Lowest Old-generalization Points

| variant | min ACC | ACC step | min MF1 | MF1 step |
|---|---:|---:|---:|---:|
| clean | 0.6502 | 10 | 0.6032 | 10 |
| attack | 0.6856 | 10 | 0.6557 | 10 |

## Attack Diagnostics

| metric | mean | min | max |
|---|---:|---:|---:|
| clean_pass_rate | 0.4438 | 0.0000 | 0.8485 |
| adv_pass_rate | 0.6072 | 0.0000 | 0.9432 |
| accepted_rate | 0.6072 | 0.0000 | 0.9432 |
| mean_rel_eog | 0.0191 | 0.0000 | 0.0319 |
| mean_rel_eeg | 0.0165 | 0.0000 | 0.0302 |
| mean_feature_shift | 25.9810 | 0.0000 | 47.8205 |
