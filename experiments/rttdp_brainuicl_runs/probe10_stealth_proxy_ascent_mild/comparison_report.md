# BrainUICL Attack-only Finalization

This report compares an attack-only run against an existing clean baseline with the same seed and new-individual order.

## Final Stability From CL Metrics

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.6502 | 0.6032 | 0.7046 | 0.6828 | 0.0744 |
| attack | 0.3726 | 0.2120 | 0.6094 | 0.5662 | 0.4696 |
| attack-clean | -0.2776 | -0.3912 | -0.0953 | -0.1166 | 0.3952 |

## Final Checkpoint Evaluation

| group | clean ACC | attack ACC | delta ACC | clean MF1 | attack MF1 | delta MF1 | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_generalization | 0.6502 | 0.3726 | -0.2776 | 0.6032 | 0.2120 | -0.3912 | 16700 |
| new_order_all | 0.5255 | 0.2985 | -0.2269 | 0.4824 | 0.1522 | -0.3301 | 8880 |
| source_train | 0.6644 | 0.4014 | -0.2630 | 0.6357 | 0.2385 | -0.3972 | 20600 |
| validation | 0.6004 | 0.4156 | -0.1848 | 0.5567 | 0.2818 | -0.2749 | 5260 |

## Buffer

| variant | final length | total added | total biased | mean added/subject |
|---|---:|---:|---:|---:|
| clean | 1271 | 241 | 0 | 24.10 |
| attack | 1294 | 264 | 67 | 26.40 |

## Lowest Old-generalization Points

| variant | min ACC | ACC step | min MF1 | MF1 step |
|---|---:|---:|---:|---:|
| clean | 0.6502 | 10 | 0.6032 | 10 |
| attack | 0.3726 | 10 | 0.2120 | 10 |

## Attack Diagnostics

| metric | mean | min | max |
|---|---:|---:|---:|
| clean_pass_rate | 0.6753 | 0.1163 | 1.0000 |
| adv_pass_rate | 0.8840 | 0.3915 | 1.0000 |
| accepted_rate | 0.8840 | 0.3915 | 1.0000 |
| mean_rel_eog | 0.0287 | 0.0112 | 0.0371 |
| mean_rel_eeg | 0.0262 | 0.0215 | 0.0323 |
| mean_feature_shift | 30.4534 | 2.7985 | 59.6679 |
