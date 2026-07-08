# BrainUICL Attack-only Finalization

This report compares an attack-only run against an existing clean baseline with the same seed and new-individual order.

## Final Stability From CL Metrics

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.6502 | 0.6032 | 0.7046 | 0.6828 | 0.0744 |
| attack | 0.1950 | 0.0654 | 0.3058 | 0.2164 | 0.7224 |
| attack-clean | -0.4552 | -0.5379 | -0.3988 | -0.4664 | 0.6480 |

## Final Checkpoint Evaluation

| group | clean ACC | attack ACC | delta ACC | clean MF1 | attack MF1 | delta MF1 | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_generalization | 0.6502 | 0.1950 | -0.4552 | 0.6032 | 0.0654 | -0.5379 | 16700 |
| new_order_all | 0.5255 | 0.2117 | -0.3137 | 0.4824 | 0.0699 | -0.4125 | 8880 |
| source_train | 0.6644 | 0.1860 | -0.4784 | 0.6357 | 0.0627 | -0.5730 | 20600 |
| validation | 0.6004 | 0.2105 | -0.3899 | 0.5567 | 0.0695 | -0.4872 | 5260 |

## Buffer

| variant | final length | total added | total biased | mean added/subject |
|---|---:|---:|---:|---:|
| clean | 1271 | 241 | 0 | 24.10 |
| attack | 1426 | 396 | 141 | 39.60 |

## Lowest Old-generalization Points

| variant | min ACC | ACC step | min MF1 | MF1 step |
|---|---:|---:|---:|---:|
| clean | 0.6502 | 10 | 0.6032 | 10 |
| attack | 0.1827 | 6 | 0.0640 | 7 |

## Attack Diagnostics

| metric | mean | min | max |
|---|---:|---:|---:|
| clean_pass_rate | 0.7765 | 0.0913 | 1.0000 |
| adv_pass_rate | 0.8538 | 0.0754 | 1.0000 |
| accepted_rate | 0.8538 | 0.0754 | 1.0000 |
| mean_rel_eog | 0.0260 | 0.0013 | 0.0361 |
| mean_rel_eeg | 0.0250 | 0.0023 | 0.0452 |
| mean_feature_shift | 34.2547 | 0.3135 | 57.6323 |
