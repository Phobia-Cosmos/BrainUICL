# BrainUICL Attack-only Finalization

This report compares an attack-only run against an existing clean baseline with the same seed and new-individual order.

## Final Stability From CL Metrics

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.7120 | 0.6825 | 0.6970 | 0.6726 | 0.0136 |
| attack | 0.7026 | 0.6808 | 0.6888 | 0.6650 | 0.0003 |
| attack-clean | -0.0093 | -0.0017 | -0.0082 | -0.0076 | -0.0133 |

## Final Checkpoint Evaluation

| group | clean ACC | attack ACC | delta ACC | clean MF1 | attack MF1 | delta MF1 | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_generalization | 0.7120 | 0.7026 | -0.0093 | 0.6825 | 0.6808 | -0.0017 | 16700 |
| new_order_all | 0.6095 | 0.6428 | 0.0333 | 0.5787 | 0.6089 | 0.0302 | 42960 |
| source_train | 0.7112 | 0.7618 | 0.0506 | 0.6894 | 0.7435 | 0.0542 | 20600 |
| validation | 0.6392 | 0.6319 | -0.0072 | 0.6132 | 0.6082 | -0.0050 | 5260 |

## Buffer

| variant | final length | total added | total biased | mean added/subject |
|---|---:|---:|---:|---:|
| clean | 2341 | 1311 | 0 | 26.76 |
| attack | 2150 | 1120 | 559 | 22.86 |

## Lowest Old-generalization Points

| variant | min ACC | ACC step | min MF1 | MF1 step |
|---|---:|---:|---:|---:|
| clean | 0.5374 | 17 | 0.4825 | 17 |
| attack | 0.5577 | 17 | 0.5138 | 17 |

## Attack Diagnostics

| metric | mean | min | max |
|---|---:|---:|---:|
| clean_pass_rate | 0.3853 | 0.0000 | 0.8068 |
| adv_pass_rate | 0.6562 | 0.0000 | 0.9688 |
| accepted_rate | 0.6562 | 0.0000 | 0.9688 |
| mean_rel_eog | 0.0240 | 0.0000 | 0.0720 |
| mean_rel_eeg | 0.0200 | 0.0000 | 0.0674 |
| mean_feature_shift | 27.6994 | 0.0000 | 58.6873 |
