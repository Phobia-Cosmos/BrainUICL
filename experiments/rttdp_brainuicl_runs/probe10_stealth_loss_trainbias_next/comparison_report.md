# BrainUICL Attack-only Finalization

This report compares an attack-only run against an existing clean baseline with the same seed and new-individual order.

## Final Stability From CL Metrics

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.6502 | 0.6032 | 0.7046 | 0.6828 | 0.0744 |
| attack | 0.6785 | 0.6449 | 0.7084 | 0.6863 | 0.0341 |
| attack-clean | 0.0283 | 0.0416 | 0.0038 | 0.0035 | -0.0403 |

## Final Checkpoint Evaluation

| group | clean ACC | attack ACC | delta ACC | clean MF1 | attack MF1 | delta MF1 | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_generalization | 0.6502 | 0.6785 | 0.0283 | 0.6032 | 0.6449 | 0.0416 | 16700 |
| new_order_all | 0.5255 | 0.5765 | 0.0510 | 0.4824 | 0.5364 | 0.0540 | 8880 |
| source_train | 0.6644 | 0.7018 | 0.0374 | 0.6357 | 0.6830 | 0.0472 | 20600 |
| validation | 0.6004 | 0.6319 | 0.0316 | 0.5567 | 0.6032 | 0.0465 | 5260 |

## Buffer

| variant | final length | total added | total biased | mean added/subject |
|---|---:|---:|---:|---:|
| clean | 1271 | 241 | 0 | 24.10 |
| attack | 1235 | 205 | 99 | 20.50 |

## Lowest Old-generalization Points

| variant | min ACC | ACC step | min MF1 | MF1 step |
|---|---:|---:|---:|---:|
| clean | 0.6502 | 10 | 0.6032 | 10 |
| attack | 0.6785 | 10 | 0.6449 | 10 |

## Attack Diagnostics

| metric | mean | min | max |
|---|---:|---:|---:|
| clean_pass_rate | 0.4472 | 0.0000 | 0.8447 |
| adv_pass_rate | 0.5904 | 0.0000 | 0.9419 |
| accepted_rate | 0.5904 | 0.0000 | 0.9419 |
| mean_rel_eog | 0.0183 | 0.0000 | 0.0303 |
| mean_rel_eeg | 0.0157 | 0.0000 | 0.0246 |
| mean_feature_shift | 24.9732 | 0.0000 | 47.7106 |
