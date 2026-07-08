# BrainUICL Attack-only Finalization

This report compares an attack-only run against an existing clean baseline with the same seed and new-individual order.

## Final Stability From CL Metrics

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.7120 | 0.6825 | 0.6970 | 0.6726 | 0.0136 |
| attack | 0.6981 | 0.6765 | 0.6920 | 0.6671 | 0.0061 |
| attack-clean | -0.0138 | -0.0060 | -0.0050 | -0.0055 | -0.0074 |

## Final Checkpoint Evaluation

| group | clean ACC | attack ACC | delta ACC | clean MF1 | attack MF1 | delta MF1 | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_generalization | 0.7120 | 0.6981 | -0.0138 | 0.6825 | 0.6765 | -0.0060 | 16700 |
| new_order_all | 0.6095 | 0.6347 | 0.0252 | 0.5787 | 0.6055 | 0.0268 | 42960 |
| source_train | 0.7112 | 0.7383 | 0.0271 | 0.6894 | 0.7199 | 0.0305 | 20600 |
| validation | 0.6392 | 0.6095 | -0.0297 | 0.6132 | 0.5964 | -0.0168 | 5260 |

## Buffer

| variant | final length | total added | total biased | mean added/subject |
|---|---:|---:|---:|---:|
| clean | 2341 | 1311 | 0 | 26.76 |
| attack | 2121 | 1091 | 323 | 22.27 |

## Lowest Old-generalization Points

| variant | min ACC | ACC step | min MF1 | MF1 step |
|---|---:|---:|---:|---:|
| clean | 0.5374 | 17 | 0.4825 | 17 |
| attack | 0.5587 | 17 | 0.5140 | 17 |

## Attack Diagnostics

| metric | mean | min | max |
|---|---:|---:|---:|
| clean_pass_rate | 0.3922 | 0.0000 | 0.8239 |
| adv_pass_rate | 0.6526 | 0.0000 | 0.9500 |
| accepted_rate | 0.6526 | 0.0000 | 0.9500 |
| mean_rel_eog | 0.0240 | 0.0000 | 0.0666 |
| mean_rel_eeg | 0.0197 | 0.0000 | 0.0708 |
| mean_feature_shift | 26.8520 | 0.0000 | 47.0053 |
