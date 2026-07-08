# BrainUICL Attack-only Finalization

This report compares an attack-only run against an existing clean baseline with the same seed and new-individual order.

## Final Stability From CL Metrics

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.7120 | 0.6825 | 0.6970 | 0.6726 | 0.0136 |
| attack | 0.6878 | 0.6657 | 0.6813 | 0.6545 | 0.0208 |
| attack-clean | -0.0241 | -0.0169 | -0.0157 | -0.0181 | 0.0072 |

## Final Checkpoint Evaluation

| group | clean ACC | attack ACC | delta ACC | clean MF1 | attack MF1 | delta MF1 | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_generalization | 0.7120 | 0.6878 | -0.0241 | 0.6825 | 0.6657 | -0.0169 | 16700 |
| new_order_all | 0.6095 | 0.6250 | 0.0155 | 0.5787 | 0.5912 | 0.0125 | 42960 |
| source_train | 0.7112 | 0.7452 | 0.0340 | 0.6894 | 0.7226 | 0.0333 | 20600 |
| validation | 0.6392 | 0.6464 | 0.0072 | 0.6132 | 0.6253 | 0.0121 | 5260 |

## Buffer

| variant | final length | total added | total biased | mean added/subject |
|---|---:|---:|---:|---:|
| clean | 2341 | 1311 | 0 | 26.76 |
| attack | 2742 | 1712 | 0 | 34.94 |

## Lowest Old-generalization Points

| variant | min ACC | ACC step | min MF1 | MF1 step |
|---|---:|---:|---:|---:|
| clean | 0.5374 | 17 | 0.4825 | 17 |
| attack | 0.5278 | 17 | 0.4667 | 17 |

## Attack Diagnostics

| metric | mean | min | max |
|---|---:|---:|---:|
| clean_pass_rate | 0.4155 | 0.0000 | 0.7530 |
| adv_pass_rate | 0.4844 | 0.0023 | 0.7404 |
| accepted_rate | 0.4844 | 0.0023 | 0.7404 |
| mean_rel_eog | 0.0759 | 0.0004 | 0.2154 |
| mean_rel_eeg | 0.0659 | 0.0003 | 0.2498 |
| mean_feature_shift | 25.7028 | 0.0776 | 45.9053 |
