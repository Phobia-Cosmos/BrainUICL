# BrainUICL Attack-only Finalization

This report compares an attack-only run against an existing clean baseline with the same seed and new-individual order.

## Final Stability From CL Metrics

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.7120 | 0.6825 | 0.6970 | 0.6726 | 0.0136 |
| attack | 0.1949 | 0.0652 | 0.2744 | 0.1687 | 0.7225 |
| attack-clean | -0.5171 | -0.6173 | -0.4226 | -0.5039 | 0.7090 |

## Final Checkpoint Evaluation

| group | clean ACC | attack ACC | delta ACC | clean MF1 | attack MF1 | delta MF1 | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_generalization | 0.7120 | 0.1949 | -0.5171 | 0.6825 | 0.0652 | -0.6173 | 16700 |
| new_order_all | 0.6095 | 0.2018 | -0.4077 | 0.5787 | 0.0672 | -0.5115 | 42960 |
| source_train | 0.7112 | 0.1860 | -0.5252 | 0.6894 | 0.0627 | -0.6266 | 20600 |
| validation | 0.6392 | 0.2105 | -0.4287 | 0.6132 | 0.0695 | -0.5437 | 5260 |

## Buffer

| variant | final length | total added | total biased | mean added/subject |
|---|---:|---:|---:|---:|
| clean | 2341 | 1311 | 0 | 26.76 |
| attack | 2843 | 1813 | 537 | 37.00 |

## Lowest Old-generalization Points

| variant | min ACC | ACC step | min MF1 | MF1 step |
|---|---:|---:|---:|---:|
| clean | 0.5374 | 17 | 0.4825 | 17 |
| attack | 0.1515 | 26 | 0.0606 | 28 |

## Attack Diagnostics

| metric | mean | min | max |
|---|---:|---:|---:|
| clean_pass_rate | 0.8100 | 0.0826 | 1.0000 |
| adv_pass_rate | 0.8698 | 0.0644 | 1.0000 |
| accepted_rate | 0.8698 | 0.0644 | 1.0000 |
| mean_rel_eog | 0.0324 | 0.0023 | 0.0940 |
| mean_rel_eeg | 0.0272 | 0.0017 | 0.0881 |
| mean_feature_shift | 53.7539 | 0.2557 | 156.2943 |
