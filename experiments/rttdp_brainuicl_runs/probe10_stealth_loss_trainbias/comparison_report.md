# BrainUICL Attack-only Finalization

This report compares an attack-only run against an existing clean baseline with the same seed and new-individual order.

## Final Stability From CL Metrics

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.6502 | 0.6032 | 0.7046 | 0.6828 | 0.0744 |
| attack | 0.6684 | 0.6364 | 0.7094 | 0.6874 | 0.0485 |
| attack-clean | 0.0182 | 0.0332 | 0.0047 | 0.0046 | -0.0259 |

## Final Checkpoint Evaluation

| group | clean ACC | attack ACC | delta ACC | clean MF1 | attack MF1 | delta MF1 | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_generalization | 0.6502 | 0.6684 | 0.0182 | 0.6032 | 0.6364 | 0.0332 | 16700 |
| new_order_all | 0.5255 | 0.5517 | 0.0262 | 0.4824 | 0.5183 | 0.0359 | 8880 |
| source_train | 0.6644 | 0.6767 | 0.0122 | 0.6357 | 0.6566 | 0.0208 | 20600 |
| validation | 0.6004 | 0.6321 | 0.0317 | 0.5567 | 0.6085 | 0.0518 | 5260 |

## Buffer

| variant | final length | total added | total biased | mean added/subject |
|---|---:|---:|---:|---:|
| clean | 1271 | 241 | 0 | 24.10 |
| attack | 1234 | 204 | 62 | 20.40 |

## Lowest Old-generalization Points

| variant | min ACC | ACC step | min MF1 | MF1 step |
|---|---:|---:|---:|---:|
| clean | 0.6502 | 10 | 0.6032 | 10 |
| attack | 0.6684 | 10 | 0.6364 | 10 |

## Attack Diagnostics

| metric | mean | min | max |
|---|---:|---:|---:|
| clean_pass_rate | 0.4492 | 0.0000 | 0.8485 |
| adv_pass_rate | 0.6208 | 0.0000 | 0.9583 |
| accepted_rate | 0.6208 | 0.0000 | 0.9583 |
| mean_rel_eog | 0.0197 | 0.0000 | 0.0352 |
| mean_rel_eeg | 0.0172 | 0.0000 | 0.0336 |
| mean_feature_shift | 29.0612 | 0.0000 | 68.2694 |
