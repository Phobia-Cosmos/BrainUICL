# BrainUICL Attack-only Finalization

This report compares an attack-only run against an existing clean baseline with the same seed and new-individual order.

## Final Stability From CL Metrics

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.6502 | 0.6032 | 0.7046 | 0.6828 | 0.0744 |
| attack | 0.6716 | 0.6253 | 0.7028 | 0.6767 | 0.0439 |
| attack-clean | 0.0214 | 0.0221 | -0.0018 | -0.0062 | -0.0305 |

## Final Checkpoint Evaluation

| group | clean ACC | attack ACC | delta ACC | clean MF1 | attack MF1 | delta MF1 | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_generalization | 0.6502 | 0.6716 | 0.0214 | 0.6032 | 0.6253 | 0.0221 | 16700 |
| new_order_all | 0.5255 | 0.5720 | 0.0465 | 0.4824 | 0.5172 | 0.0348 | 8880 |
| source_train | 0.6644 | 0.6958 | 0.0314 | 0.6357 | 0.6707 | 0.0349 | 20600 |
| validation | 0.6004 | 0.6241 | 0.0238 | 0.5567 | 0.5871 | 0.0304 | 5260 |

## Buffer

| variant | final length | total added | total biased | mean added/subject |
|---|---:|---:|---:|---:|
| clean | 1271 | 241 | 0 | 24.10 |
| attack | 1284 | 254 | 0 | 25.40 |

## Lowest Old-generalization Points

| variant | min ACC | ACC step | min MF1 | MF1 step |
|---|---:|---:|---:|---:|
| clean | 0.6502 | 10 | 0.6032 | 10 |
| attack | 0.6716 | 10 | 0.6253 | 10 |

## Attack Diagnostics

| metric | mean | min | max |
|---|---:|---:|---:|
| clean_pass_rate | 0.4387 | 0.0000 | 0.8247 |
| adv_pass_rate | 0.3267 | 0.0000 | 0.5316 |
| accepted_rate | 0.3267 | 0.0000 | 0.5316 |
| mean_rel_eog | 0.0196 | 0.0000 | 0.0333 |
| mean_rel_eeg | 0.0187 | 0.0000 | 0.0329 |
| mean_feature_shift | 16.0352 | 0.0000 | 27.6342 |
