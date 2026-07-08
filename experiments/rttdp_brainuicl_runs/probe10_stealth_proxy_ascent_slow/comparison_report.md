# BrainUICL Attack-only Finalization

This report compares an attack-only run against an existing clean baseline with the same seed and new-individual order.

## Final Stability From CL Metrics

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.6502 | 0.6032 | 0.7046 | 0.6828 | 0.0744 |
| attack | 0.5911 | 0.5291 | 0.6794 | 0.6544 | 0.1585 |
| attack-clean | -0.0590 | -0.0742 | -0.0253 | -0.0284 | 0.0841 |

## Final Checkpoint Evaluation

| group | clean ACC | attack ACC | delta ACC | clean MF1 | attack MF1 | delta MF1 | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_generalization | 0.6502 | 0.5911 | -0.0590 | 0.6032 | 0.5291 | -0.0742 | 16700 |
| new_order_all | 0.5255 | 0.4806 | -0.0448 | 0.4824 | 0.4162 | -0.0661 | 8880 |
| source_train | 0.6644 | 0.6186 | -0.0458 | 0.6357 | 0.5670 | -0.0688 | 20600 |
| validation | 0.6004 | 0.5865 | -0.0139 | 0.5567 | 0.5447 | -0.0120 | 5260 |

## Buffer

| variant | final length | total added | total biased | mean added/subject |
|---|---:|---:|---:|---:|
| clean | 1271 | 241 | 0 | 24.10 |
| attack | 1238 | 208 | 52 | 20.80 |

## Lowest Old-generalization Points

| variant | min ACC | ACC step | min MF1 | MF1 step |
|---|---:|---:|---:|---:|
| clean | 0.6502 | 10 | 0.6032 | 10 |
| attack | 0.5911 | 10 | 0.5291 | 10 |

## Attack Diagnostics

| metric | mean | min | max |
|---|---:|---:|---:|
| clean_pass_rate | 0.4870 | 0.0000 | 0.7841 |
| adv_pass_rate | 0.6755 | 0.0000 | 0.9735 |
| accepted_rate | 0.6755 | 0.0000 | 0.9735 |
| mean_rel_eog | 0.0211 | 0.0000 | 0.0345 |
| mean_rel_eeg | 0.0188 | 0.0000 | 0.0328 |
| mean_feature_shift | 29.1329 | 0.0000 | 56.6754 |
