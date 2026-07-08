# BrainUICL Attack-only Finalization

This report compares an attack-only run against an existing clean baseline with the same seed and new-individual order.

## Final Stability From CL Metrics

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.6502 | 0.6032 | 0.7046 | 0.6828 | 0.0744 |
| attack | 0.6699 | 0.6266 | 0.6898 | 0.6601 | 0.0463 |
| attack-clean | 0.0198 | 0.0234 | -0.0149 | -0.0227 | -0.0281 |

## Final Checkpoint Evaluation

| group | clean ACC | attack ACC | delta ACC | clean MF1 | attack MF1 | delta MF1 | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_generalization | 0.6502 | 0.6699 | 0.0198 | 0.6032 | 0.6266 | 0.0234 | 16700 |
| new_order_all | 0.5255 | 0.5682 | 0.0428 | 0.4824 | 0.5227 | 0.0403 | 8880 |
| source_train | 0.6644 | 0.7047 | 0.0402 | 0.6357 | 0.6841 | 0.0484 | 20600 |
| validation | 0.6004 | 0.6211 | 0.0207 | 0.5567 | 0.5797 | 0.0230 | 5260 |

## Buffer

| variant | final length | total added | total biased | mean added/subject |
|---|---:|---:|---:|---:|
| clean | 1271 | 241 | 0 | 24.10 |
| attack | 1063 | 33 | 0 | 3.30 |

## Lowest Old-generalization Points

| variant | min ACC | ACC step | min MF1 | MF1 step |
|---|---:|---:|---:|---:|
| clean | 0.6502 | 10 | 0.6032 | 10 |
| attack | 0.6699 | 10 | 0.6266 | 10 |

## Attack Diagnostics

| metric | mean | min | max |
|---|---:|---:|---:|
