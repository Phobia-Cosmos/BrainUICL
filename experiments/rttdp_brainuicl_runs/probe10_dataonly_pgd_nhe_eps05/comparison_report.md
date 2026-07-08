# BrainUICL Attack-only Finalization

This report compares an attack-only run against an existing clean baseline with the same seed and new-individual order.

## Final Stability From CL Metrics

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.6502 | 0.6032 | 0.7046 | 0.6828 | 0.0744 |
| attack | 0.6426 | 0.5886 | 0.6753 | 0.6413 | 0.0852 |
| attack-clean | -0.0075 | -0.0146 | -0.0293 | -0.0416 | 0.0107 |

## Final Checkpoint Evaluation

| group | clean ACC | attack ACC | delta ACC | clean MF1 | attack MF1 | delta MF1 | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_generalization | 0.6502 | 0.6426 | -0.0075 | 0.6032 | 0.5886 | -0.0146 | 16700 |
| new_order_all | 0.5255 | 0.5349 | 0.0095 | 0.4824 | 0.4836 | 0.0013 | 8880 |
| source_train | 0.6644 | 0.6687 | 0.0043 | 0.6357 | 0.6392 | 0.0035 | 20600 |
| validation | 0.6004 | 0.6241 | 0.0238 | 0.5567 | 0.5715 | 0.0148 | 5260 |

## Buffer

| variant | final length | total added | total biased | mean added/subject |
|---|---:|---:|---:|---:|
| clean | 1271 | 241 | 0 | 24.10 |
| attack | 1033 | 3 | 0 | 0.30 |

## Lowest Old-generalization Points

| variant | min ACC | ACC step | min MF1 | MF1 step |
|---|---:|---:|---:|---:|
| clean | 0.6502 | 10 | 0.6032 | 10 |
| attack | 0.6426 | 10 | 0.5886 | 10 |

## Attack Diagnostics

| metric | mean | min | max |
|---|---:|---:|---:|
