# Stealth Drift Full 49-subject Experiment

生成日期：2026-07-07

## 运行命令

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --gpu 0 \
  --output-root experiments/rttdp_brainuicl_runs/full49_stealth_drift_seed4321 \
  --max-subjects 0 \
  --ssl-epoch 10 \
  --incremental-epoch 10 \
  --batch 16 \
  --num-worker 4 \
  --attack-mode stealth_drift \
  --stealth-eps-scale 0.01 \
  --stealth-steps 5 \
  --stealth-drift-weight 0.2 \
  --stealth-pass-weight 2.0
```

输出目录：

```text
experiments/rttdp_brainuicl_runs/full49_stealth_drift_seed4321/
experiments/distribution_trajectory/full49_stealth_drift_seed4321/
```

## 最终结果

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.7120 | 0.6825 | 0.6970 | 0.6726 | 0.0136 |
| stealth_drift | 0.7043 | 0.6855 | 0.6982 | 0.6737 | 0.0026 |
| model_nhe full run | 0.2191 | 0.0974 | 0.2137 | 0.1033 | 0.6881 |

结论：当前 `stealth_drift` 参数非常隐蔽，但攻击强度不足。它没有像 `model_nhe` 那样破坏模型；最终 ACC 只比 clean 低 `0.0077`，MF1 还略高。

## Plasticity 汇总

| variant | initial ACC | before ACC | after ACC | initial MF1 | before MF1 | after MF1 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.6464 | 0.6154 | 0.6148 | 0.5568 | 0.5428 | 0.5567 |
| stealth_drift | 0.6464 | 0.6372 | 0.6240 | 0.5568 | 0.5603 | 0.5672 |

## Buffer 与通过率

| metric | clean | stealth_drift |
|---|---:|---:|
| final buffer length | 2341 | 2323 |
| added pseudo labels | 1311 | 1293 |
| mean added / subject | 26.76 | 26.39 |

`stealth_drift` 诊断均值：

| metric | mean | min | max |
|---|---:|---:|---:|
| clean pass rate | 0.4545 | 0.0000 | 0.8102 |
| adv pass rate | 0.6665 | 0.0000 | 0.9628 |
| accepted rate | 0.4420 | 0.0000 | 0.8057 |
| rel EOG perturbation | 0.0051 | 0.0000 | 0.0152 |
| rel EEG perturbation | 0.0044 | 0.0000 | 0.0130 |
| feature shift | 5.7250 | 0.0000 | 12.3948 |

解释：

- 扰动很小，平均相对 L2 在 `0.4%-0.5%`。
- buffer 基本正常增长，说明污染样本没有被置信度过滤大量挡掉。
- accepted rate 接近 clean pass rate，说明“只污染原本可通过样本”的策略有效。

## 表征分布轨迹

轨迹图：

```text
experiments/distribution_trajectory/full49_stealth_drift_seed4321/raw_signal_tsne.png
experiments/distribution_trajectory/full49_stealth_drift_seed4321/feature_tsne_by_checkpoint.png
experiments/distribution_trajectory/full49_stealth_drift_seed4321/centroid_shift_from_pretrain.png
experiments/distribution_trajectory/full49_stealth_drift_seed4321/old_new_distance_by_checkpoint.png
```

old/new centroid distance：

| state | old-new distance |
|---|---:|
| pretrain | 18.3609 |
| clean_10 | 23.9944 |
| clean_25 | 24.9056 |
| clean_49 | 22.8168 |
| stealth_10 | 25.1749 |
| stealth_25 | 22.5212 |
| stealth_49 | 25.6501 |

centroid shift from pretrain：

| state/group | shift |
|---|---:|
| clean_49:old_generalization | 23.3954 |
| stealth_49:old_generalization | 19.3032 |
| clean_49:new_order | 23.3314 |
| stealth_49:new_order | 23.2720 |
| clean_49:source_train | 49.1309 |
| stealth_49:source_train | 23.5894 |

解释：

- `model_nhe` 会让 old/new distance 在 attack_25 后塌缩到 `0`。
- `stealth_drift` 没有塌缩，old/new distance 最终仍为 `25.65`。
- 当前 drift direction 没有把表征稳定地推向有害方向，反而保持了较正常的结构。

## 关键结论

当前 `stealth_drift` 完成了隐蔽性目标：

```text
扰动小；
通过过滤；
buffer 正常增长；
模型没有快速崩溃。
```

但没有完成攻击效果目标：

```text
没有造成最终 old/generalization 性能明显下降；
没有形成可积累的有害表征偏移；
随机 latent drift direction 不够有效。
```

## 下一轮改进

需要把 drift direction 从固定随机方向改成 proxy 优化方向。

建议下一版：

1. 只攻击 clean confidence pass 的 sequence，保留当前策略。
2. 把 `stealth_eps_scale` 提到 `0.015` 或 `0.02`，但保持 accepted rate 不低于 clean pass rate 的 60%。
3. 把 `stealth_drift_weight` 从 `0.2` 提到 `0.5/1.0`。
4. 把 drift direction 改成以下之一：

```text
old-generalization loss ascent direction
stable-important core drift direction
class-boundary bias direction
source/new centroid separation direction
```

5. 每个 subject 后保存：

```text
feature centroid drift
old/new distance
stable-core update ratio
confidence pass rate
buffer added
old ACC/MF1
```

这次 full run 的价值是建立了隐蔽攻击基线：它证明“可通过过滤、buffer 正常增长”是可做到的。下一步要提高的是累计破坏方向，而不是再回到 `model_nhe` 那种快速崩溃。
