# Stealth Loss Drift + Buffer Bias Full Validation

生成日期：2026-07-07

## 这次 resume 接上的未完成任务

上一轮 session 已经完成 `stealth_loss_drift + adv-pass + buffer pseudo-label bias` 的训练主体，但 50% bias run 在写最终 comparison/report 前被中断。本次补齐了：

- attack-only run 与同 seed clean baseline 的正式 comparison；
- 最终 checkpoint 在 old_generalization、new_order_all、source_train、validation 上的重新评估；
- 30% 和 50% buffer bias 的分布轨迹图；
- 一个可复用的 attack-only 收尾脚本：`experiments/finalize_attack_only.py`。

clean 对照使用：

```text
experiments/rttdp_brainuicl_runs/full49_stealth_loss_drift_advpass_seed4321/clean
```

attack-only 输出：

```text
experiments/rttdp_brainuicl_runs/full49_stealth_loss_drift_bufferbias_seed4321/
experiments/rttdp_brainuicl_runs/full49_stealth_loss_drift_bufferbias50_seed4321/
```

## Attack 配置

两版均为完整 49-subject attack-only，复用同 seed、同 new-order：

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python experiments/rttdp_brainuicl_full.py \
  --gpu 0 \
  --max-subjects 0 \
  --ssl-epoch 10 \
  --incremental-epoch 10 \
  --batch 16 \
  --num-worker 4 \
  --attack-mode stealth_loss_drift \
  --run-attack-only \
  --stealth-eps-scale 0.03 \
  --stealth-steps 5 \
  --stealth-eta 0.5 \
  --stealth-drift-weight 2.0 \
  --stealth-pass-weight 5.0 \
  --stealth-conf-weight 0.5 \
  --stealth-centroid-batches 4 \
  --stealth-accept-adv-only \
  --stealth-buffer-bias-mode second
```

差异：

```text
30% version: --stealth-buffer-bias-rate 0.3
50% version: --stealth-buffer-bias-rate 0.5
```

## 最终 old/generalization 稳定性

| variant | ACC | MF1 | AAA | AAF1 | FR | delta ACC | delta MF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.7120 | 0.6825 | 0.6970 | 0.6726 | 0.0136 | - | - |
| 30% buffer bias | 0.6981 | 0.6765 | 0.6920 | 0.6671 | 0.0061 | -0.0138 | -0.0060 |
| 50% buffer bias | 0.7026 | 0.6808 | 0.6888 | 0.6650 | 0.0003 | -0.0093 | -0.0017 |

结论：30% bias 的最终 old/generalization 下降更明显；50% bias 并没有随着 bias rate 提高而更强。

## 最终 checkpoint 重新评估

这里不是训练过程中的 per-subject plasticity，而是直接拿最后一个 `individual_49` checkpoint 评估完整 subject 组。

| variant | group | delta ACC | delta MF1 | clean ACC/MF1 | attack ACC/MF1 |
|---|---|---:|---:|---:|---:|
| 30% | old_generalization | -0.0138 | -0.0060 | 0.7120 / 0.6825 | 0.6981 / 0.6765 |
| 30% | new_order_all | +0.0252 | +0.0268 | 0.6095 / 0.5787 | 0.6347 / 0.6055 |
| 30% | validation | -0.0297 | -0.0168 | 0.6392 / 0.6132 | 0.6095 / 0.5964 |
| 30% | source_train | +0.0271 | +0.0305 | 0.7112 / 0.6894 | 0.7383 / 0.7199 |
| 50% | old_generalization | -0.0093 | -0.0017 | 0.7120 / 0.6825 | 0.7026 / 0.6808 |
| 50% | new_order_all | +0.0333 | +0.0302 | 0.6095 / 0.5787 | 0.6428 / 0.6089 |
| 50% | validation | -0.0072 | -0.0050 | 0.6392 / 0.6132 | 0.6319 / 0.6082 |
| 50% | source_train | +0.0506 | +0.0542 | 0.7112 / 0.6894 | 0.7618 / 0.7435 |

关键判断：

- 已验证：过滤通过、buffer 持续增长、old/generalization 和 validation 存在最终轻度退化。
- 未验证：最终 `new_order_all` 退化。当前两版在最终 new_order_all 上反而高于 clean。
- 因此当前攻击更像是“偏向 new/domain 的慢性重放偏置”，不是 old/new 同时下降的全局破坏。

## Buffer 与通过率

| variant | final buffer length | total added | total biased | mean added/subject | accepted rate | adv pass rate | clean pass rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 2341 | 1311 | 0 | 26.76 | - | - | - |
| 30% bias | 2121 | 1091 | 323 | 22.27 | 0.6526 | 0.6526 | 0.3922 |
| 50% bias | 2150 | 1120 | 559 | 22.86 | 0.6562 | 0.6562 | 0.3853 |

这说明 `--stealth-accept-adv-only` 确实让污染样本可以通过 confidence 过滤；攻击没有被过滤链路完全挡住。

## 慢性低谷

| variant | min old ACC | step | min old MF1 | step |
|---|---:|---:|---:|---:|
| clean | 0.5374 | - | 0.4825 | - |
| 30% bias | 0.5587 | 17 | 0.5140 | 17 |
| 50% bias | 0.5577 | 17 | 0.5138 | 17 |

注意：clean 自身也有 subject-order 引起的低谷，所以“慢性退化”不能只看最低点，必须看最终 delta 和分组 final checkpoint eval。

## 分布轨迹

输出目录：

```text
experiments/distribution_trajectory/full49_stealth_loss_drift_bufferbias_seed4321/
experiments/distribution_trajectory/full49_stealth_loss_drift_bufferbias50_seed4321/
```

关键 old/new centroid distance：

| state | clean | 30% bias | 50% bias |
|---|---:|---:|---:|
| pretrain | 18.3609 | 18.3609 | 18.3609 |
| clean_49 | 22.8168 | 22.8168 | 22.8168 |
| attack_49 | - | 26.9703 | 23.6447 |

30% bias 把最终 old/new 表征距离拉得更大，且 old/new centroid 相对 pretrain 的漂移都高于 clean；50% bias 的最终漂移反而更收敛。这与性能结果一致：简单提高 buffer label bias 不会线性增强破坏，可能反而让模型形成更强的 new/domain 适配。

## 结论

当前最佳版本是 30% buffer bias：

```text
通过过滤：是
buffer 持续增长：是
最终 old/generalization 下降：是，ACC -0.0138，MF1 -0.0060
最终 validation 下降：是，ACC -0.0297，MF1 -0.0168
最终 new_order_all 下降：否，ACC +0.0252，MF1 +0.0268
```

所以这次验证完成后的判断是：`stealth_loss_drift + adv-pass + buffer second-best bias` 能制造隐蔽的、可通过过滤的、对 old/validation 有轻度影响的慢性偏置，但还没有满足“攻击完成后 old/new 都退化”的完整目标。

要继续逼近 old/new 同时退化，下一步不应只提高 buffer bias rate。更合理的方向是把 bias 提前到当前 subject 的 joint pseudo-label loss 中，或者把 drift direction 换成稳定/重要神经元的 loss-ascent 方向；否则 buffer 只影响后续 replay，很难压低当前 new subject 的即时适配表现。
