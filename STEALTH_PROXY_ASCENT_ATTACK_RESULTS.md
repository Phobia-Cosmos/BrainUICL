# Stealth Proxy Ascent Attack Results

生成日期：2026-07-07

## 为什么上一版没有造成最终退化

上一版 `stealth_loss_drift + adv-pass + buffer bias` 不是“完全没影响”，但影响没有落到关键更新路径：

- `buffer_bias` 在每个 subject 训练结束后才写入 buffer，不影响当前 subject 的 CPC/joint 更新；
- `adv-pass` 增加了可通过 confidence filter 的样本，反而增强了 new/domain 适配；
- second-best label bias 太像边界正则化，模型和 replay/CEA alignment 能吸收；
- 对 source/replay 做弱 loss-ascent 不等价于让 old_generalization 退化，弱扰动甚至提升泛化。

所以之前看到的是：old/validation 有轻微影响，但 new_order_all 和 source_train 反而提升。

## 这次修改

核心代码：[experiments/rttdp_brainuicl_full.py](/home/undefined/Desktop/bci/papers/TTAP/BrainUICL/experiments/rttdp_brainuicl_full.py:1)

新增能力：

- `--stealth-train-new-bias-rate`：训练期对当前 new pseudo labels 做 label bias；
- `--stealth-train-replay-bias-rate`：训练期对 replay labels 做 label bias；
- `--stealth-train-new-loss-scale`：只在 stealth attack 中放大 new pseudo-label loss；
- `--stealth-new-ascent-weight` / `--stealth-replay-ascent-weight`：对当前 batch 的 clean loss 做反向优化；
- `--stealth-old-proxy-ascent-weight`：用 old proxy batch 做 CE loss ascent；
- `--stealth-new-entropy-ascent-weight`：对当前 new batch 最大化 entropy，降低可学习性；
- `--stealth-ascent-lr`：单独 attack optimizer 学习率。

关键变化是把攻击从“buffer 后处理”推进到“joint update 梯度本身”，否则污染信号太晚、太弱。

## 有效参数

10-subject slow probe 使用：

```bash
--stealth-ascent-lr 2e-6
--stealth-old-proxy-ascent-weight 0.3
--stealth-new-entropy-ascent-weight 0.1
--stealth-train-new-bias-rate 0.20
--stealth-train-new-loss-scale 5
--stealth-buffer-bias-rate 0.3
--stealth-accept-adv-only
```

## 10-subject probe

输出：

```text
experiments/rttdp_brainuicl_runs/probe10_stealth_proxy_ascent_slow/
```

| group | clean ACC/MF1 | attack ACC/MF1 | delta ACC | delta MF1 |
|---|---:|---:|---:|---:|
| old_generalization | 0.6502 / 0.6032 | 0.5911 / 0.5291 | -0.0590 | -0.0742 |
| new_order_all | 0.5255 / 0.4824 | 0.4806 / 0.4162 | -0.0448 | -0.0661 |
| source_train | 0.6644 / 0.6357 | 0.6186 / 0.5670 | -0.0458 | -0.0688 |
| validation | 0.6004 / 0.5567 | 0.5865 / 0.5447 | -0.0139 | -0.0120 |

buffer 仍增长：

```text
final buffer length: 1238
total added: 208
total biased: 52
```

这个 probe 是目前最接近“慢性、可通过过滤、old/new 同步退化”的版本。

## 49-subject full run

输出：

```text
experiments/rttdp_brainuicl_runs/full49_stealth_proxy_ascent_slow_seed4321/
experiments/distribution_trajectory/full49_stealth_proxy_ascent_slow_seed4321/
```

完整 10+10 epoch 下，同一组参数被放大成强攻击。

| group | clean ACC/MF1 | attack ACC/MF1 | delta ACC | delta MF1 |
|---|---:|---:|---:|---:|
| old_generalization | 0.7120 / 0.6825 | 0.1949 / 0.0652 | -0.5171 | -0.6173 |
| new_order_all | 0.6095 / 0.5787 | 0.2018 / 0.0672 | -0.4077 | -0.5115 |
| source_train | 0.7112 / 0.6894 | 0.1860 / 0.0627 | -0.5252 | -0.6266 |
| validation | 0.6392 / 0.6132 | 0.2105 / 0.0695 | -0.4287 | -0.5437 |

buffer 和过滤没有断：

```text
final buffer length: 2843
total added: 1813
total biased: 537
mean accepted rate: 0.8698
mean adv pass rate: 0.8698
mean clean pass rate: 0.8100
```

## 表征轨迹

`attack_49` 时 old/new centroid distance 被压到很低：

| state | old-new centroid distance |
|---|---:|
| pretrain | 18.3609 |
| clean_49 | 22.8168 |
| attack_10 | 21.0811 |
| attack_25 | 17.2752 |
| attack_49 | 5.1851 |

centroid shift from pretrain：

| group | clean_49 | attack_49 |
|---|---:|---:|
| source_train | 49.1309 | 614.8929 |
| old_generalization | 23.3954 | 650.7833 |
| new_order | 23.3314 | 647.3033 |

这说明攻击不是只改了分类头输出，而是把特征空间整体拉坏了。

## 结论

上一版攻击不是完全无关，而是没有打到正确位置。有效修改是：

```text
buffer 后处理 -> joint update 内的 proxy ascent
source replay 弱扰动 -> old proxy loss ascent + new entropy ascent
共用 CL lr=1e-7 -> 单独 attack lr
```

当前 49-sub full run 已证明攻击能让 old/new 同时退化，但强度偏大。后续要做“更隐蔽慢性”的 full-run 参数，应继续降低：

```text
stealth-ascent-lr: 5e-7 到 1e-6
stealth-old-proxy-ascent-weight: 0.1 到 0.2
stealth-new-entropy-ascent-weight: 0.03 到 0.08
```

本轮最重要的验证结论是：要让 BrainUICL 真的退化，必须攻击更新梯度或可替代 old/generalization proxy；只改进入 buffer 的标签会被 BrainUICL 的 replay、过滤和 subject adaptation 大量吸收。
