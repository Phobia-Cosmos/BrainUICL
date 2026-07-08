# BrainUICL Data-only Threat Model Results

生成日期：2026-07-07

## 结论

在“只能修改传给目标模型的数据，不能修改目标模型参数、学习率、loss、label、buffer 写入逻辑”的威胁模型下：

- BrainUICL 的 buffer 属于 replay/memory-based continual learning。
- confidence threshold + `confident_epoch_n` 是质量控制，也可以视为一种防护式过滤。
- BrainUICL 没有 EWC/SI/MAS 这类显式参数正则；但 replay、CEA/feature alignment、weight decay/dropout、CPC 自监督适配都起到稳定化作用。
- 之前的 buffer label bias / train label bias / proxy ascent 都不是严格 data-only，只能作为 upper-bound 诊断。
- 本次新增的 `stealth_target_flip + store-poisoned-buffer` 是严格 data-only：只改输入数据，并把污染后的 sequence 存入 buffer，不直接改 label。

## BrainUICL buffer 如何工作

新增到 buffer 的单位是 sequence，不是单个 30s epoch。

一个 sequence 对应一个 `data/{idx}.npy`，内部包含 `SeqLength=20` 个 epoch。加入条件：

```text
对 sequence 内 20 个 epoch 预测；
如果至少 confident_epoch_n 个 epoch 的 max softmax prob >= confidence；
则把该 sequence 加入 replay buffer。
```

当前实验默认：

```text
confidence = 0.9
confident_epoch_n = 15
SeqLength = 20
```

## 模型更新发生在哪里

每个新 subject 到来后：

1. CPC/self-supervised adaptation：用当前 new subject 做自监督适配；
2. joint incremental update：用 new subject 伪标签 + replay buffer 更新模型；
3. buffer merge：训练后再预测当前 subject，筛选高置信 sequence 写入 buffer。

因此，buffer 写入发生在训练之后。

## 本次严格 data-only 修改

代码：[experiments/rttdp_brainuicl_full.py](/home/undefined/Desktop/bci/papers/TTAP/BrainUICL/experiments/rttdp_brainuicl_full.py:1)

新增：

```text
--attack-mode stealth_target_flip
--store-poisoned-buffer
```

含义：

- `stealth_target_flip`：只优化输入 `x_adv`，让模型在污染输入上倾向 second-best class，同时保留 confidence pass 和 raw/L2 约束；
- `store-poisoned-buffer`：buffer 中保存污染后的 `.npy` sequence，而不是 clean sequence + 人为改 label；
- 伪标签仍由目标模型自己预测生成。

## 10-subject probe

输出：

```text
experiments/rttdp_brainuicl_runs/probe10_dataonly_target_flip_eps008/
experiments/rttdp_brainuicl_runs/probe10_dataonly_target_flip_eps02/
```

结果：两个 probe 都没有形成 old/new 退化，反而普遍提升。说明简单 target-flip 输入扰动更像数据增强/域适配。

## 49-subject full run

命令核心参数：

```bash
--attack-mode stealth_target_flip
--run-attack-only
--store-poisoned-buffer
--stealth-eps-scale 0.20
--stealth-steps 12
--stealth-conf-weight 3.0
--stealth-pass-weight 3.0
--stealth-raw-weight 0.005
--stealth-l2-weight 0.001
--stealth-accept-adv-only
```

输出：

```text
experiments/rttdp_brainuicl_runs/full49_dataonly_target_flip_eps02_seed4321/
experiments/distribution_trajectory/full49_dataonly_target_flip_eps02_seed4321/
```

最终 checkpoint 评估：

| group | clean ACC/MF1 | attack ACC/MF1 | delta ACC | delta MF1 |
|---|---:|---:|---:|---:|
| old_generalization | 0.7120 / 0.6825 | 0.6878 / 0.6657 | -0.0241 | -0.0169 |
| new_order_all | 0.6095 / 0.5787 | 0.6250 / 0.5912 | +0.0155 | +0.0125 |
| source_train | 0.7112 / 0.6894 | 0.7452 / 0.7226 | +0.0340 | +0.0333 |
| validation | 0.6392 / 0.6132 | 0.6464 / 0.6253 | +0.0072 | +0.0121 |

buffer 持续增长，说明过滤没有挡住：

```text
final buffer length: 2742
```

## 表征轨迹

old/new centroid distance：

| state | distance |
|---|---:|
| pretrain | 18.3609 |
| clean_49 | 22.8168 |
| attack_10 | 27.0703 |
| attack_25 | 22.2365 |
| attack_49 | 23.2049 |

没有出现表征塌缩。`attack_49` 和 `clean_49` 的 old/new distance 接近。

## 解释

严格 data-only target-flip 没有效果，原因是：

- 污染样本高置信通过过滤后，被 BrainUICL 当作更多 new-domain 数据使用；
- 如果目标模型最终伪标签没有系统性错误，样本会增强适配；
- replay buffer 保存的是目标模型自己高置信预测的 sequence，错误不会像直接 label bias 那样被强行写入；
- CPC + replay + CEA alignment 会吸收输入扰动；
- second-best target 在 EEG 睡眠分期中可能只是边界增强，不一定是有害梯度。

## 下一步

严格 data-only 下，下一版不能再直接改 label 或 loss。应改成 data-only meta-poisoning：

```text
1. 本地 proxy 复制目标当前参数；
2. 对候选 x_adv 模拟目标的原始 BrainUICL 更新；
3. 外层优化 x_adv，使模拟更新后 old proxy loss 上升、new entropy 上升；
4. 同时要求 x_adv 在目标模型上高置信通过；
5. 只把 x_adv 发给目标模型。
```

也就是从“让当前预测变成 second-best”升级为“让目标模型按原始训练流程更新后自己变差”。这仍然只改数据，但比普通 PGD/target-flip 更接近真正的投毒目标。


## 2026-07-08 更新：proxy-meta conflict 已验证

上一节提出的 data-only meta-poisoning 已实现为 `--attack-mode proxy_meta_conflict`。10-sub 对照中，attack 相比 clean：

- final old ACC/MF1：`0.6772/0.6396 -> 0.6487/0.6106`；
- AAA/AAF1：`0.7032/0.6844 -> 0.5816/0.5314`；
- final new after ACC/MF1：`0.6047/0.5401 -> 0.4063/0.3183`。

完整报告：[PROXY_META_CONFLICT_ATTACK_RESULTS.md](/home/undefined/Desktop/bci/papers/TTAP/BrainUICL/PROXY_META_CONFLICT_ATTACK_RESULTS.md:1)
