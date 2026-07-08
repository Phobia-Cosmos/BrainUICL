# Checkpoint、分布可视化与 Stealth Drift 攻击实现说明

生成日期：2026-07-07

相关文件：

```text
experiments/distribution_trajectory.py
experiments/rttdp_brainuicl_full.py
experiments/rttdp_brainuicl_runs/smoke_stealth_drift_2sub/
```

## 1. checkpoint 是什么

checkpoint 是模型在某个训练时刻的参数快照。

在 BrainUICL 里，一个 checkpoint 通常包含三部分：

```text
feature_extractor_parameter_4321.pkl
feature_encoder_parameter_4321.pkl
sleep_classifier_parameter_4321.pkl
```

它不是数据，也不会改变原始 EEG/EOG 文件。它只是模型参数。

## 2. 什么叫“原始输入经过 checkpoint”

更准确的说法是：

```text
同一批原始 EEG/EOG 输入 x
分别送入不同 checkpoint 对应的模型 f_theta
得到不同的 feature embedding z = f_theta(x)
```

所以“输入经过 checkpoint”不是输入被修改，而是模型参数不同，提取出的特征不同。

例如：

```text
z_pretrain = f_pretrain(x)
z_clean49  = f_clean_after_49_subjects(x)
z_attack49 = f_attack_after_49_subjects(x)
```

同一个 `x` 不变，但 `z` 可以变。

## 3. 数据分布应该看 raw 还是 feature

都要看，但回答的问题不同。

| 分布空间 | 看什么 | 回答什么问题 |
|---|---|---|
| raw EEG/EOG | 原始信号统计、频段功率、时间序列形态 | 输入是否像异常/OOD 数据 |
| model feature | feature extractor/encoder 后的 embedding | 模型“如何理解”这些输入 |
| classifier output | confidence、pseudo label、class probability | 是否能通过伪标签过滤，是否影响预测 |

攻击可以在 raw 空间看起来很小，但在 feature 空间造成明显偏移。因此不能只看 raw，也不能只看 feature。

## 4. 为什么会有不同 checkpoint

BrainUICL 是 continual learning，每来一个 new subject，模型会更新一次。因此会有：

```text
Pretrain checkpoint
individual_1 checkpoint
individual_2 checkpoint
...
individual_49 checkpoint
```

在可视化里我取了几个阶段：

```text
pretrain
clean_10, clean_25, clean_49
attack_10, attack_25, attack_49
```

这样可以看到模型表征随 CL 过程如何变化，而不是只看最终结果。

## 5. pretrain 以后的“分布”和 centroid 是什么

给定一组样本，例如 old_generalization subjects：

```text
x_1, x_2, ..., x_n
```

送入某个 checkpoint：

```text
z_i = feature_encoder(feature_extractor(x_i))
```

这些 `z_i` 构成这个 checkpoint 下的 feature distribution。

centroid 是这些 embedding 的均值：

```text
centroid = mean(z_1, z_2, ..., z_n)
```

所以：

- distribution 是一堆点；
- centroid 是这堆点的中心；
- centroid shift 是这个中心相对 pretrain 移动了多少。

## 6. clean 的 centroid 相对 pretrain 偏移大吗

要相对解释。

这次轨迹分析结果：

| state/group | centroid shift from pretrain |
|---|---:|
| clean_10:old_generalization | 26.3125 |
| clean_25:old_generalization | 25.4245 |
| clean_49:old_generalization | 23.3954 |
| clean_10:new_order | 25.7784 |
| clean_25:new_order | 25.1508 |
| clean_49:new_order | 23.3314 |
| attack_10:old_generalization | 76.4362 |
| attack_49:old_generalization | 83.9796 |
| attack_10:new_order | 71.5969 |
| attack_49:new_order | 78.5071 |

解释：

- clean CL 的 feature centroid 确实会移动，幅度在 `23-26` 左右。
- 这说明正常 CL 也会改变表征，不能把“有漂移”直接视为攻击。
- attack 分支漂移到 `70-84`，明显大得多，且伴随 old/new 表征塌缩。

所以 clean 偏移不算小，但它是有结构的、受 replay/CEA/pseudo label 约束的偏移；当前强攻击是失控式偏移。

## 7. old/new distance 是什么

old/new distance 是：

```text
distance(mean(feature(old_subjects)), mean(feature(new_subjects)))
```

它衡量同一个 checkpoint 下，old_generalization subjects 和 new_order subjects 在 feature 空间里的中心距离。

它不是准确率指标，而是表征空间指标。

## 8. 为什么要在 clean_10 / clean_25 / clean_49 等阶段计算 old/new distance

因为我们想知道：

```text
随着 CL 持续进行，
模型是否仍能保持 old 和 new 的表征结构，
还是把它们拉到一起/拉乱。
```

结果：

| state | old/new centroid distance |
|---|---:|
| pretrain | 18.3609 |
| clean_10 | 23.9944 |
| clean_25 | 24.9056 |
| clean_49 | 22.8168 |
| attack_10 | 3.0812 |
| attack_25 | 0.0000 |
| attack_49 | 0.0000 |

解释：

- old 和 new 本来就不完全在一起，因为不同 subject 存在 domain shift。
- clean CL 后 old/new 距离仍保持在 `22-25`，说明模型表征结构还在。
- 当前强 attack 会让 old/new centroid 距离塌缩到接近 0，说明表征崩掉了，不是隐蔽偏移。

## 9. old 和 new 的分布本来就有偏差吗

有。

它们是不同 subject group：

- old_generalization：历史泛化测试 subject；
- new_order：continual update 到来的新 subject。

虽然它们共享同一套睡眠分期标签，但 EEG/EOG 受个体差异、睡眠结构、噪声、幅值、采集差异影响，所以 raw 和 feature distribution 都会有 subject/domain gap。

这也是 BrainUICL 要做 individual continual learning 的原因。

## 10. 已实现的更隐蔽攻击版本

已在独立 runner 中新增：

```text
--attack-mode stealth_drift
```

文件：

```text
experiments/rttdp_brainuicl_full.py
```

它没有修改原始 BrainUICL CL 入口。

### 攻击思想

`stealth_drift` 不再直接用 NHE 让模型输出崩掉，而是对输入做 constrained PGD：

```text
保持 clean pseudo label
保持 confidence
限制 raw signal 扰动
轻微推动 feature centroid 沿固定 proxy drift direction 移动
如果扰动后不通过 confidence filter，则回退为 clean 输入
```

核心目标：

```text
让污染样本尽量通过置信度过滤，
每一步只制造小 feature drift，
靠多 subject 累积造成慢性退化。
```

### 新增参数

```text
--stealth-steps
--stealth-eps-scale
--stealth-eta
--stealth-conf-weight
--stealth-pass-weight
--stealth-raw-weight
--stealth-l2-weight
--stealth-align-weight
--stealth-drift-weight
```

默认较保守：

```text
--stealth-eps-scale 0.01
--stealth-steps 5
```

## 11. smoke 运行结果

命令：

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --gpu 0 \
  --output-root experiments/rttdp_brainuicl_runs/smoke_stealth_drift_2sub \
  --max-subjects 2 \
  --ssl-epoch 1 \
  --incremental-epoch 1 \
  --batch 4 \
  --num-worker 0 \
  --attack-mode stealth_drift \
  --stealth-eps-scale 0.01 \
  --stealth-steps 5 \
  --stealth-drift-weight 0.2 \
  --stealth-pass-weight 2.0
```

结果：

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.7139 | 0.7017 | 0.7110 | 0.6971 | 0.0163 |
| attack_stealth_drift | 0.6599 | 0.6311 | 0.6838 | 0.6618 | 0.0605 |

buffer：

| subject | clean added | stealth added |
|---:|---:|---:|
| 64 | 25 | 23 |
| 89 | 31 | 27 |

stealth 诊断：

| subject | clean pass rate | adv pass rate | accepted rate | rel EOG | rel EEG | feature shift |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.465 | 0.884 | 0.430 | 0.00358 | 0.00333 | 5.5646 |
| 89 | 0.580 | 0.850 | 0.580 | 0.00531 | 0.00532 | 8.0062 |

解释：

- 扰动幅度非常小，约 `0.3%-0.5%` relative L2。
- buffer 仍然新增了大部分样本，不像强攻击那样完全阻断伪标签链路。
- old/generalization 已经出现轻度退化，但没有快速崩溃。
- 这更接近“隐蔽、缓慢、可持续退化”的方向。

## 12. 当前实现还不是最终攻击

现在的 `stealth_drift` 是第一版：

- drift direction 目前是固定 proxy latent direction；
- 还没有按 stable-core/Fisher 自动选择最危险方向；
- 还没有动态调节 eps 和 drift weight；
- 还没有完整 49-subject full run。

下一步应做：

```text
stealth_drift full49
动态选择高置信 sequence
基于 stable-important/plastic-important units 选择 drift direction
每个 subject 保存 feature centroid drift 和 stable-core update ratio
```
