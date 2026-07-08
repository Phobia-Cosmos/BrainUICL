# Stealth Proxy Drift Attack for BrainUICL

生成日期：2026-07-07

相关输出：

```text
experiments/distribution_trajectory/full49_model_nhe_seed4321/
experiments/attack_diagnostics/full49_model_nhe_seed4321_pgd_eps001/
```

## 1. 当前可视化说明了什么

新生成的 checkpoint 轨迹可视化把“输入数据分布”和“模型表征分布”分开看。

输出图像：

```text
experiments/distribution_trajectory/full49_model_nhe_seed4321/raw_signal_tsne.png
experiments/distribution_trajectory/full49_model_nhe_seed4321/feature_tsne_by_checkpoint.png
experiments/distribution_trajectory/full49_model_nhe_seed4321/centroid_shift_from_pretrain.png
experiments/distribution_trajectory/full49_model_nhe_seed4321/old_new_distance_by_checkpoint.png
```

关键数值：

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

- 原始输入没有经过 checkpoint，所以 raw distribution 不会因为 CL checkpoint 改变。
- 同一批 clean 输入进入不同 checkpoint 后，feature distribution 会改变。
- clean CL 的 feature centroid 相对 pretrain 有漂移，但 old/new 距离仍然维持在 `22-25` 左右。
- 当前 `model_nhe` attack 会让 old/new 表征距离快速塌缩到接近 `0`，说明模型表征发生了明显崩溃。

这说明当前攻击不是隐蔽慢性攻击，而是强破坏上界攻击。它适合说明“BrainUICL 可以被 white-box 代理攻击拉崩”，但不适合模拟 stealthy persistent degradation。

## 2. 原始数据空间和特征空间的关系

现在应按三层分开理解：

```text
raw EEG/EOG signal space
-> signal-stat / frequency / time-series space
-> model embedding / alignment space
-> classifier output / confidence space
```

当前 PGD `eps=0.01` 的相对扰动只有约 `1%` batch std，原始信号统计空间里可能仍然离 clean 很近；但模型 embedding 和 confidence 已经明显变化。这说明攻击主要利用模型敏感方向，而不是制造肉眼明显的原始信号偏移。

因此后续不能只看 raw t-SNE，也不能只看 feature t-SNE。需要同时看：

- raw signal stats t-SNE：是否像输入异常；
- feature t-SNE：模型表征是否被拉偏；
- confidence pass：是否能进入 buffer；
- gradient cosine/norm：是否产生异常更新方向；
- stable-core drift：是否影响稳定重要表征；
- per-step centroid drift：是否发生慢性积累。

## 3. 论文中的 CL 分布是否会一直变化

会变化，但变化不一定是坏事。

CL 本来就需要在新任务/新个体到来时改变模型表征。关键区别是：

- 正常 CL：变化受旧任务约束，old/new 表征仍可分，old performance 不应明显崩。
- 当前强攻击：变化方向被攻击目标主导，old/new 表征塌缩，buffer 不增长，性能崩。
- 隐蔽慢性攻击：每一步变化都不超过对齐/检测阈值，但多步累积后把表征中心带偏。

所以“分布一直变化”不是攻击证据；异常的是：

```text
变化方向持续一致地远离历史表征，
且每一步单独看都不明显，
最后 old/generalization 预测边界发生偏移。
```

## 4. BrainUICL 的对齐在攻击中如何变化

代码里的对齐主要是 CEA/feature alignment：

```text
feature_before = replay/source features saved at an earlier epoch
feature_latter = current replay/source features
loss_cea = KL(log_softmax(feature_latter), softmax(feature_before))
```

它更像相邻训练阶段的 replay feature consistency，不是全局分布对齐。

这带来一个安全问题：如果攻击每次只让分布移动一点点，并且每次移动都在 CEA 可接受范围内，那么相邻对齐可能不会报警，但长期会发生累积漂移。

攻击目标可以从：

```text
一次性拉崩模型
```

改成：

```text
每个 subject / 每个 CL step 只移动 feature centroid 一小步，
让 CEA 只看到相邻状态差异很小，
但多步累计后 old/new 表征和分类边界偏移。
```

## 5. 新攻击目标：Stealth Proxy Drift

目标：

```text
生成能通过 confidence filter 的污染 EEG/EOG；
每一步只轻微拉偏 feature/alignment distribution；
污染样本不能大量被过滤；
clean ACC/MF1 初期不能快速崩；
多 subject 累计后 old/generalization 预测发生偏移。
```

攻击者假设：

- proxy model 可白盒访问；
- 当前模型参数、teacher/guiding model、buffer 更新逻辑可见；
- 不考虑用户 budget；
- 但攻击要隐蔽，不能让置信度过滤直接挡掉大多数样本。

## 6. 代理攻击优化目标

对每个 new subject，先在 clean 输入上得到：

```text
y_pseudo = argmax(M_teacher(x))
conf_clean = max prob(M_teacher(x))
z_clean = feature_encoder(feature_extractor(x))
mu_clean = mean(z_clean)
```

维护一个长期漂移方向：

```text
d_t = normalized accumulated drift direction
mu_target_t = mu_clean + eta * d_t
```

PGD 生成污染样本时，不再直接用 NHE 让输出崩，而是优化：

```text
min_delta
    lambda_conf * CE(M_teacher(x + delta), y_pseudo)
  + lambda_pass * ReLU(tau - confidence(M_teacher(x + delta)))
  + lambda_raw  * ||raw_stats(x + delta) - raw_stats(x)||^2
  + lambda_l2   * ||delta||^2
  + lambda_align * ||mean(z_adv) - mu_target_t||^2
  - lambda_drift * <mean(z_adv) - mu_clean, d_t>

s.t.
    ||delta||_inf <= eps
    sequence passes confidence filter
```

直觉：

- `lambda_conf`：保持原 pseudo label，不让置信度掉太多。
- `lambda_pass`：强制通过置信度阈值。
- `lambda_raw`：原始数据统计不要明显异常。
- `lambda_align`：把 feature centroid 拉到目标小偏移位置。
- `lambda_drift`：沿长期攻击方向持续推进。

这样污染样本更像“正常 subject drift”，而不是明显 OOD。

## 7. 选择攻击样本

不要攻击所有 sequence。应只攻击高置信、易通过过滤的样本：

```text
candidate = sequence with clean confidence pass
attack only top-k confidence sequences
after PGD, keep only samples still passing confidence
if pass rate < target_pass_rate, reduce eps/lambda_drift
```

建议目标：

```text
target_pass_rate >= 50% of clean pass rate
relative perturbation <= 1%-2% batch std
per-step feature centroid shift below clean subject-to-subject natural shift percentile
```

## 8. 稳定集不应一次性固定

stable set 应该在线更新：

```text
每 K 个 subject:
  用最近 K 个 clean/accepted checkpoints 计算 low-drift units
  用 source/replay data 计算 Fisher important units
  stable_core = low-drift ∩ high-Fisher
```

防御侧监控：

```text
stable_core_update_ratio
fisher_weighted_drift
gradient_projection_on_stable_core
```

攻击侧如果想隐蔽：

- 避免一次性强推 stable core；
- 优先利用 shared-but-plastic important units；
- 让 stable_core drift 每步低于阈值，但长期方向保持一致。

## 9. 推荐可视化方案

EEG 场景有时间变化和个体差异，单张 scatter 很难解释。建议使用多视图：

### View A: Raw Input

```text
raw_signal_tsne.png
per-subject signal stats
bandpower / mean / std / quantile
```

回答：污染输入在原始数据统计上是否明显异常。

### View B: Model Feature

```text
feature_tsne_by_checkpoint.png
centroid_shift_from_pretrain.png
old_new_distance_by_checkpoint.png
```

回答：同一输入经过不同 checkpoint 后，表征是否被 CL 或攻击拉偏。

### View C: Time / Subject Trajectory

每个 subject 一个小图：

```text
x-axis: sequence id or sleep epoch time
y-axis: confidence / feature drift / stable-core drift
color: sleep stage pseudo label
```

回答：攻击是否集中在某些睡眠阶段或某些时间段。

### View D: CL Step Trajectory

```text
x-axis: new subject step
y-axis:
  clean ACC/MF1
  buffer added
  confidence pass rate
  feature centroid drift
  stable-core update ratio
  ASR or target bias score
```

回答：慢性攻击是否逐步累积，而不是一步崩。

### View E: Gradient

```text
gradient_cosine.png
gradient_module_norms.png
stable-core gradient projection
```

回答：污染数据的梯度方向是否偏离正常 CL，但又没有大到容易检测。

## 10. 为什么继续使用 t-SNE

t-SNE 更适合做局部簇结构展示，适合回答：

```text
clean 和 poisoned 是否混在一起？
不同 subject 是否自然分开？
attack 后 feature 是否逐步偏离 clean cluster？
```

PCA 更适合看全局线性趋势和可重复定量，t-SNE 更直观但不适合直接读距离大小。后续报告应同时给：

- t-SNE scatter：人能看懂簇；
- centroid L2 / MMD / Wasserstein：做定量；
- confidence pass：说明是否能过过滤；
- old ACC/MF1 / ASR：说明攻击效果。

## 11. 下一步实现顺序

建议按这个顺序做：

1. 在 `attack_diagnostics.py` 中保留当前 `NHE PGD` 作为强基线。
2. 新增 `stealth_drift_pgd`，使用 confidence-preserving + feature-centroid drift objective。
3. 只攻击 clean confidence 已通过的 sequence。
4. 在 full CL runner 中新增独立 attack mode，不修改原始 BrainUICL CL 代码。
5. 每步保存：

```text
pass_rate_clean
pass_rate_poisoned
feature_centroid_shift
stable_core_update_ratio
buffer_added
old ACC/MF1
target bias / ASR
```

6. 用 `distribution_trajectory.py` 生成每轮攻击后的轨迹图。

最终要证明的不是“模型能被打崩”，而是：

```text
污染数据多数能通过过滤；
每一步分布偏移都接近正常 subject drift；
多步累计后 old/generalization prediction 出现系统性偏移；
对齐机制只看相邻分布时难以及时发现。
```
