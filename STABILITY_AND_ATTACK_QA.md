# BrainUICL 稳定神经元与攻击诊断 QA

生成日期：2026-07-07

相关文件：

```text
experiments/stable_neuron_analysis.py
experiments/stable_neuron_analysis/full49_model_nhe_seed4321/stable_neuron_analysis.md
experiments/attack_diagnostics.py
experiments/attack_diagnostics/full49_model_nhe_seed4321/
experiments/attack_diagnostics/full49_model_nhe_seed4321_pgd_eps003/
experiments/attack_diagnostics/full49_model_nhe_seed4321_pgd_eps001/
```

## 1. 稳定性分析中的 units 是什么

这里的 `units` 不是一个严格生物学意义的神经元，而是工程近似：

- 对 `Conv1d.weight[out_channel, in_channel, kernel]`，一个 unit 是一个输出通道。
- 对 `Linear.weight[out_dim, in_dim]`，一个 unit 是一行输出神经元。
- 对 `BatchNorm / LayerNorm` 的一维参数，一个 unit 是一个 feature/channel 维度。
- 对 classifier 的 `sleep_stage_classifier.weight[5, 128]`，一行对应一个类别输出；它更像 class logit unit，不完全等同 hidden neuron。

这样做的原因是：论文里的稳定神经元分析本质上也是看某个神经元相关权重向量在 CL 过程中的变化。BrainUICL 里有卷积、Transformer、MLP，因此按“输出通道/输出行”切分是最直接的对应方式。

## 2. relative_change 的原理是什么

我们用：

```text
relative_change = ||w_t - w_0||_2 / (||w_0||_2 + eps)
```

含义：

- `w_0` 是某个 unit 在基线 checkpoint 中的参数向量。
- `w_t` 是后续 checkpoint 中同一个 unit 的参数向量。
- 分子表示这个 unit 变化了多少。
- 分母做归一化，避免大权重 unit 天然有更大绝对变化，使不同层、不同通道之间更可比。

这个公式能用来衡量“稳定性”，因为稳定神经元的直接表现就是跨任务后参数向量变化小。它不是唯一指标，但和 USENIX 后门论文的 weight variation 思路一致。

局限：

- 如果 `||w_0||` 很小，relative change 会被放大，所以报告里主要看 median、p10、p90、fraction < threshold，不解读 max/mean。
- 参数稳定不等于功能重要。一个 unit 变化小，可能是重要稳定单元，也可能是几乎没被使用的低活跃单元。
- 因此后面还需要 Fisher 或 activation/gradient 分析来判断“重要性”。

## 3. Fisher 重要性筛选是什么

Fisher 重要性近似衡量某个参数对任务 loss 的敏感度。常用对角 Fisher：

```text
F_i = E[(d loss / d theta_i)^2]
```

直觉：

- 如果某个参数的梯度平方长期很大，说明 loss 对它敏感。
- 改动这个参数更可能影响当前任务性能。
- Fisher 高的参数可以被视为“任务重要参数”。

论文的稳定后门思路是：先找对某个任务重要的参数/神经元，再看这些神经元后续是否稳定。真正危险的是“重要且稳定”的组件，而不是单纯低漂移组件。

我们现在做的是轻量版：

- stable：从 full CL trajectory 里选每个参数张量变化最低的 10% units。
- old/new important：用 final clean checkpoint，在 old_generalization 和 new_order subject 子集上做小样本 Fisher。
- overlap：看 stable units 和 old/new top Fisher units 的交集。

## 4. 是否可以在适应前先计算预训练稳定神经元，并用于偏移/攻击检测

可以，而且这是后续防御方向里最值得做的一条。

但要区分两件事：

1. `稳定性` 需要至少两个或多个 checkpoint 才能判断。
2. `重要性` 可以在一个 checkpoint 上用 Fisher/gradient 估计。

适应前可行流程：

```text
Pretrain 阶段保存多个 epoch checkpoint
-> 计算每个 unit 在 pretrain 后期的方差/relative drift
-> 计算 source/old 数据上的 Fisher 重要性
-> 取 low-drift + high-Fisher 的 stable-important core
-> CL 时监控每个 new subject batch 对这些 core units 的梯度或更新幅度
```

检测指标可以设计为：

```text
stable_core_grad_ratio = ||grad_on_stable_core|| / ||grad_all||
stable_core_update_ratio = ||delta_on_stable_core|| / ||delta_all||
fisher_weighted_drift = sum(F_i * (theta_t_i - theta_0_i)^2)
```

如果某个 batch 或某个 subject 对 stable-important core 的影响异常大，可以标记为强 domain shift 或潜在攻击。这个思路比只看输入分布更贴近 CL 安全，因为攻击未必在原始信号空间显眼，但可能在稳定核心参数上造成异常梯度。

当前限制：我们现有 pretrain 目录主要保存最终/最佳 checkpoint，没有完整 pretrain epoch trajectory。要做这条防御，最好重跑 pretrain 或修改 pretrain 保存策略，保留后期若干 checkpoint。

## 5. 为什么 stable 和 old/new 的重叠不是很高

因为 stable 和 important 是两个不同维度：

- stable：参数变化小。
- old/new important：当前 loss 对该 unit 敏感。

可能出现四类 unit：

| 类型 | 含义 |
|---|---|
| stable + important | 最像 USENIX 论文中的危险稳定核心 |
| stable + not important | 低漂移但可能没发挥关键作用 |
| plastic + important | 对 old/new 都重要，但需要随 subject 适配 |
| plastic + not important | 普通可变背景参数 |

我们的结果：

```text
old/new top-important unit Jaccard = 0.4137
stable units in old top-important fraction = 0.1573
stable units in new top-important fraction = 0.1542
```

解释：

- old 和 new 的重要 units 重叠不低，说明同一睡眠分期任务确实共享一批重要表征。
- stable 和 old/new important 的重叠只有约 15%，说明很多重要表征并不是完全固定的，而是会随 subject 适配。
- 这和 BrainUICL 的场景一致：new task 是新 subject/domain，不是新类别。模型需要保留一部分通用核心，同时允许另一部分重要特征做个体化调整。

## 6. stable 是如何生成的，什么情况下会变化

当前脚本中 stable 的生成方式：

```text
对每个参数张量：
  计算 Pretrain -> final CL checkpoint 的每个 unit relative_change
  选 relative_change 最低的 bottom 10% 作为 stable units
```

它会随这些因素变化：

- seed；
- new subject 顺序；
- clean vs attack；
- 使用 Pretrain、individual_1 还是 final 作为基线；
- 选择 bottom 5%、10% 还是 absolute threshold；
- 是否使用 Fisher 过滤；
- 模型是否经历强 domain shift 或攻击。

更严格的做法应该是 online 版本：

```text
在每个 CL step 前，根据历史 checkpoint 更新 stable-important set
然后监控当前 new subject 对这个 set 的影响
```

这样 stable set 不是一次性固定，而是可以随模型长期演化更新。

## 7. old/new 会针对同一类 neuron 变化，但这些 neuron 并不是 stable 吗

是的，这是当前结果最重要的解释之一。

old/new 都重要的 units 不一定稳定。它们可能是 shared-but-plastic units：

```text
同一批 feature/attention/MLP units 对 old 和 new 都重要，
但为了适配不同 subject，它们会持续小幅或中幅变化。
```

这并不矛盾。CL 的稳定性不要求所有重要参数完全不动，而是要求：

- 变化方向不破坏 old/generalization；
- 有 source replay / buffer 约束；
- pseudo labels 质量足够；
- teacher adaptation 不把模型带偏；
- CEA/feature alignment 维持历史表征；
- 学习率和 epoch 控制不要让单个 subject 过拟合。

所以 BrainUICL 的稳定性来自“稳定核心 + 受约束的可塑单元”共同作用，而不是所有重要神经元都不变。

## 8. RTTDP 迁移后，污染数据都通不过置信度过滤了吗

要先区分两种攻击：

### model_nhe

我们完整 full run 用的是 `model_nhe`。它不是输入级污染，不会生成一批明显偏移的 EEG/EOG 数据。

它做的是：

```text
在 CL 更新时额外加入 white-box 攻击目标，
把模型输出推向非当前预测类别的分布。
```

因此 attack 分支后期 buffer added = 0，不是因为“污染输入被置信度过滤识别”，而是因为模型本身已经退化，无法再对正常 new subject 产生足够高置信 pseudo labels。

### pgd_nhe / pgd_ble

这才是输入级污染。我们新做了 PGD 诊断，发现当前 NHE 目标确实会显著降低置信度通过率。

| eps_scale | subject | clean pass | PGD pass | clean conf | PGD conf |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 27 | 0.750 | 0.000 | 0.944 | 0.704 |
| 0.10 | 64 | 0.500 | 0.063 | 0.892 | 0.765 |
| 0.10 | 89 | 0.688 | 0.125 | 0.928 | 0.741 |
| 0.03 | 27 | 0.750 | 0.125 | 0.944 | 0.769 |
| 0.03 | 64 | 0.500 | 0.063 | 0.892 | 0.717 |
| 0.03 | 89 | 0.688 | 0.125 | 0.928 | 0.789 |
| 0.01 | 27 | 0.750 | 0.375 | 0.944 | 0.829 |
| 0.01 | 64 | 0.500 | 0.000 | 0.892 | 0.715 |
| 0.01 | 89 | 0.688 | 0.125 | 0.928 | 0.804 |

结论：

- `eps=0.10` 过强，容易被置信度过滤挡掉。
- `eps=0.03` 仍偏强。
- `eps=0.01` 开始出现部分通过，但 subject 之间差异很大。
- 当前 PGD 目标是让输出靠近 NHE，这会天然降低原模型置信度；如果目标是隐蔽攻击，需要加入 confidence-preserving 约束。

## 9. 是否应该让污染数据部分通过置信度过滤

是。你这个判断是对的。

如果攻击数据几乎完全通不过过滤，它更像粗暴 OOD/异常样本，不像隐蔽攻击。后续要优化成：

```text
既能通过 confidence filter，
又能让梯度方向逐渐偏移，
同时 clean ACC/MF1 不立刻崩。
```

可优化方向：

- 降低 `pgd_eps_scale`，例如 0.005、0.01。
- 只攻击原本高置信 sequence，而不是所有 sequence。
- 攻击目标从 NHE 改成 confidence-preserving target，例如保持 top-1 置信度但改变 hidden feature。
- 加入约束项：

```text
loss = attack_loss
     + lambda_conf * preserve_confidence_loss
     + lambda_feat * feature_distribution_loss
     + lambda_l2 * perturbation_norm
```

- 在 embedding PCA 空间限制 PGD 后样本不要远离 clean subject cluster。
- 做慢性攻击：每个 subject 只污染少量 sequence，避免一次性拉崩。

## 10. 正常数据和污染数据分布可视化

已生成图像：

```text
experiments/attack_diagnostics/full49_model_nhe_seed4321/embedding_pca.png
experiments/attack_diagnostics/full49_model_nhe_seed4321/signal_stats_pca.png
experiments/attack_diagnostics/full49_model_nhe_seed4321/confidence_filter.png
experiments/attack_diagnostics/full49_model_nhe_seed4321/perturbation_magnitude.png
```

弱扰动版本：

```text
experiments/attack_diagnostics/full49_model_nhe_seed4321_pgd_eps003/
experiments/attack_diagnostics/full49_model_nhe_seed4321_pgd_eps001/
```

图像含义：

- `embedding_pca.png`：模型表征空间中 clean vs PGD 的分布。
- `signal_stats_pca.png`：原始信号统计空间中 clean vs PGD 的分布。
- `confidence_filter.png`：不同 subject 的过滤通过率和平均置信度。
- `perturbation_magnitude.png`：PGD 相对扰动幅度。

## 11. 攻击梯度更新和正常 CL 梯度更新有什么区别

诊断脚本比较了 4 种梯度：

- `source_replay_ce`：历史/source replay 样本的监督 CE 梯度。
- `new_pseudo_ce`：new subject 上 teacher pseudo label 产生的正常 CL 梯度。
- `model_nhe_kl`：当前 full run 使用的模型级 NHE 攻击梯度。
- `pgd_pseudo_ce`：PGD 输入进入 pseudo-label 更新后的梯度。

`eps=0.10` 的 cosine 摘要：

```text
cos(source_replay_ce, model_nhe_kl) = -0.012
cos(new_pseudo_ce, model_nhe_kl)    =  0.267
cos(new_pseudo_ce, pgd_pseudo_ce)   =  0.036
```

解释：

- `model_nhe_kl` 与 source replay 几乎正交甚至轻微冲突，说明它不会帮助保留历史任务。
- `model_nhe_kl` 与 new pseudo 梯度有一定正相关，但它的目标不是正确 pseudo label，而是把输出推向非当前类别分布。
- `pgd_pseudo_ce` 与正常 new pseudo 梯度相似度很低，说明 PGD 后即使进入训练，也会显著改变更新方向。

模块梯度范数图：

```text
experiments/attack_diagnostics/full49_model_nhe_seed4321/gradient_cosine.png
experiments/attack_diagnostics/full49_model_nhe_seed4321/gradient_module_norms.png
```

## 12. 脚本中的 source replay 是什么

这里的 source replay 指 BrainUICL buffer 里来自历史训练集/source domain 的样本。

在 `BufferDataset` 中，每个 batch 会拼接两部分：

```text
new subject sequence
source/replay sequence
```

训练时：

- new 部分没有人工标签，用 teacher 产生 pseudo label，并经过置信度过滤。
- replay 部分有标签，参与 `loss_old`。
- 后续高置信 pseudo labels 也会加入 `train_paths`，成为 replay buffer 的一部分。

所以 BrainUICL 的 replay 不是纯 A-GEM memory，但功能相似：通过历史样本约束模型不要完全被当前 new subject 带偏。

## 13. PGD 级别扰动是什么

当前脚本中的 PGD 输入污染是：

```text
eps_eog = std(eog_batch) * pgd_eps_scale
eps_eeg = std(eeg_batch) * pgd_eps_scale
delta clipped to [-eps, eps]
```

然后每一步用梯度符号更新：

```text
delta = delta - step_size * sign(grad)
```

这里是 targeted PGD，目标是让输出靠近 `NHE/BLE` attack target。

因此 `pgd_eps_scale` 不是绝对电压单位，而是相对于当前 batch 信号标准差的扰动比例。比如：

- `0.10` 大约是 batch std 的 10%，当前看偏强。
- `0.03` 仍明显降低置信度。
- `0.01` 有部分 subject/sequence 可以通过，但还不够稳定。

## 14. 下一步建议

保留当前攻击流程作为 baseline：

```text
model_nhe: 强 white-box/no-budget 上界攻击
pgd_nhe eps=0.10/0.03/0.01: 输入级扰动基线
```

下一步优化方向：

1. 做 confidence-preserving PGD，让 poisoned data 能部分通过过滤。
2. 加 stable-core drift detector，监控稳定重要神经元上的异常梯度。
3. 同时记录 `clean ACC/MF1`、`ASR`、`buffer added`、`stable_core_update_ratio`。
4. 把攻击目标从“快速模型崩溃”改成“隐蔽、缓慢、可持续退化”。
