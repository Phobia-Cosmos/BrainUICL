# BrainUICL 中稳定神经元与 CL 安全性的分析

生成日期：2026-07-05

参考论文：

```text
/home/undefined/Desktop/bci/papers/CL_TTA_security/CL_vulnerabilities/2025USENIX-Persistent_Backdoor_Attacks_in_Continual_Learning.pdf
```

配套实证输出：

```text
experiments/stable_neuron_analysis/full49_model_nhe_seed4321/stable_neuron_analysis.md
experiments/stable_neuron_analysis/full49_model_nhe_seed4321/stable_neuron_analysis.json
```

## 1. USENIX 论文中的稳定神经元是什么

这篇 Persistent Backdoor Attacks in Continual Learning 的核心观察是：

- CL 模型在连续学习不同任务时，并不是所有参数都会同等变化。
- 一小部分对早期任务重要的神经元，在后续任务训练中会保持较小的权重变化。
- 攻击者如果把后门绑定到这些稳定且重要的组件上，后门行为更容易跨后续任务保留下来。
- 论文用参数变化、层级变化、神经元变化、Fisher 重要性和任务间重要神经元 IoU 来支持这一点。

论文中的任务设置主要是 SplitMNIST / SplitCIFAR / PermutedMNIST 这类显式 task-incremental 或 class-incremental 场景。不同任务之间的类别或输入分布差异很大，所以它观察到“不同任务主要依赖不同神经元集合”，并且早期任务的重要神经元在后续任务中保持稳定。

## 2. BrainUICL 中是否也存在这种稳定神经元

结论：存在，但含义和论文中不完全一样。

我用我们已经跑完的 `full49_model_nhe_seed4321` checkpoints 做了参数级分析。定义如下：

```text
relative_change = ||w_t - w_0||_2 / (||w_0||_2 + eps)
```

对 Conv / Linear / Norm 参数，按输出通道或输出行近似为一个“神经元”。每个参数张量里变化最低的 10% 记为 stable units。

clean 分支从 Pretrain 到 49 个 new subjects 后的结果：

| module | units | median final relative change | frac < 0.01 | frac < 0.05 | frac < 0.10 |
|---|---:|---:|---:|---:|---:|
| feature_extractor | 8576 | 0.005263 | 0.660 | 0.785 | 0.802 |
| feature_encoder | 20480 | 0.006611 | 0.566 | 0.770 | 0.817 |
| sleep_classifier | 773 | 0.001828 | 0.900 | 0.983 | 0.990 |

这说明 BrainUICL 的 clean CL 过程中确实存在大量低漂移参数单元。尤其是 sleep classifier 的中位变化非常小，feature extractor 和 Transformer encoder 中也有明显稳定子集。

但需要注意：这里的“稳定”首先是参数变化小，不等价于“对任务最重要”。论文中的强版本还会用 Fisher 重要性筛选“既重要又稳定”的神经元。我们这里也做了小样本 Fisher 交叉检查，但还不是完整攻击级 Fisher 实验。

## 3. 稳定神经元和历史任务、新任务的关系

BrainUICL 的任务划分和 USENIX 论文不同。

在我们的 ISRUC 睡眠 EEG 设置中：

- old / train / val / new 都是同一套睡眠分期标签：Wake、N1、N2、N3、REM。
- new task 不是新类别，而是新 subject / 新个体。
- 所以这是更接近 domain-incremental / subject-incremental 的 CL，而不是 class-incremental CL。

因此 BrainUICL 中稳定神经元更可能对应：

- 跨个体共享的睡眠阶段特征；
- EEG/EOG 中相对稳定的时序模式；
- 预训练阶段已经学到的通用表征；
- 用于保持 old/generalization 性能的共享分类边界。

它们不太像 SplitCIFAR 中“Task 1 的类别专属神经元”。换句话说，我们这里的稳定性更偏向“同一任务语义下的跨 subject 稳定表征”，而不是“不同任务之间互不相交的神经元集合”。

Fisher 小样本结果也支持这一点：

| metric | value |
|---|---:|
| old/new top-important unit Jaccard | 0.4137 |
| stable units in old top-important fraction | 0.1573 |
| stable units in new top-important fraction | 0.1542 |
| stable vs old/new important union Jaccard | 0.1011 |

解释：

- old 和 new 的重要神经元集合有较高重叠，平均 Jaccard 约 `0.414`。
- 这和 USENIX 论文里不同任务重要神经元 overlap 很低的现象不同。
- 这符合我们的任务性质：不同 subject 仍然解决同一个睡眠分期问题，所以会共享一批重要神经元。
- stable units 和 old/new top-important units 有一部分重叠，但不是完全重合；也就是说，很多重要单元仍然会为了新个体适配而变化。

## 4. 稳定神经元和 replay buffer 是否有关系

有间接关系，但不是论文里 A-GEM / DGR 那种显式 replay 关系。

BrainUICL 的 buffer 机制不是标准 supervised replay buffer：

- 初始 buffer 包含预训练/历史训练数据；
- 后续 new subject 到来后，模型生成高置信 pseudo labels；
- 通过置信度过滤的样本会加入动态 buffer；
- joint training 时新个体样本和 buffer 样本一起训练。

因此 buffer 的作用是通过数据混合和伪标签约束梯度方向，间接降低模型漂移。它没有像 EWC/SI 那样显式惩罚某些参数变化，也没有像 XdG 那样给不同任务分配 gating mask。

实证相关性：

| variant | corr(global delta, buffer added) | corr(stable delta, buffer added) | corr(global delta, old MF1) | final buffer length |
|---|---:|---:|---:|---:|
| clean | 0.134 | -0.100 | -0.024 | 2341 |
| attack_model_nhe | NA | NA | 0.323 | 1030 |

解释：

- clean 分支里，global 参数漂移和 buffer 新增量只有弱正相关，说明新个体适配越明显时，可能产生更多高置信伪标签，但关系不强。
- stable-core 漂移和 buffer 新增量是弱负相关，说明稳定单元越不动时，buffer 反而更容易正常增长；这符合“稳定表征支持可靠 pseudo labels”的直觉。
- attack 分支中 buffer 最终停在 `1030`，几乎不再新增伪标签。这和我们前面的 full attack 结果一致：模型退化后无法通过置信度过滤，动态 buffer 的自训练闭环被破坏。

所以 buffer 和稳定神经元之间不是“buffer 直接保护某些神经元”的关系，而是：

```text
稳定共享表征 -> 更可靠的伪标签 -> buffer 正常增长 -> 训练梯度更受历史/伪历史约束 -> 进一步降低灾难遗忘
```

攻击或严重 domain shift 会破坏这个链路：

```text
模型输出退化 -> 置信度过滤失败 -> buffer 不增长 -> 历史约束减弱 -> CL 性能继续退化
```

## 5. 同一类任务中还会出现稳定性吗

会，而且原因可能更强。

USENIX 论文讨论的是不同任务之间仍然存在稳定神经元。我们的场景是更保守的同一标签空间、同一任务语义，只是 subject 发生变化。由于睡眠分期的类别定义不变，模型有更强理由保留一批通用 EEG/EOG 表征。

但这里的稳定性不应解释成“每个 subject 有一组独立稳定神经元”。更合理的解释是：

- 有一批 domain-invariant 单元服务于所有 subject；
- 有一批 plastic 单元负责适配 subject-specific 信号幅值、噪声、睡眠结构比例、EOG/EEG 分布；
- 还有一些 classifier / attention 单元在特定 subject 上会发生较大调整。

这也意味着，BrainUICL 中的后门风险形式和 USENIX 论文不同：

- 在 SplitCIFAR 中，攻击可以绑定到某个 task 的稳定重要神经元上。
- 在 BrainUICL 中，攻击更可能绑定到跨 subject 共享的睡眠表征、通道伪影、特定频段或伪标签机制上。
- 如果触发器能激活共享稳定表征，它可能跨多个 subject 持久存在。
- 如果触发器只依赖某个 subject 的噪声模式，它可能不稳定。

## 6. 为什么 BrainUICL 没有直接对比 SI、EWC、XdG、LwF、DGR、A-GEM

USENIX 论文列出这些算法，是因为它要证明 persistent backdoor 对不同 CL 类型都有效：

- 正则化类：SI、EWC、XdG、LwF
- 重放类：DGR、A-GEM

BrainUICL 的目标不同。它解决的是 EEG 应用中的无监督个体连续学习，关键约束是 new individual 没有人工标签。它的核心流程是：

- source/pretrain model；
- new subject 到来；
- CPC 自监督 teacher adaptation；
- guiding model 生成 pseudo labels；
- 置信度过滤；
- dynamic buffer；
- 新个体更新和历史保持。

这些设定和传统 CL benchmark 不同，所以原论文没有简单把所有经典 CL 算法都搬过来对比。尤其是：

- EWC / SI 需要估计参数重要性，通常依赖有标签任务损失。
- XdG 通常需要明确 task identity，并给不同任务使用不同 gate。
- LwF 依赖 teacher-student distillation，比较适合同标签空间，但仍要处理无标签 new subject。
- A-GEM 需要带标签或可靠伪标签的 episodic memory，并做梯度投影。
- DGR 需要训练生成模型复现旧数据分布，对 EEG/EOG 时序信号成本和可靠性都更高。

## 7. 这些算法放到 EEG / BrainUICL 场景是否可行

可行，但需要改造成无监督或伪标签版本。

| algorithm | EEG 场景可行性 | 主要问题 | 和 BrainUICL 的关系 |
|---|---|---|---|
| EWC | 可行 | 需要用 source/old 数据或 pseudo labels 估计 Fisher；lambda 难调 | 可作为参数正则项，保护稳定重要神经元 |
| SI | 可行 | 在线路径积分需要可靠 loss；无标签时依赖伪标签 | 可和 BrainUICL joint training 结合 |
| XdG | 较弱可行 | 需要 subject/task identity；每个 subject 一个 gate 会很重 | EEG 中更像 subject-specific adapter，不太自然 |
| LwF | 很可行 | teacher 质量决定效果；teacher 若被攻击会传播错误 | BrainUICL 已经有 teacher/guiding model 思想，最自然的 baseline |
| A-GEM | 可行 | 需要 memory labels；梯度投影成本更高 | BrainUICL buffer 类似 memory，但没有 A-GEM 投影 |
| DGR | 理论可行，工程较难 | 需要生成 EEG/EOG 序列，生理可信度难保证 | 可作为研究 baseline，但实现成本最高 |

如果我们要把 BrainUICL 写成安全/鲁棒性方向的工作，建议优先做：

1. BrainUICL + EWC-style Fisher regularization。
2. BrainUICL + LwF distillation。
3. BrainUICL + A-GEM-like gradient projection using pseudo-label buffer。

DGR 和 XdG 可以放到次优先级：DGR 工程成本高，XdG 需要明确 task gate，在真实新个体部署里假设较强。

## 8. 对我们后续攻击实验的启发

当前 `model_nhe` 攻击已经证明：在 white-box/no-budget 设置下，代理目标可以让 BrainUICL 出现严重退化。但这不是 persistent backdoor 的完整复现。

如果要进一步验证 USENIX 论文思想在 BrainUICL 中是否成立，应做一个 EEG 版 stable-neuron backdoor：

1. 用 old/train 数据或 pretrain checkpoint 计算 Fisher 重要性。
2. 选出既重要又低漂移的稳定单元，优先关注 feature extractor / fusion / Transformer 表征层。
3. 设计 EEG/EOG 触发器，例如特定通道上的微弱频段扰动、短时 pattern、EOG 伪影 pattern。
4. 在某个 new subject 更新时，把触发器和目标睡眠阶段绑定。
5. 后续 subject 正常 clean update。
6. 评估每一步 clean ACC/MF1、ASR、buffer 增长、stable units 漂移。

关键评价不是只看模型是否崩，而是看：

```text
clean performance 是否保持
triggered ASR 是否跨后续 subjects 保持
后门是否绑定在稳定且重要的 BrainUICL 表征单元上
buffer 是否会放大或抑制后门
```

## 9. 当前结论

BrainUICL 中确实存在稳定神经元/稳定参数单元，但它们和 USENIX 论文中的 task-specific stable neurons 不完全同构。

更准确的说法是：

```text
BrainUICL 有跨 subject 的稳定共享表征；
这些表征服务于同一睡眠分期任务的历史和新个体；
buffer 通过伪标签 replay 间接维持这些表征；
攻击若能污染这些共享稳定表征或伪标签链路，就可能产生持续影响。
```

因此，USENIX 的稳定神经元思想可以迁移到 BrainUICL，但攻击目标应该从“不同类别任务的稳定神经元”调整为“跨个体共享睡眠表征 + 动态 buffer 伪标签链路”。
