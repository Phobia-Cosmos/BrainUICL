# SPR-EEG Continual Learning Defense Report

Date: 2026-07-12

This report adapts Self-Purified Replay (SPR, ICCV 2021) to the
BrainUICL ISRUC individual continual-learning protocol and evaluates both its
useful operating range and its failure modes.

## 1. Experimental Scope

- Dataset: ISRUC subgroup I, 98 available subjects.
- Input: sequences of 20 sleep epochs, each with 2 EOG + 6 EEG channels.
- Task: five-class sleep staging.
- Split seed: 4321.
- Probe stream: the first 10 BrainUICL new individuals.
- Training budget: 3 CPC epochs and 3 joint incremental epochs per individual.
- Pretrained model and source replay data are identical across variants.
- Metrics: final old/generalization ACC, MF1, average accuracy (AAA), average
  MF1 (AAF1), forgetting rate (FR), new-individual plasticity, and replay-label
  error measured with held-out ground truth.

Ground-truth labels are only used to report replay-label error. The defense
does not read them.

## 2. Extracted SPR Method

The original image SPR method contains two networks and two memories:

1. Delayed buffer `D`: temporarily holds the current data stream.
2. Purified buffer `P`: stores samples judged likely to have clean labels.
3. Expert network: trained self-supervised on `D`, then used to compute
   class-conditional feature centrality.
4. Base network: trained with self-supervised replay on `D union P`.
5. Self-Centered Filter: constructs one feature-similarity graph per observed
   class, estimates stochastic eigenvector centrality, fits a two-component
   Beta mixture, and interprets the high-centrality posterior as cleanliness.
6. Downstream inference: supervised training only uses purified memory.

## 3. EEG Mapping

| SPR component | BrainUICL / EEG implementation |
| --- | --- |
| Incoming task-free stream | Sequential unseen ISRUC individuals |
| Delayed buffer | All sequences from the current individual |
| Observed noisy label | BrainUICL teacher pseudo-label for each sleep epoch |
| Expert self-supervision | BrainUICL CPC adapted only on the current individual |
| Base Self-Replay | Optional CPC on current data plus sampled replay data |
| Class graph vertex | One 30-second EEG epoch embedding |
| Class graph grouping | Predicted sleep-stage pseudo-label |
| Edge weight | Non-negative cosine similarity between expert embeddings |
| Stochastic ensemble | Five sampled similarity graphs per sleep stage |
| Clean posterior | Two-component Beta-mixture posterior over centrality |
| Purified memory unit | A 20-epoch EEG sequence |

Sequence acceptance first requires BrainUICL's original confidence rule
(`15/20` epochs with confidence at least `0.9`). The epoch clean posteriors are
then aggregated into a sequence score. A ranked minimum-acceptance fallback
keeps the highest-centrality 75% of candidates when an absolute threshold
would remove too much individual or class coverage.

## 4. Implementation

- `model/spr_eeg.py`
  - stochastic graph construction;
  - power-iteration eigenvector centrality;
  - guarded two-component Beta-mixture EM;
  - epoch-to-sequence purification.
- `experiments/rttdp_brainuicl_full.py`
  - `--defense-mode spr`;
  - optional EEG Self-Replay;
  - SPR buffer filtering and purity diagnostics;
  - reproducible symmetric buffer-label noise;
  - clean, noisy, and adaptive-attack variants with reset random seeds.
- `tests/test_spr_eeg.py`
  - verifies that pseudo-label/feature-cluster mismatches receive lower clean
    probabilities;
  - verifies input-shape validation.

## 5. Main Results

### 5.1 Clean stream and 40% random buffer-label noise

| Variant | ACC | MF1 | AAA | AAF1 | FR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BrainUICL clean | 0.6943 | 0.6601 | 0.7089 | 0.6876 | 0.0117 |
| SPR ranked clean | 0.7005 | 0.6666 | 0.7059 | 0.6841 | 0.0028 |
| BrainUICL + 40% noise | 0.7005 | 0.6734 | 0.7091 | 0.6886 | 0.0028 |
| Full SPR, strict filter | 0.6861 | 0.6515 | 0.7029 | 0.6817 | 0.0233 |
| Full SPR, relaxed filter | 0.6775 | 0.6416 | 0.6923 | 0.6661 | 0.0355 |
| SPR ranked filter-only | **0.7059** | **0.6805** | 0.6980 | 0.6761 | 0.0049 |

The ranked filter-only variant improves final noisy-stream ACC by 0.54
percentage points and MF1 by 0.71 points over noisy BrainUICL. It also has no
measurable clean-stream penalty in this probe.

However, the final gain is small and AAA/AAF1 are lower. The defense changes
the intermediate trajectory and does not dominate BrainUICL at every step.

### 5.2 Purification diagnostics

| Variant | Mean error before | Mean error after | Accepted / candidates |
| --- | ---: | ---: | ---: |
| Strict SPR | 0.5569 | **0.4339** | 93 / 191 |
| Ranked SPR | 0.5304 | 0.5196 | 172 / 227 |

Strict filtering removes substantially more noisy labels, but loses too much
EEG individual/class coverage and hurts classification. Ranked filtering gives
up most of the purity gain to preserve diversity, producing better final
accuracy. This is the central purity-diversity tradeoff on EEG.

### 5.3 Adaptive proxy-meta poisoning

| Variant | ACC | MF1 | AAA | AAF1 | FR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BrainUICL clean | 0.6943 | 0.6601 | 0.7089 | 0.6876 | 0.0117 |
| BrainUICL proxy-meta | 0.6195 | 0.5684 | 0.5774 | 0.5146 | 0.1181 |
| SPR ranked proxy-meta | **0.5495** | **0.4874** | 0.5749 | 0.5130 | 0.2178 |

SPR does not defend this attack. Mean pseudo-label error changes from 0.7432
before filtering to 0.7464 after filtering. The attacker moves many samples
into a coherent, high-confidence wrong cluster, violating SPR's assumption
that clean samples form the largest central feature cluster inside each label.
Filtering then retains the attack cluster and removes useful diversity.

The stronger direct `model_nhe` diagnostic similarly collapses SPR ranked to
ACC 0.2270 and FR 0.6768. A replay-purification defense is not expected to stop
an attacker that directly changes model updates.

## 6. Interpretation

The extracted method is useful as a narrow label-noise defense:

- It identifies isolated pseudo-label/feature mismatches.
- It can measurably increase replay purity.
- With diversity-preserving ranked selection, it provides a small final gain
  under random buffer-label noise without harming clean final accuracy.

It is not a general poisoning defense:

- Full Self-Replay directly transferred from image SPR causes EEG feature
  drift under this short BrainUICL budget.
- Absolute purity thresholds remove too many subject-specific sequences.
- Centrality cannot identify a coherent adversarial cluster whose labels,
  confidence, and features have all moved together.

The practical configuration from this probe is therefore `SPR ranked
filter-only`, not the literal full image SPR recipe. A stronger EEG defense
would need temporal consistency, source-anchor distances, class/subject quotas,
and an explicit detector for coherent distribution shifts.

## 7. Reproduction Commands

Environment:

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python
```

BrainUICL clean/noisy plus SPR noisy:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --output-root experiments/rttdp_brainuicl_runs/probe10_spr_buffer_noise40_e3_seed4321 \
  --max-subjects 10 --ssl-epoch 3 --incremental-epoch 3 --cross-epoch 2 \
  --batch 16 --num-worker 0 \
  --attack-mode buffer_label_noise --buffer-label-noise-rate 0.40 \
  --defense-mode spr --no-save-checkpoints
```

Ranked SPR filter-only:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --output-root experiments/rttdp_brainuicl_runs/probe10_spr_filter_ranked_noise40_e3_seed4321 \
  --max-subjects 10 --ssl-epoch 3 --incremental-epoch 3 --cross-epoch 2 \
  --batch 16 --num-worker 0 --run-defense-only \
  --attack-mode buffer_label_noise --buffer-label-noise-rate 0.40 \
  --defense-mode spr --spr-disable-self-replay --spr-min-accept-ratio 0.75 \
  --no-save-checkpoints
```

Proxy-meta comparison:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --output-root experiments/rttdp_brainuicl_runs/probe10_spr_proxy_meta_ranked_e3_seed4321 \
  --max-subjects 10 --ssl-epoch 3 --incremental-epoch 3 --cross-epoch 2 \
  --batch 16 --num-worker 0 --attack-mode proxy_meta_conflict \
  --proxy-meta-poison-scope individual --proxy-meta-steps 5 \
  --proxy-meta-eps-scale 0.50 --proxy-meta-param-scope classifier \
  --proxy-meta-conflict-weight 5.0 --proxy-meta-confidence-weight 0.1 \
  --proxy-meta-grad-norm-weight 0.0 --proxy-meta-raw-weight 0.001 \
  --proxy-meta-l2-weight 0.0005 --pgd-random-start \
  --defense-mode spr --spr-disable-self-replay --spr-min-accept-ratio 0.75 \
  --no-save-checkpoints
```

## 8. Result Locations

- `experiments/rttdp_brainuicl_runs/probe10_spr_buffer_noise40_e3_seed4321`
- `experiments/rttdp_brainuicl_runs/probe10_spr_filter_ranked_noise40_e3_seed4321`
- `experiments/rttdp_brainuicl_runs/probe10_spr_filter_ranked_clean_e3_seed4321`
- `experiments/rttdp_brainuicl_runs/probe10_spr_proxy_meta_ranked_e3_seed4321`

## 9. 原始 SPR 与 EEG 迁移 FAQ

### 9.1 SPR 原文是否使用 CPC

没有。原始 SPR 使用 SimCLR 风格的 NT-Xent 对比损失。每个输入生成两个
增强视图，同一样本的两个视图作为正样本，batch 内其他视图作为负样本。
SPR 的 expert 在 Delayed Buffer 上训练，base 在 Delayed Buffer 与 Purified
Buffer 的并集上训练。

对正样本视图 `(i,j)`，NT-Xent 为：

```text
L(i,j) = -log exp(sim(z_i,z_j)/tau)
              / sum_{k != i} exp(sim(z_i,z_k)/tau)
```

`z` 是 projection head 输出的 L2-normalized embedding，`sim` 为余弦
相似度，`tau` 是 temperature；SPR 配置使用 `tau=0.5`。一个 batch 的
`B` 个输入生成 `2B` 个视图，对每个视图分别计算损失。

EEG 版本使用 CPC，是因为 BrainUICL 已经提供了适合 EEG 时间序列的 CPC
自监督目标。这是模态迁移，不是 SPR 原文配置。CPC 负责学习特征，本身不
执行样本过滤。

### 9.2 SPR 原文的 buffer 大小

| Dataset | Delayed Buffer | Purified Buffer |
| --- | ---: | ---: |
| MNIST | 300 | 300 |
| CIFAR-10 | 500 | 500 |
| CIFAR-100 | 1250 | 5000 |
| WebVision | 1000 | 1000 |

原 SPR 的 Purified Buffer 是固定容量。满容量后，class-aware reservoir
先确定需要淘汰的类别，再优先删除该类别中 clean probability 最低的样本。

当前 EEG 对比实验为了保持 BrainUICL 的 memory protocol，没有额外施加
固定容量：初始 source buffer 为 1030 条 sequence；40% 噪声 BrainUICL
最终为 1231 条，strict SPR 为 1123 条，ranked SPR 为 1202 条。这是与
原 SPR 的明确差异。

### 9.3 EEG buffer 存储单位

BrainUICL 和当前 SPR-EEG buffer 都存储 sequence，而不是完整 subject，也
不是独立的 30 秒 epoch：

```text
one stored sequence = [20 epochs, 8 channels, 3000 samples]
stored label          = [20 epoch labels]
```

当前 subject 的全部 sequence 构成逻辑上的 Delayed Buffer。过滤图的顶点
是 sequence 内的 30 秒 epoch，但最终进入长期 replay 的单位仍是完整
20-epoch sequence。一个 subject 的 sequence 只会有一部分进入 replay。

### 9.4 Self-Centered Filter 基于哪些数据

原 SPR 只对当前 Delayed Buffer 建图和过滤，不会在每个步骤重新过滤整个
历史 Purified Buffer。流程为：

1. 所有新流样本先进入 Delayed Buffer。
2. expert 使用 Delayed Buffer 做自监督训练。
3. 按 Delayed Buffer 中的观测标签分组。
4. 每个类别分别建立相似图，计算 eigenvector centrality。
5. Beta mixture 将中心性转换为 clean posterior。
6. 按 clean posterior 随机选择进入 Purified Buffer 的样本。
7. base 使用 Delayed Buffer 与 Purified Buffer 做 Self-Replay。
8. 清空 Delayed Buffer，处理下一段数据流。

原文没有进入 Delayed Buffer 的置信度要求。Delayed Buffer 的作用正是先
隔离尚未判断是否干净的数据，积累足够的同类邻居后再决定是否进入长期
memory。

### 9.5 原文是否使用分类器置信度过滤

原文没有 `softmax >= 0.9` 或 `15/20 epochs` 规则。原 SPR 有带标签的数据
流，直接使用可能带噪的观测标签构图。它使用中心性 Beta mixture posterior
作为 clean probability，并通过 `clean_probability > Uniform(0,1)` 随机
接纳样本。

当前 EEG 版本的 `0.9 + 15/20` 是 BrainUICL 伪标签协议的前置门限：

```text
BrainUICL confidence gate -> SPR centrality filter -> replay buffer
```

### 9.6 为什么构建随机相似图

随机图不是与特征无关的随机连接。非负余弦相似度被当作边存在概率：

```text
edge(i,j) = 1 if cosine_similarity(i,j) > Uniform(0,1)
```

高相似样本更容易连接。原文采样五张图，分别计算中心性和 Beta mixture
posterior 后取平均，降低单次相似度误差、偶然边以及少量错标样本获得虚假
高中心性的影响。

### 9.7 EEG 中一张 graph 的范围

不是每条 sequence 的 20 个 epoch 单独构成一张图。当前实现先收集一个
subject 所有候选 sequence 中的高置信 epoch，再按预测睡眠阶段分为最多
五组，每个睡眠阶段建立一张图。

若一个 subject 有 `N` 条 sequence，则最多有 `20N` 个 epoch 顶点，分别
进入五个 class graph。中心性在当前 subject（逻辑 Delayed Buffer）范围内
计算，之后再把 epoch clean posterior 聚合成 sequence score。

### 9.8 old/new individual 是否都评估

是。每适配一个新 subject 后，都在固定 19 个 old/generalization subjects
上计算 ACC、MF1、AAA、AAF1 和 FR。对新 subject 则分别评估：

- initial：原始预训练模型；
- before：当前 continual model 在适配前；
- after：适配当前 subject 后。

40% buffer-label noise 的 10-subject 实验中：

| Method | Initial ACC | Before ACC | After ACC | After MF1 |
| --- | ---: | ---: | ---: | ---: |
| BrainUICL noisy | 0.5920 | 0.5885 | 0.6124 | 0.5302 |
| SPR ranked | 0.5920 | 0.5705 | 0.5898 | 0.5167 |

因此 ranked SPR 改善了最终 old ACC/MF1，但降低了新个体 plasticity。

### 9.9 strict SPR 与 ranked filter-only

| Setting | Strict SPR | Ranked filter-only |
| --- | --- | --- |
| Base Self-Replay | current + replay CPC | disabled |
| Absolute clean threshold | enabled | enabled |
| Minimum clean epochs | 12 | ranked fallback |
| Minimum acceptance | none in completed strict run | top 75% candidates |
| Accepted/candidates | 93/191 | 172/227 |
| Error before/after | 0.5569/0.4339 | 0.5304/0.5196 |
| Final ACC | 0.6861 | 0.7059 |

ranked filter-only 是 EEG 工程变体，不是原文 SPR。它同时改变了 Self-Replay
和过滤回退规则，因此不能视为单变量消融。

### 9.10 EEG 标签噪声如何生成

40% symmetric buffer-label noise 不修改 EEG/EOG 信号。student 为每个
sleep epoch 生成伪标签后，每个标签独立以 40% 概率替换成另外四类中的
随机类别，并保证替换后类别不同。噪声只在写入 replay 前注入，影响后续
subject 的 replay；当前 subject 已完成的即时更新不受该噪声影响。

BrainUICL baseline 直接保存这些噪声标签；SPR 在噪声注入后、写入 replay
前执行中心性过滤。随机过程由 seed、subject step 和 sequence index 固定。

## 10. 纯 SPR-EEG 与当前混合方案的边界

### 10.1 当前伪标签由谁生成

当前混合实现中，CPC 不生成监督标签：

1. CPC-adapted guiding/teacher model 在 joint update 中生成训练伪目标。
2. 完成当前 subject 适配后，student 生成最终 buffer 伪标签和置信度。
3. CPC-adapted expert 只提供 Self-Centered graph 的 EEG embedding。

原始 SPR 的 expert 同样不生成标签；原文直接使用数据流携带的观测标签。

### 10.2 为什么不能默认伪标签完全正确

未见个体存在明显 EEG domain shift，高 softmax confidence 不等于标签正确。
在没有人工注入噪声的 clean probe 中，通过 BrainUICL confidence gate 的
候选 sequence 仍观察到约 10% 到 25% 的 epoch 伪标签错误。若默认伪标签
完全正确，就无法评估 replay error accumulation，也是引入 SPR 的原因。

可以把伪标签视为“正常但带未知噪声的观测标签”，这与 SPR 的问题设定
一致；不应把它们视为 ground truth。

### 10.3 CPC 会不会导致保留数据太少

CPC 只在所有输入 `x` 上训练 representation，不会删除数据。造成数据过少
的是 BrainUICL confidence gate 和 strict SPR sequence threshold 的串联。

更接近原 SPR 的 EEG 方案应让所有新样本先进入 Delayed Buffer，CPC 在全部
输入上训练 expert，再由中心性决定进入 Purified Buffer 的概率。若必须使用
伪标签，可以降低或取消硬 confidence gate，改用 confidence soft weighting、
每类 top-k 或 ranked minimum acceptance，避免在构图前丢失太多样本。

### 10.4 subject 候选 sequence 如何生成

当前混合流程为：

1. 加载当前 subject 的全部 36 到 52 条 sequence。
2. 模型输出 `[N, 5, 20]` logits。
3. 对每个 epoch 计算 softmax confidence 和 pseudo-label。
4. 至少 15/20 个 epoch 的 confidence 不低于 0.9，sequence 才成为候选。
5. 噪声实验在这里对 epoch pseudo-label 注入对称噪声。
6. expert 为候选 epoch 生成 embedding，按 pseudo-label 构建 class graph。
7. epoch clean posterior 聚合成 sequence score。
8. strict threshold 或 ranked fallback 决定是否写入 replay。

### 10.5 不参考 BrainUICL 的纯 SPR-EEG

当前报告中的 0.7059 ACC 不是纯 SPR-EEG 结果。它仍复用了 BrainUICL 的：

- ISRUC source/new/old individual split；
- 预训练 EEG backbone；
- CPC guiding model；
- 0.9 与 15/20 confidence gate；
- pseudo-label joint update；
- old/new evaluation protocol。

在第 12 节正式实验完成前没有可报告的纯 SPR-EEG 数值；现在应以第 12 节
独立 runner 的结果为准，不能用前述混合结果代替。

纯 SPR-EEG 必须先确定监督条件：

- 若在线流提供 sleep-stage 标签：直接把可能带噪的观测标签作为 SPR 的
  `y`，不使用 guiding model 或 confidence gate；CPC/NT-Xent 只学习特征。
- 若在线流完全无标签：SPR 本身无法按类别构建 Self-Centered graph，仍需
  pretrained classifier、聚类或其他 pseudo-labeler，不能声称完全不依赖
  标签生成机制。

纯 SPR-EEG 已在第 12 节通过独立 runner 实现并运行。当前 BrainUICL+
SPR-ranked 的结果仍不能代替该数值。

## 11. 纯 SPR-EEG 实验协议（已锁定）

### 11.1 保持不变的 BrainUICL 协议

新实验固定使用 `seed=4321`，直接复用 `split_subjects()` 产生的四组 subject：

- `source/train`：仅用于已有 ISRUC 预训练模型和 source memory；
- `val`：不进入在线持续学习；
- `old/generalization`：每次适配后都重新评价；
- `new`：按原顺序逐个 subject 到达。

评价协议也保持不变。old subject 记录 ACC、MF1、AAA、AAF1、FR；每个 new
subject 分别记录 initial（原预训练模型）、before（上一时刻模型）和 after
（处理当前 subject 后的模型）的 ACC/MF1。这样纯 SPR 与 BrainUICL 的结果
具有相同 split 和评价口径。

### 11.2 被替换为原文 SPR 的部分

在线方法不再使用 BrainUICL guiding model、teacher pseudo-label、`0.9`
confidence threshold、`15/20` gate 或 pseudo-label joint update。当前 new
subject 的人工 sleep-stage 标签作为 SPR 的观测标签；噪声实验只在训练侧按
0%、20%、40% 或 60% 注入 symmetric label noise，评价始终使用原始标签。

每个 subject 是一个逻辑 Delayed Buffer，执行顺序与 SPR 一致：

1. 独立 expert 在当前 Delayed Buffer 上用 SimCLR NT-Xent 训练；
2. continual base 在当前 Delayed Buffer 和 Purified Buffer replay 上做
   NT-Xent Self-Replay；
3. expert embedding 按带噪观测标签分组，建立 `E_max=5` 个随机余弦相似图；
4. eigenvector centrality 和两分量 Beta mixture 得到 epoch clean posterior；
5. 按 `clean_probability > Uniform(0,1)` 接纳 epoch；
6. 从 base representation 复制 evaluation model，只用 Purified Buffer 的
   retained epoch mask 做监督 fine-tuning；
7. 执行相同的 old/new 评价。

这里的 NT-Xent temperature 固定为原文的 `0.5`。EEG 的两种 view 使用轻微
jitter、幅值缩放、时间遮挡和 channel dropout；CPC 不参与纯 SPR 实验。

### 11.3 EEG Purified Buffer 预算

Purified Buffer 固定为 5000 个 30 秒 epoch reference，其中 3000 个是
sequence-aware 抽样的 source epoch，2000 个是 new-subject 动态分区。动态
分区使用 class-aware replacement，并优先淘汰过量类别中 clean posterior
最低的 epoch。每条记录为：

```text
(sequence_path, epoch_index, observed_label, clean_probability)
```

模型加载时仍读取完整 `[20 epochs, 8 channels, 3000 samples]` sequence 作为
Transformer 上下文，但监督 loss 只作用于 retained epoch mask。这个设计使
过滤粒度与原 SPR 的 per-sample 语义一致，同时不破坏 BrainUICL backbone 的
20-epoch 输入约束。source 与 dynamic 容量分开，是为了避免 clean probability
为 1 的 source epoch 在 memory 满后永久挤掉所有 new epoch。

### 11.4 实现与结果边界

独立 runner 为 `experiments/spr_eeg_pure.py`。它可以复用 ISRUC loader、预训练
backbone、subject split 和评价函数，但不会调用 BrainUICL 的 continual
training 路径。纯 SPR 数值必须来自该 runner 的完整输出；第 8 节的 0.7059
仍是 BrainUICL+SPR-ranked 混合实验，不能改名为纯 SPR。

## 12. 纯 SPR-EEG 正式实验结果

### 12.1 运行配置

- new subject：固定顺序前 10 个，`64, 89, 1, 27, 60, 5, 52, 42, 80, 26`；
- old/generalization subject：固定 19 个，与 BrainUICL 完全相同；
- 训练轮次：expert NT-Xent 10、base NT-Xent 10、evaluation fine-tune 10；
- noise：clean（0%）和 symmetric 40%；
- memory：5000 epoch，其中 source 3000、dynamic 2000；
- temperature：0.5，Self-Centered graph ensemble：5；
- confidence gate、guiding model、pseudo-label joint update：均关闭。

正式输出在本地（由 `.gitignore` 排除）：

```text
experiments/rttdp_brainuicl_runs/pure_spr_10sub_e10_seed4321/
experiments/rttdp_brainuicl_runs/brainuicl_10sub_e10_noise40_seed4321/
```

两组纯 SPR 运行均完成 10/10 subjects；每组 stability 曲线包含 initial 加 10
次适配共 11 个点，最终 buffer 均严格满足 5000-epoch 容量。

### 12.2 与同轮次 BrainUICL 协议对照

| Method | Noise | Old ACC | Old MF1 | AAA | AAF1 | FR | New after ACC | New after MF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BrainUICL, 10 epochs | 0% | 0.6676 | 0.6247 | 0.7076 | 0.6837 | 0.0496 | 0.5946 | 0.5235 |
| Pure SPR-EEG, 10 epochs | 0% | **0.7123** | **0.6881** | **0.7153** | **0.6954** | **0.0140** | **0.6424** | **0.5746** |
| BrainUICL, 10 epochs | 40% | 0.6678 | 0.6238 | 0.7072 | 0.6833 | 0.0493 | 0.5942 | 0.5234 |
| Pure SPR-EEG, 10 epochs | 40% | **0.7324** | **0.7088** | **0.7155** | **0.6914** | **0.0426** | **0.6293** | **0.5608** |

在相同 subject 顺序和 10-epoch 预算下，纯 SPR-EEG 的 clean 最终 old ACC/MF1
比 BrainUICL 高 4.47/6.35 个百分点，40% noise 下高 6.46/8.50 个百分点。
clean 的 new after ACC/MF1 高 4.78/5.11 个百分点，40% noise 下高
3.52/3.74 个百分点。

这不是完全相同监督条件下的消融。BrainUICL 在线阶段使用 guiding/teacher
伪标签，且 40% noise 只污染写入 replay 的伪标签；纯 SPR 遵循原文的有标签
noisy stream 假设，当前 subject 的人工标签是观测标签，40% noise 在进入
Delayed Buffer 时注入。表格用于回答“保留 split 和评价协议、替换 CL 方法”
后的结果，不能解释成 SPR 在无标签适配条件下优于 BrainUICL。

### 12.3 过滤和 memory 诊断

| Diagnostic | Clean | 40% noise |
| --- | ---: | ---: |
| Delayed epochs | 8880 | 8880 |
| Injected noisy epochs | 0 | 3609 |
| Accepted epochs | 5624 | 4897 |
| Acceptance rate | 63.33% | 55.15% |
| Accepted noisy epochs | 0 | 约 1149 |
| Injected noise removed | - | **68.16%** |
| Final total buffer purity | 100% | 94.22% |
| Final dynamic partition purity | 100% | 85.55% |

40% 运行的实际随机噪声率为 `3609/8880=40.64%`。Self-Centered Filter 接纳
的 4897 个 epoch 中约 1149 个仍为错标，因此即时 accepted-set purity 约
76.5%；后续 class-aware replacement 使最终 2000 个 dynamic epoch purity
达到 85.55%。94.22% 是把 3000 个已知干净 source epoch 一起计算的总 purity，
不能用它夸大过滤器单独的效果。

### 12.4 与 SPR 原论文比较

原论文 Table 1 在 40% symmetric noise 下报告最终 accuracy：MNIST 86.7%、
CIFAR-10 43.0%；Table 3 报告注入噪声被过滤的比例：MNIST 96.5%、CIFAR-10
70.5%。本次 EEG 的 injected-noise removal 为 68.2%，接近论文 CIFAR-10 的
70.5%，但低于 MNIST 的 96.5%。

EEG 的 73.24% final old ACC 不能与论文的 86.7% 或 43.0% 直接比较，因为
数据模态、类别、任务构造、backbone、source pretraining 和评价集合都不同。
可比较的是方法行为：在 40.64% 输入噪声下，过滤器显著降低长期 memory 的
错标率；代价是只接纳 55.15% 的输入 epoch，且 new after ACC 从 clean 的
64.24% 降到 62.93%，仍存在 purity 与 plasticity 的权衡。

### 12.5 正式复现命令

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/spr_eeg_pure.py \
  --output-root experiments/rttdp_brainuicl_runs/pure_spr_10sub_e10_seed4321 \
  --max-subjects 10 --noise-rates 0.0 0.4 \
  --expert-epochs 10 --base-epochs 10 --ft-epochs 10
```

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --output-root experiments/rttdp_brainuicl_runs/brainuicl_10sub_e10_noise40_seed4321 \
  --max-subjects 10 --ssl-epoch 10 --incremental-epoch 10 --cross-epoch 2 \
  --batch 16 --attack-mode buffer_label_noise --buffer-label-noise-rate 0.40 \
  --no-save-checkpoints
```
