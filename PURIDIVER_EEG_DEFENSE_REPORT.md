# PuriDivER 在 EEG 持续学习上的提取、实现与对比

本文对应的 PuriDivER 不是一个新的 EEG backbone，而是一套面向污染数据流的 replay 防护方法。它的核心可以提取为四步：在线构建兼顾 purity 和 diversity 的有限 memory；根据样本对当前标签的 loss 用二成分 GMM 区分 clean/noisy；再根据预测不确定性把 noisy 样本分为可重标注和只适合无标签一致性学习的样本；最后用 clean 分类、soft relabel 和 consistency 三部分共同训练。

原方法的第一层 GMM 使用逐样本交叉熵。低损失高斯对应 clean component，高损失高斯对应 contaminated component：

$$
w_i=P(z_i=\mathrm{clean}\mid \ell_i)
$$

对于低 clean posterior 的样本，再计算不确定性并拟合第二个二成分 GMM。低不确定样本进入 relabel 集，高不确定样本进入 unlabeled 集：

$$
\begin{aligned}
\mathcal C &= \{i:w_i\ge \tau_c\}\\
\mathcal R &= \{i:w_i<\tau_c,\ q_i\ge \tau_u\}\\
\mathcal U &= \{i:w_i<\tau_c,\ q_i<\tau_u\}
\end{aligned}
$$

BrainUICL 的新受试者没有人工标签，因此 EEG 版不能直接照搬图像代码。这里保留 BrainUICL 的预训练模型、CPC teacher、受试者顺序、source replay、CEA 和评估协议，只把 PuriDivER 的防护逻辑接入 joint continual update 和 memory update。

EEG 版以一个 30 秒 sleep epoch 作为净化判断单位，以包含 20 个 epoch 的 sequence 作为 buffer 存储单位。当前受试者的 teacher pseudo label 相当于论文中的 observed label；source 区域的 1030 条 sequence 是人工真标签，强制视为 clean，不交给 GMM 怀疑；只有后续 pseudo-labeled replay 可能被净化。

新个体的不确定性同时考虑 teacher entropy 和 teacher/student 分歧：

$$
u_i=
\frac{H(p_t(x_i))}{\log C}
+
\operatorname{JS}(p_t(x_i),p_s(x_i))
$$

低不确定可疑样本的 soft target 使用 teacher/student 平均并做 temperature sharpening：

$$
\bar y_i=
\operatorname{Sharpen}
\left(
\frac{p_t(x_i)+p_s(x_i)}{2};T
\right)
$$

高不确定样本不使用其标签，而是对原始 EEG 和 amplitude scaling、Gaussian noise 后的 EEG 预测做一致性约束。每个 new/replay 分支的目标为：

$$
\mathcal L_D=
\frac{
\sum_{i\in\mathcal C_D}\operatorname{CE}(p_i,y_i)
+
\lambda_r\sum_{i\in\mathcal R_D}\operatorname{SCE}(p_i,\bar y_i)
}{
|\mathcal C_D|+\lambda_r|\mathcal R_D|
}
+
\eta
\frac{1}{|\mathcal U_D|}
\sum_{i\in\mathcal U_D}
\left\|p_i^{strong}-\operatorname{sg}(p_i^{weak})\right\|_2^2
$$

最终仍沿用 BrainUICL 的新数据/replay 权重：

$$
\mathcal L=
\alpha\mathcal L_{new}
+
(1-\alpha)\mathcal L_{replay}
$$

memory 默认上限为 1600，其中 source 的 1030 条始终保留。超过预算后，对 pseudo memory 计算 loss-GMM clean posterior，并在 dominant sleep stage 内按 purity-diversity 分数选择：

$$
s_i=
\lambda_p(1-w_i)
+
(1-\lambda_p)
\max_{j\in\mathcal M_c}
\frac{f_i^\top f_j}{\|f_i\|\|f_j\|}
$$

分数越低表示标签越可信且与已选样本越不冗余。

## 实验结果

数据为完整 ISRUC Group-I 98 subjects 预处理结果，使用 BrainUICL 的 seed 4321 预训练 checkpoint。10-sub probe 和 full 49-sub 均使用相同 train/val/old/new split、相同 new subject order、`ssl_epoch=10`、`incremental_epoch=10`、`batch=16`。

首先只考虑正常数据流，不施加攻击。这组实验直接回答“加入 PuriDivER 后，持续学习是否还能保持较好性能”。

| 数据流长度 | 方法 | final ACC | final MF1 | AAA | AAF1 | FR |
|---|---|---:|---:|---:|---:|---:|
| 10 subjects | BrainUICL | 0.6676 | 0.6247 | 0.7076 | 0.6837 | 0.0496 |
| 10 subjects | BrainUICL + PuriDivER | 0.6499 | 0.6175 | 0.6923 | 0.6711 | 0.0748 |
| 49 subjects | BrainUICL | 0.6569 | 0.6231 | 0.6934 | 0.6685 | 0.0649 |
| 49 subjects | BrainUICL + PuriDivER | **0.6947** | **0.6801** | 0.6760 | 0.6500 | **0.0111** |

10-sub 短数据流中，PuriDivER 的 final ACC 比 BrainUICL 低 `1.77` 个百分点，MF1 低 `0.72` 个百分点，说明在 memory 还没有积累足够污染时，GMM 净化和一致性约束会带来额外成本。

完整 49-sub 数据流中，PuriDivER 的 final ACC 提高 `3.78` 个百分点，final MF1 提高 `5.70` 个百分点，FR 从 `0.0649` 降至 `0.0111`。因此只看最终旧任务保持能力，加入防护后可以保持较好性能，而且最后一个检查点优于 BrainUICL。

但 PuriDivER 的 AAA 和 AAF1 分别低 `1.74` 和 `1.84` 个百分点。在 49 个增量检查点中，PuriDivER 的 ACC 只有 11 步高于 BrainUICL，MF1 只有 13 步高于 BrainUICL；最后 10 步的 ACC/MF1 平均差值仍为 `-2.90/-3.32` 个百分点。最后一步的提升并不是一条稳定的后期上升趋势，而是最终检查点表现较好。因此若问题是“整条持续学习曲线是否始终不弱于 BrainUICL”，当前答案是否定的；若问题是“加入防护后是否仍有可用的持续学习能力”，答案是肯定的，但有约 1.7 至 1.8 个百分点的全程平均性能成本。

两者的 memory 预算也不同。49-sub 结束时，原始 BrainUICL memory 增长到 2311 条 sequence；PuriDivER 将 memory 固定在 1600 条，其中 1030 条 source sequence 始终保留。最后一步 PuriDivER 对 pseudo replay 的 GMM clean-rate 估计为 `90.08%`。这说明它在使用约少 31% replay memory 的情况下仍取得更高 final ACC/MF1，但若要形成严格的算法消融，还应增加一个同样限制为 1600 条 memory、但不进行 GMM 净化的 BrainUICL baseline。

10-sub、60% buffer symmetric-like label noise：

| 方法 | final ACC | final MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| BrainUICL clean | 0.6676 | 0.6247 | 0.7076 | 0.6837 | 0.0496 |
| BrainUICL + 60% label noise | 0.6122 | 0.5672 | 0.7034 | 0.6788 | 0.1285 |
| BrainUICL + PuriDivER + 60% label noise | 0.6611 | 0.6318 | 0.6936 | 0.6726 | 0.0589 |
| BrainUICL + PuriDivER clean stream | 0.6499 | 0.6175 | 0.6923 | 0.6711 | 0.0748 |

在这个与原论文最匹配的标签污染设置下，PuriDivER 相比未防护攻击分支恢复 `+4.89` 个 ACC 百分点和 `+6.46` 个 MF1 百分点，FR 从 `0.1285` 降到 `0.0589`。代价是 clean stream 上 ACC 比原始 BrainUICL 低约 `1.77` 个百分点，说明净化不是零成本的。

10-sub、20% label noise 时，攻击本身几乎没有伤害 BrainUICL：final ACC 只从 `0.6676` 变为 `0.6627`，MF1 还略升。因此该档位不能用于证明防护收益；PuriDivER 的 final ACC/MF1 为 `0.6594/0.6303`。

完整 49-sub、60% label noise：

| 方法 | final ACC | final MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| BrainUICL clean | 0.6569 | 0.6231 | 0.6934 | 0.6685 | 0.0649 |
| BrainUICL + 60% label noise | 0.6650 | 0.6405 | 0.6933 | 0.6691 | 0.0534 |
| BrainUICL + PuriDivER + 60% label noise | 0.6927 | 0.6768 | 0.6810 | 0.6566 | 0.0139 |

full stream 中随机 label noise 没有稳定破坏 BrainUICL，未防护攻击分支的 final ACC/MF1 反而高于 clean，因此不能把 PuriDivER 的 final 提升完全解释成“恢复攻击损失”。防护版 final ACC/MF1 和 FR 更好，但 AAA/AAF1 比未防护低约 `1.23/1.26` 个百分点，说明中间阶段存在适应性成本。

full run 的机制诊断更有意义：最后一步 loss-GMM 将 pseudo replay 的 clean rate 估计为 `35.4%`，和 60% 噪声下理论约 40% clean 接近；固定 memory 最终为 1600，18 次触发 purity-diversity pruning，共移除 398 条 pseudo sequence。

## 强攻击边界

还测试了 10-sub `proxy_meta_conflict` individual-upload 攻击。该攻击污染 EEG 输入并刻意保持高 confidence，使错误伪标签在模型内部保持自洽。结果如下：

| 方法 | final ACC | final MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| BrainUICL clean | 0.6676 | 0.6247 | 0.7076 | 0.6837 | 0.0496 |
| BrainUICL attacked | 0.5870 | 0.5473 | 0.5778 | 0.5180 | 0.1644 |
| BrainUICL + PuriDivER attacked | 0.5866 | 0.5286 | 0.5571 | 0.4889 | 0.1649 |

PuriDivER 对该攻击无效。进入 buffer 的伪标签错误率达到约 65% 到 95%，但由于攻击维持高置信、低不确定，loss/entropy GMM 无法可靠区分“自洽但错误”的污染。这个结果说明 PuriDivER 适合 noisy-label contamination，不应被表述成能防住任意 EEG adversarial poisoning。

当前结论基于一个预训练 seed。clean-stream 实验表明 PuriDivER 保留了可用的持续学习能力，完整 49-sub 的 final ACC/MF1 和 FR 优于 BrainUICL，但多数中间检查点和全程平均指标略差；标签污染实验进一步说明防护链路有效。要形成论文级结论仍需更多 checkpoint seeds，以及相同 1600 条 memory budget 的 BrainUICL 对照消融。

代码入口：

```text
experiments/puridiver_eeg.py
experiments/rttdp_brainuicl_full.py --defense-mode puridiver
```

主要结果：

```text
experiments/rttdp_brainuicl_runs/probe10_puridiver_buffer_label_noise60_seed4321/
experiments/rttdp_brainuicl_runs/full49_puridiver_buffer_label_noise60_seed4321/
experiments/rttdp_brainuicl_runs/full49_puridiver_clean_seed4321/
experiments/rttdp_brainuicl_runs/probe10_puridiver_v2_proxy_meta_eps05_cw5_seed4321/
```
