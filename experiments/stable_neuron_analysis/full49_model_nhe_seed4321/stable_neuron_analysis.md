# BrainUICL Stable Neuron Analysis

生成日期：2026-07-05

## 方法

这里把 BrainUICL 的神经元近似为 Conv/Linear/Norm 参数的输出通道或输出行。对每个 checkpoint，计算相对 L2 变化：

```text
relative_change = ||w_t - w_0||_2 / (||w_0||_2 + eps)
```

每个参数张量中变化最低的 10% 单元被记为 stable units。这个定义对应论文中的“跨 continual tasks 权重变化小”的稳定性，但不是完整 Fisher 版本；后面用小样本 Fisher 做了旧任务/新任务重要性的交叉检查。

## Clean 分支模块稳定性

| module | units | median final rel-change | p10 | p90 | frac < 0.01 | frac < 0.05 | frac < 0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| feature_extractor | 8576 | 0.005263 | 0.000046 | 2.706246 | 0.660 | 0.785 | 0.802 |
| feature_encoder | 20480 | 0.006611 | 0.000437 | 0.969590 | 0.566 | 0.770 | 0.817 |
| sleep_classifier | 773 | 0.001828 | 0.000824 | 0.009910 | 0.900 | 0.983 | 0.990 |

## Clean vs Attack 模块变化

| module | clean median | attack median | clean p90 | attack p90 |
|---|---:|---:|---:|---:|
| feature_extractor | 0.005263 | 0.999991 | 2.706246 | 21.749058 |
| feature_encoder | 0.006611 | 1.000096 | 0.969590 | 109.282875 |
| sleep_classifier | 0.001828 | 0.913015 | 0.009910 | 1.024225 |

## 与 Replay Buffer / 性能的关系

| variant | corr(global delta, buffer added) | corr(stable delta, buffer added) | corr(global delta, old MF1) | corr(global delta, new MF1 gain) | final buffer length |
|---|---:|---:|---:|---:|---:|
| clean | 0.134 | -0.100 | -0.024 | 0.147 | 2341 |
| attack_model_nhe | NA | NA | 0.323 | -0.099 | 1030 |

## 旧任务/新任务 Fisher 重要性重叠

Fisher 使用 final clean checkpoint，在 old_generalization subjects 和 new_order subjects 上各采样少量 sequence。下面是各层平均结果：

| metric | value |
|---|---:|
| stable_in_old_top_fraction | 0.1573 |
| stable_in_new_top_fraction | 0.1542 |
| old_top_in_stable_fraction | 0.1573 |
| new_top_in_stable_fraction | 0.1542 |
| old_new_top_jaccard | 0.4137 |
| stable_vs_old_new_union_jaccard | 0.1011 |

按参数张量的细节：

| key | stable in old top | stable in new top | old/new top Jaccard | stable vs old/new union Jaccard |
|---|---:|---:|---:|---:|
| feature_encoder:encoder.encoder.sublayer.0.norm.a1 | 0.115 | 0.096 | 0.351 | 0.057 |
| feature_encoder:encoder.encoder.sublayer.0.norm.b1 | 0.038 | 0.096 | 0.333 | 0.057 |
| feature_encoder:encoder.encoder.sublayer.1.norm.a1 | 0.173 | 0.135 | 0.425 | 0.096 |
| feature_encoder:encoder.encoder.sublayer.1.norm.b1 | 0.058 | 0.154 | 0.368 | 0.067 |
| feature_encoder:encoder.feedforward.linear1.bias | 0.083 | 0.068 | 0.595 | 0.057 |
| feature_encoder:encoder.feedforward.linear1.weight | 0.000 | 0.000 | 0.640 | 0.000 |
| feature_encoder:encoder.feedforward.linear2.bias | 0.038 | 0.135 | 0.238 | 0.062 |
| feature_encoder:encoder.feedforward.linear2.weight | 0.173 | 0.173 | 0.169 | 0.110 |
| feature_encoder:encoder.multi_attention.dense.bias | 0.077 | 0.192 | 0.351 | 0.093 |
| feature_encoder:encoder.multi_attention.dense.weight | 0.154 | 0.154 | 0.316 | 0.092 |
| feature_encoder:encoder.multi_attention.w_key.bias | 0.115 | 0.173 | 0.368 | 0.076 |
| feature_encoder:encoder.multi_attention.w_key.weight | 0.154 | 0.192 | 0.387 | 0.104 |
| feature_encoder:encoder.multi_attention.w_query.bias | 0.019 | 0.058 | 0.268 | 0.023 |
| feature_encoder:encoder.multi_attention.w_query.weight | 0.212 | 0.269 | 0.300 | 0.119 |
| feature_encoder:encoder.multi_attention.w_value.bias | 0.096 | 0.115 | 0.368 | 0.049 |
| feature_encoder:encoder.multi_attention.w_value.weight | 0.135 | 0.173 | 0.405 | 0.105 |
| feature_extractor:FEBlock_EEG.0.weight | 0.429 | 0.286 | 0.400 | 0.214 |
| feature_extractor:FEBlock_EEG.1.bias | 0.429 | 0.143 | 0.400 | 0.214 |
| feature_extractor:FEBlock_EEG.1.weight | 0.143 | 0.143 | 0.400 | 0.062 |
| feature_extractor:FEBlock_EEG.11.bias | 0.115 | 0.154 | 0.268 | 0.072 |
| feature_extractor:FEBlock_EEG.11.weight | 0.423 | 0.231 | 0.195 | 0.241 |
| feature_extractor:FEBlock_EEG.12.bias | 0.058 | 0.019 | 0.195 | 0.030 |
| feature_extractor:FEBlock_EEG.12.weight | 0.135 | 0.192 | 0.095 | 0.105 |
| feature_extractor:FEBlock_EEG.5.bias | 0.154 | 0.231 | 0.625 | 0.115 |
| feature_extractor:FEBlock_EEG.5.weight | 0.154 | 0.077 | 0.444 | 0.069 |
| feature_extractor:FEBlock_EEG.6.bias | 0.000 | 0.000 | 0.625 | 0.000 |
| feature_extractor:FEBlock_EEG.6.weight | 0.231 | 0.154 | 0.529 | 0.111 |
| feature_extractor:FEBlock_EEG.8.bias | 0.154 | 0.231 | 0.529 | 0.111 |
| feature_extractor:FEBlock_EEG.8.weight | 0.577 | 0.500 | 0.444 | 0.378 |
| feature_extractor:FEBlock_EEG.9.bias | 0.231 | 0.192 | 0.625 | 0.115 |
| feature_extractor:FEBlock_EEG.9.weight | 0.077 | 0.077 | 0.677 | 0.036 |
| feature_extractor:FEBlock_EOG.0.weight | 0.000 | 0.000 | 0.400 | 0.000 |
| feature_extractor:FEBlock_EOG.1.bias | 0.286 | 0.286 | 0.556 | 0.143 |
| feature_extractor:FEBlock_EOG.1.weight | 0.429 | 0.286 | 0.556 | 0.231 |
| feature_extractor:FEBlock_EOG.11.bias | 0.212 | 0.192 | 0.387 | 0.124 |
| feature_extractor:FEBlock_EOG.11.weight | 0.192 | 0.192 | 0.600 | 0.136 |
| feature_extractor:FEBlock_EOG.12.bias | 0.038 | 0.000 | 0.507 | 0.017 |
| feature_extractor:FEBlock_EOG.12.weight | 0.135 | 0.038 | 0.284 | 0.073 |
| feature_extractor:FEBlock_EOG.5.bias | 0.385 | 0.385 | 0.529 | 0.250 |
| feature_extractor:FEBlock_EOG.5.weight | 0.308 | 0.385 | 0.625 | 0.318 |
| feature_extractor:FEBlock_EOG.6.bias | 0.077 | 0.077 | 0.529 | 0.034 |
| feature_extractor:FEBlock_EOG.6.weight | 0.000 | 0.154 | 0.368 | 0.067 |
| feature_extractor:FEBlock_EOG.8.bias | 0.115 | 0.115 | 0.733 | 0.057 |
| feature_extractor:FEBlock_EOG.8.weight | 0.538 | 0.615 | 0.576 | 0.439 |
| feature_extractor:FEBlock_EOG.9.bias | 0.077 | 0.115 | 0.677 | 0.056 |
| feature_extractor:FEBlock_EOG.9.weight | 0.192 | 0.154 | 0.529 | 0.091 |
| feature_extractor:fusion.bias | 0.115 | 0.135 | 0.333 | 0.083 |
| feature_extractor:fusion.weight | 0.288 | 0.154 | 0.405 | 0.167 |
| sleep_classifier:sleep_stage_classifier.weight | 0.000 | 0.000 | 0.000 | 0.000 |
| sleep_classifier:sleep_stage_mlp.0.bias | 0.000 | 0.000 | 0.238 | 0.000 |
| sleep_classifier:sleep_stage_mlp.0.weight | 0.000 | 0.077 | 0.156 | 0.029 |
| sleep_classifier:sleep_stage_mlp.3.bias | 0.000 | 0.000 | 0.300 | 0.000 |
| sleep_classifier:sleep_stage_mlp.3.weight | 0.000 | 0.000 | 0.300 | 0.000 |

## 解释

- BrainUICL clean 分支确实存在参数变化很小的稳定单元，尤其是在卷积特征提取和 Transformer 表征层中更明显。
- 这些稳定单元更像是跨 subject 共享的睡眠阶段表征，而不是论文中 SplitMNIST/SplitCIFAR 那种绑定到某个离散 class-incremental task 的神经元。
- replay buffer 不直接“固定某些神经元”，但它通过混合历史样本和高置信伪标签约束梯度方向，间接降低全局参数漂移；攻击分支 buffer 不再增长时，模型退化明显。
- 因为 BrainUICL 的每个 new task 仍然是同一套 5 类睡眠分期任务，只是 subject/domain 改变，所以稳定性仍会出现；但原因更偏向共享生理模式和预训练表征，而不是不同任务使用近乎不相交的神经元集合。
