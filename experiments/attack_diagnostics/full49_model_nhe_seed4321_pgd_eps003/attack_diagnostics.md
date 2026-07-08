# BrainUICL Attack Diagnostics

生成日期：2026-07-07

## 关键区分

- `model_nhe` 是模型/损失级 white-box 攻击，不直接修改输入 EEG/EOG。full run 后期 buffer 不增长，是因为模型输出退化后无法产生足够高置信伪标签，不是因为某个 poisoned input 被过滤。
- `pgd_nhe/pgd_ble` 才是输入级污染。PGD 扰动被限制在 `eps = batch_std * pgd_eps_scale` 内，并通过多步 sign-gradient 让输出靠近攻击目标。

## 生成文件

```text
embedding_pca.png
signal_stats_pca.png
confidence_filter.png
perturbation_magnitude.png
gradient_cosine.png
gradient_module_norms.png
attack_diagnostics.json
```

## 置信度过滤摘要

| subject | clean pass | pgd pass | clean avg conf | pgd avg conf | rel EOG delta | rel EEG delta |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.000 | 0.000 | 0.782 | 0.752 | 0.02902 | 0.03009 |
| 27 | 0.750 | 0.125 | 0.944 | 0.769 | 0.02974 | 0.02758 |
| 64 | 0.500 | 0.062 | 0.892 | 0.717 | 0.02194 | 0.02267 |
| 89 | 0.688 | 0.125 | 0.928 | 0.789 | 0.02424 | 0.02262 |

## 梯度方向摘要

梯度项含义：

- `source_replay_ce`：历史/source replay 样本的监督 CE 梯度。
- `new_pseudo_ce`：new subject 上由 teacher pseudo label 产生的正常 CL 梯度。
- `model_nhe_kl`：当前 full run 使用的 `model_nhe` 攻击目标梯度。
- `pgd_pseudo_ce`：PGD 污染输入进入 pseudo-label 更新后的梯度。

Cosine 矩阵：

| gradient | source_replay_ce | new_pseudo_ce | model_nhe_kl | pgd_pseudo_ce |
|---|---:|---:|---:|---:|
| source_replay_ce | 1.000 | -0.025 | -0.016 | -0.051 |
| new_pseudo_ce | -0.025 | 1.000 | 0.185 | 0.103 |
| model_nhe_kl | -0.016 | 0.185 | 1.000 | 0.342 |
| pgd_pseudo_ce | -0.051 | 0.103 | 0.342 | 1.000 |

## 解释

如果 `pgd pass` 接近 clean pass，说明输入级扰动仍然位于模型可接受的分布附近，攻击更隐蔽；如果明显低于 clean pass，则当前 PGD 过强或目标过偏，容易被置信度过滤发现。

如果 `model_nhe_kl` 与 `new_pseudo_ce/source_replay_ce` 的 cosine 为负或很低，说明攻击更新方向与正常 CL 明显冲突；这可以解释为什么 full `model_nhe` 会导致模型快速退化和 buffer 停止增长。
