# BrainUICL 代理退化验证记录

生成日期：2026-07-05

## 目标

验证一个问题：借鉴 `RTTDP` 的 surrogate/proxy attack 思想，在 BrainUICL 的 EEG 睡眠 continual learning 流程中，是否可以用一个代理目标让模型在 CL 过程中出现明显的灾难遗忘或性能退化。

约束：

- 不修改原始 CL 代码：`main.py`、`trainer/trainer.py`、`model/incremental_algorithm.py` 未改动。
- 新增独立 probe 脚本：`experiments/proxy_degradation_probe.py`。
- 使用 white-box 设置：允许读取 M0 权重和模型输出。
- 不考虑 attacker budget：只做可行性验证，不做现实攻击约束。
- 不写入原 `model_parameter`，只在仓库 `experiments/proxy_degradation/` 下写结果。

## 从 RTTDP 提取的思想

RTTDP 的核心不是只看被攻击样本本身，而是看被污染输入进入在线适配流程后，是否让模型在后续 benign samples 上退化。迁移到 BrainUICL 后，对应关系是：

| RTTDP | BrainUICL/UICL |
|---|---|
| TTA online stream | new individual continual flow |
| surrogate/white-box model | BrainUICL M0 或当前 incremental model 的代理副本 |
| poisoned/adversarial batch | 被代理目标驱动的 new subject 更新 |
| other users' benign samples error | old/generalization set ACC/MF1 下降 |

本次先不做 EEG 原始信号 PGD 扰动，而是做更直接的 white-box proxy update：使用 NHE-style 目标把模型推离当前预测类别，看是否会破坏 old/generalization set。

## 实验脚本

新增脚本：

```text
experiments/proxy_degradation_probe.py
```

它做三件事：

1. 加载完整复现得到的 ISRUC pretrained M0 checkpoint。
2. 对候选 new subjects 做短步数 proxy update，估计谁最容易导致 old/generalization set 下降。
3. 选出最 harmful 的 subject 顺序，与 benign pseudo-label update 做对比。

脚本不调用原 `trainer.trainer()`，因此不会覆盖原 BrainUICL checkpoint，也不会改变 full run 结果。

## 代理目标

使用 NHE-style objective。对每个 epoch：

1. 先由当前模型得到 softmax prediction。
2. 找到当前预测类别 `argmax(prob)`。
3. 构造一个 target distribution：当前预测类概率为 0，其余 4 类均分概率。
4. 最小化模型输出到这个 target distribution 的 KL loss。

直观上，这会让模型“不再相信当前预测类别”，把特征和分类边界推向混乱状态。这个目标不使用真实标签，符合“代理模型只利用模型输出”的验证思路。

为了更接近 CL replay，脚本还加入少量 source replay：

```text
source_weight = 0.2
target_weight = 1.0
```

即每一步同时保留一部分 source true-label CE loss，但主要受 proxy objective 影响。

## 运行命令

GPU 当时被其他进程占满，因此本次用 CPU 跑小规模验证：

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/proxy_degradation_probe.py \
  --gpu -1 \
  --candidate-count 6 \
  --sequential-k 2 \
  --update-batches 4 \
  --batch 4 \
  --eval-max-batches 0 \
  --attack-lr 8e-5 \
  --benign-lr 8e-5 \
  --source-weight 0.2 \
  --target-weight 1.0
```

结果文件：

```text
experiments/proxy_degradation/proxy_degradation_report.md
experiments/proxy_degradation/proxy_degradation_results.json
```

## 结果

Baseline M0 在 old/generalization set 上：

```text
ACC = 0.7024
MF1 = 0.6880
```

候选 subject 为 natural new order 前 6 个：

```text
64, 89, 1, 27, 60, 5
```

proxy 评分选出的最 harmful 顺序：

```text
64, 5
```

完整 old/generalization set 曲线：

| 设置 | step 0 ACC/MF1 | step 1 ACC/MF1 | step 2 ACC/MF1 | 最终下降 |
|---|---:|---:|---:|---:|
| benign natural: 64 -> 89 | 0.7024 / 0.6880 | 0.6815 / 0.6694 | 0.6912 / 0.6697 | ACC -0.0112, MF1 -0.0183 |
| benign selected: 64 -> 5 | 0.7024 / 0.6880 | 0.6916 / 0.6575 | 0.6468 / 0.5941 | ACC -0.0556, MF1 -0.0939 |
| proxy attack selected: 64 -> 5 | 0.7024 / 0.6880 | 0.5443 / 0.4914 | 0.3786 / 0.2739 | ACC -0.3238, MF1 -0.4141 |

结论：

- 单纯选择较 harmful 的 subject 顺序已经能比 natural order 带来更多退化。
- 使用 proxy/NHE-style update 后，old/generalization MF1 从 `0.6880` 降到 `0.2739`，下降 `0.4141`。
- 同样 subject 顺序下，benign update 只让 MF1 下降 `0.0939`。
- 因此，在 white-box、无 budget 限制的设定下，代理目标确实可以让 BrainUICL 风格的 CL 流程出现明显灾难遗忘/性能退化。

## 注意事项

这只是可行性验证，不是严格论文级攻击实验：

- 当前 probe 没有完整复刻 BrainUICL 的 CPC、DCB、CEA 全流程。
- 当前 probe 没有对 EEG raw signal 做 PGD 级别扰动，只验证了 proxy objective 影响更新方向。
- 当前只跑了候选前 6 个 subject 和 2-step 序列。
- GPU 被占用，所以用 CPU 小规模验证。

但这个结果足够说明：这个方向值得继续做。下一步应当把 proxy objective 接入一个“复制版 trainer”中，完整保留 CPC/DCB/CEA，但输出到单独 checkpoint 目录，继续保持原 CL 代码不变。

## 下一步建议

1. 做完整 proxy-copy trainer：复制 `trainer.py` 到 `experiments/`，只改输出目录和代理目标。
2. 对比三组：
   - original BrainUICL clean order
   - harmful subject order only
   - harmful order + proxy NHE/BLE objective
3. 做 5 个 random order，报告 mean/std 的 AAA、AAF1、FR。
4. 加入更接近 RTTDP 的 EEG signal-level perturbation，但先限制在 processed `.npy` 的临时副本，避免污染原数据。
5. 测试 BrainUICL 防御组件是否有效：去掉/保留 DCB、CEA，比较 proxy degradation 的幅度。
