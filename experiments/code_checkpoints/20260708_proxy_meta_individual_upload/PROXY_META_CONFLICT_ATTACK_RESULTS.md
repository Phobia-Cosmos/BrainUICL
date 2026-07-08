# BrainUICL Proxy-meta Conflict Attack Results

生成日期：2026-07-08

## 威胁模型

本实验使用更严格、但比 confidence/buffer stealth 更宽松的 data-only 设置：

- 不能直接修改目标 BrainUICL 的参数、学习率、loss、label、buffer 逻辑；
- 可以从目标获得当前参数/反馈，在本地维护可迭代 proxy；
- 攻击者只提交污染后的输入数据；
- 本轮不要求污染 sequence 必须通过 buffer confidence 或写入 replay buffer。

## 实现

代码：[experiments/rttdp_brainuicl_full.py](/home/undefined/Desktop/bci/papers/TTAP/BrainUICL/experiments/rttdp_brainuicl_full.py:687)

新增攻击模式：

```text
--attack-mode proxy_meta_conflict
```

核心思想：

1. proxy 使用当前 target/student 参数和 teacher 参数；
2. 在每个 joint update batch 中，取正常 replay half 作为旧知识参考；
3. 计算旧任务 loss 对 proxy 参数的梯度 `g_old`；
4. 对候选污染输入 `x_adv`，用 BrainUICL 原始伪标签更新 loss 计算新样本更新梯度 `g_new(x_adv)`；
5. 优化 `x_adv`，让 `g_new` 与 `g_old` 的余弦/内积变小，同时维持一定 teacher confidence；
6. 目标模型随后照常执行原始 `algorithm.update()`，攻击不直接改目标模型。

这与之前 `model_nhe`、train label bias、buffer label bias 不同：那些会直接改变目标训练过程，只能算 upper-bound 诊断；本实验只改变输入数据。

## 10-subject 对照验证

输出目录：

```text
experiments/rttdp_brainuicl_runs/probe10_proxy_meta_conflict_eps05_cw5/
```

命令核心参数：

```bash
--attack-mode proxy_meta_conflict
--max-subjects 10
--batch 16
--proxy-meta-steps 5
--proxy-meta-eps-scale 0.50
--proxy-meta-param-scope classifier
--proxy-meta-conflict-weight 5.0
--proxy-meta-confidence-weight 0.1
--proxy-meta-grad-norm-weight 0.0
--proxy-meta-raw-weight 0.001
--proxy-meta-l2-weight 0.0005
--pgd-random-start
--no-save-checkpoints
```

## 结果

Final stability：

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.6772 | 0.6396 | 0.7032 | 0.6844 | 0.0359 |
| attack | 0.6487 | 0.6106 | 0.5816 | 0.5314 | 0.0765 |
| delta | -0.0285 | -0.0289 | -0.1216 | -0.1530 | +0.0406 |

Final plasticity：

| variant | initial ACC | before ACC | after ACC | initial MF1 | before MF1 | after MF1 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.5920 | 0.5820 | 0.6047 | 0.5025 | 0.4987 | 0.5401 |
| attack | 0.5920 | 0.4698 | 0.4063 | 0.5025 | 0.3909 | 0.3183 |
| delta | 0.0000 | -0.1122 | -0.1984 | 0.0000 | -0.1078 | -0.2218 |

旧任务稳定曲线：

```text
clean ACC:  [0.7025, 0.7015, 0.7251, 0.6620, 0.7126, 0.7162, 0.6996, 0.7011, 0.7070, 0.7298, 0.6772]
attack ACC: [0.7025, 0.4482, 0.5393, 0.6419, 0.5562, 0.5385, 0.6730, 0.5510, 0.5117, 0.5865, 0.6487]
```

攻击诊断摘要：

| step | subject | clean pass | adv pass | proxy conflict | proxy conf | rel EEG | buffer added |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.698 | 0.628 | 0.0238 | 0.921 | 0.386 | 17 |
| 2 | 89 | 0.580 | 0.794 | 0.0516 | 0.939 | 0.425 | 26 |
| 3 | 1 | 0.698 | 0.633 | -0.0270 | 0.919 | 0.546 | 12 |
| 4 | 27 | 0.795 | 0.648 | 0.0416 | 0.919 | 0.397 | 18 |
| 5 | 60 | 0.773 | 0.680 | 0.1584 | 0.921 | 0.416 | 24 |
| 6 | 5 | 0.810 | 0.595 | -0.1897 | 0.902 | 0.394 | 30 |
| 7 | 52 | 0.727 | 0.643 | 0.1154 | 0.920 | 0.390 | 14 |
| 8 | 42 | 0.385 | 0.749 | 0.0094 | 0.931 | 0.552 | 3 |
| 9 | 80 | 0.512 | 0.807 | 0.2365 | 0.933 | 0.382 | 32 |
| 10 | 26 | 0.846 | 0.581 | -0.1118 | 0.909 | 0.398 | 12 |

## 解释

这次攻击有效，原因与之前失败的 PGD/target-flip 不同：

- 之前 PGD/target-flip 主要优化当前预测错误，BrainUICL 会把它吸收成增强/域适配；
- `proxy_meta_conflict` 优化的是“目标正常更新之后的方向”，即让新数据诱导出的更新梯度与旧知识保持方向冲突；
- 攻击发生在 joint incremental update 的新样本输入部分，而不是训练后 buffer 写入；
- 本次没有启用 `--store-poisoned-buffer`，buffer 中仍保存 clean sequence，buffer added 还少于 clean，因此效果不是靠把污染 sequence 长期写入 replay；
- final plasticity 也显著下降，说明污染输入不只是造成旧知识遗忘，也破坏了新 subject 学习。

注意：`mean_proxy_conflict` 不是每步都为负，但攻击仍有效。这说明在 Adam、BatchNorm、CEA、replay 混合训练下，单一 classifier-scope 梯度余弦只是代理指标；真正有害的是整个 batch 内多轮高幅度污染输入诱导的累计更新偏移。

## 存储

本次加入 `--no-save-checkpoints`，10-sub 对照目录只有约 2.1M。旧 PGD probe 中可再生成的 checkpoint 和 poisoned_sequences 已删除，只保留 metrics/report。

## 下一步

如果要进一步增强攻击，同时保持 data-only，可以尝试：

- 把 `--proxy-meta-param-scope` 从 `classifier` 提升到 `encoder_head`，但会增加二阶梯度开销；
- 对 `proxy_meta_conflict` 加入 final-delta diagnostics，记录最终污染输入上的真实 conflict；
- 做 full 49-sub run，确认 10-sub 结论是否延续；
- 降低 `eps_scale`，找到“效果/扰动幅度”的 Pareto 边界。

## 2026-07-08 更新：individual upload scope

用户指出 BrainUICL 的实际更新单位是“一个新个体”，不是原始 RTTDP 的任意 batch。因此新增：

```text
--proxy-meta-poison-scope individual
```

含义：

1. 每个新 subject 到来时，先把该 subject 的全部 clean sequence 预污染成 uploaded poisoned sequence；
2. CPC/self-supervised adaptation 读取 poisoned upload；
3. joint incremental update 读取 poisoned upload + replay；
4. buffer merge 也按 BrainUICL 原逻辑在 poisoned upload 上做 confidence 筛选；
5. plasticity 评估仍使用 clean subject 数据，避免测试数据也被污染。

代码入口：[experiments/rttdp_brainuicl_full.py](/home/undefined/Desktop/bci/papers/TTAP/BrainUICL/experiments/rttdp_brainuicl_full.py:851)

输出目录：

```text
experiments/rttdp_brainuicl_runs/probe10_proxy_meta_individual_upload_eps05_cw5/
```

命令核心参数：

```bash
--attack-mode proxy_meta_conflict
--proxy-meta-poison-scope individual
--max-subjects 10
--batch 16
--proxy-meta-steps 5
--proxy-meta-eps-scale 0.50
--proxy-meta-param-scope classifier
--proxy-meta-conflict-weight 5.0
--proxy-meta-confidence-weight 0.1
--proxy-meta-grad-norm-weight 0.0
--proxy-meta-raw-weight 0.001
--proxy-meta-l2-weight 0.0005
--pgd-random-start
--no-save-checkpoints
```

Final stability：

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.6772 | 0.6396 | 0.7032 | 0.6844 | 0.0359 |
| attack | 0.6062 | 0.5626 | 0.5710 | 0.5124 | 0.1370 |
| delta | -0.0710 | -0.0770 | -0.1322 | -0.1719 | +0.1011 |

Final plasticity：

| variant | initial ACC | before ACC | after ACC | initial MF1 | before MF1 | after MF1 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.5920 | 0.5820 | 0.6047 | 0.5025 | 0.4987 | 0.5401 |
| attack | 0.5920 | 0.4555 | 0.4152 | 0.5025 | 0.3748 | 0.3160 |
| delta | 0.0000 | -0.1265 | -0.1895 | 0.0000 | -0.1239 | -0.2241 |

Buffer absorption：

| step | subject | uploaded seq | buffer added | absorb rate | clean added |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 43 | 28 | 0.651 | 24 |
| 2 | 89 | 50 | 28 | 0.560 | 38 |
| 3 | 1 | 43 | 5 | 0.116 | 27 |
| 4 | 27 | 44 | 27 | 0.614 | 38 |
| 5 | 60 | 44 | 31 | 0.705 | 34 |
| 6 | 5 | 42 | 8 | 0.190 | 23 |
| 7 | 52 | 44 | 19 | 0.432 | 23 |
| 8 | 42 | 39 | 20 | 0.513 | 29 |
| 9 | 80 | 43 | 27 | 0.628 | 30 |
| 10 | 26 | 52 | 1 | 0.019 | 16 |

Total：

```text
uploaded poisoned sequences: 430
poisoned sequences absorbed into buffer: 194
absorption rate: 45.1%
clean buffer added: 282
```

结论：

- individual-upload 更符合 BrainUICL 的实际在线个体更新流程；
- 上传数据确实会被 buffer 原逻辑吸收，但不是全部吸收；
- confidence threshold 起到了过滤作用，尤其后期模型被攻击后低置信，buffer added 明显下降；
- 即使 buffer 吸收少于 clean，攻击仍明显降低 old stability 和 clean new-subject plasticity；
- 因此攻击效果来自两部分：CPC/joint update 直接在 poisoned upload 上训练，以及一部分 poisoned upload 被 replay 长期保留。

存储处理：

```text
poisoned_uploads 原始数据约 814M，已删除；
保留 metrics/comparison/report 后目录约 2.1M。
```
