# BrainUICL RTTDP-style Full Run Results

生成日期：2026-07-05

## 运行对象

这次跑的是完整 49 个 new subjects 的 clean CL vs attacked CL，对比条件保持一致：

- 同一个 seed：`4321`
- 同一个 old/train/val/new split
- 同一个 new-individual update 顺序
- 同样 `ssl_epoch=10`、`incremental_epoch=10`
- 同样 batch/worker 设置：`batch=16`、`num_worker=4`
- 攻击模式：`model_nhe`，即 white-box/no-budget 代理目标强攻击

原始 BrainUICL CL 代码没有被这次攻击实验修改；完整流程在独立脚本 `experiments/rttdp_brainuicl_full.py` 中运行。

## 命令

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --gpu 0 \
  --output-root experiments/rttdp_brainuicl_runs/full49_model_nhe_seed4321 \
  --max-subjects 0 \
  --ssl-epoch 10 \
  --incremental-epoch 10 \
  --batch 16 \
  --num-worker 4 \
  --attack-mode model_nhe \
  --attack-lr 8e-5
```

## 输出

```text
experiments/rttdp_brainuicl_runs/full49_model_nhe_seed4321/
  run.log
  split.json
  comparison.json
  comparison_report.md
  clean/metrics.json
  attack_model_nhe/metrics.json
```

## 新个体顺序

```text
64, 89, 1, 27, 60, 5, 52, 42, 80, 26, 91, 22, 61, 85, 17, 36, 98,
33, 55, 86, 54, 84, 49, 3, 2, 10, 87, 15, 95, 57, 70, 11, 47, 65,
66, 96, 83, 9, 46, 31, 4, 19, 99, 63, 81, 25, 97, 20, 13
```

## 最终结果

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.7120 | 0.6825 | 0.6970 | 0.6726 | 0.0136 |
| attack_model_nhe | 0.2191 | 0.0974 | 0.2137 | 0.1033 | 0.6881 |

Plasticity 汇总：

| variant | initial ACC | before ACC | after ACC | initial MF1 | before MF1 | after MF1 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.6464 | 0.6154 | 0.6148 | 0.5568 | 0.5428 | 0.5567 |
| attack_model_nhe | 0.6464 | 0.2181 | 0.2055 | 0.5568 | 0.0997 | 0.0857 |

## 结果说明

clean 分支最终 `ACC=0.7120`、`MF1=0.6825`，说明 BrainUICL 在这个单一 full order 下仍然能维持比较稳定的 old/generalization 性能，FR 只有 `0.0136`。

attack 分支最终 `ACC=0.2191`、`MF1=0.0974`，FR 达到 `0.6881`。这说明在 white-box/no-budget 放宽设置下，代理攻击可以显著放大 continual learning 过程中的灾难遗忘和整体性能退化。

攻击分支后期 buffer 基本保持在 `1030`，新增高置信伪标签为 `0`。这表示模型退化后无法再通过置信度过滤产生可靠 pseudo labels，动态 buffer 和自训练更新链路被破坏。

## 和 RTTDP 的关系

这次 full run 用的是 `model_nhe`，它不是最贴近 RTTDP 原文的 signal-level PGD 输入投毒，而是用户允许 white-box、无 budget 后的强代理模型攻击。它的价值是验证上界风险：如果攻击者能直接把代理目标注入 CL 更新，BrainUICL 的 old/generalization 性能会严重崩塌。

更接近 RTTDP 原始形式的 `pgd_nhe/pgd_ble` 路径已经 smoke 通过，但完整 49-subject PGD 还需要单独跑。
