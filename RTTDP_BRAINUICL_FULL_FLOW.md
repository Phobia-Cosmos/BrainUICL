# RTTDP-style BrainUICL 完整流程迁移记录

生成日期：2026-07-05

## 目标

在不修改 BrainUICL 原始 CL 代码的前提下，把 RTTDP 的攻击流程迁移到 ISRUC 睡眠 EEG continual learning 设置中，并比较：

```text
同一 new-individual 顺序下：
clean CL vs attacked CL
```

所有输出都保存到单独目录，不覆盖原复现 checkpoint。

## 新增脚本

```text
experiments/rttdp_brainuicl_full.py
```

这个脚本是独立 runner，复制了 BrainUICL continual 阶段必要逻辑：

- 读取已有 full pretrain M0 checkpoint。
- 使用和原始 `main.py` 一致的 subject split 和 new order。
- clean 和 attack 使用完全相同的 new-individual order。
- 保留 CPC teacher adaptation、pseudo-label filtering、dynamic buffer、CEA 逻辑。
- checkpoint、pseudo labels、metrics 都写入 `experiments/rttdp_brainuicl_runs/<run_name>/`。

原始 CL 文件未改：

```text
main.py
trainer/trainer.py
model/incremental_algorithm.py
```

## 已迁移的攻击模式

脚本当前支持：

```text
model_nhe
model_ble
pgd_nhe
pgd_ble
```

含义：

- `model_nhe`：white-box 代理目标直接参与当前 subject 的更新，把模型输出推向“非当前预测类别”的均匀分布。这个是无 budget 放宽设置下的强攻击，用于验证上界风险。
- `model_ble`：white-box 代理目标使用 BLE-style class bias mapping，把当前预测类推向代理估计出的混淆类。
- `pgd_nhe`：RTTDP-style signal-level poisoning。对当前 new subject 的 EEG/EOG 输入做 PGD，使模型输出靠近 NHE target，再进入 CPC/joint training。
- `pgd_ble`：RTTDP-style signal-level poisoning。对当前 new subject 输入做 PGD，使模型输出靠近 BLE target。

说明：`pgd_*` 更接近 RTTDP 的“poisoned samples enter online adaptation stream”；`model_*` 是在用户允许 white-box、无 budget 约束后做的强代理更新验证。

## 输出目录结构

以 `smoke_model_nhe_2sub` 为例：

```text
experiments/rttdp_brainuicl_runs/smoke_model_nhe_2sub/
  split.json
  comparison.json
  comparison_report.md
  clean/
    checkpoints/
    pseudo_labels/
    metrics.json
  attack_model_nhe/
    checkpoints/
    pseudo_labels/
    metrics.json
```

## 当前 smoke 结果

由于 GPU 当前被其他 Python 进程占用，本次先用 CPU 跑 2-subject smoke：

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --gpu -1 \
  --output-root experiments/rttdp_brainuicl_runs/smoke_model_nhe_2sub \
  --max-subjects 2 \
  --ssl-epoch 1 \
  --incremental-epoch 1 \
  --batch 4 \
  --num-worker 0 \
  --attack-mode model_nhe \
  --attack-lr 8e-5
```

new-individual 顺序完全相同：

```text
64 -> 89
```

结果：

| variant | final ACC | final MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.7156 | 0.7039 | 0.7108 | 0.6972 | 0.0188 |
| attack_model_nhe | 0.3198 | 0.2789 | 0.4733 | 0.4310 | 0.5447 |

Stability curves：

```text
clean ACC:  [0.7024, 0.7144, 0.7156]
attack ACC: [0.7024, 0.3975, 0.3198]

clean MF1:  [0.6880, 0.6998, 0.7039]
attack MF1: [0.6880, 0.3260, 0.2789]
```

结论：在同一 new order、同样 1+1 epoch 的完整流程 smoke 中，`model_nhe` attack 使 old/generalization MF1 从 `0.7039` 降到 `0.2789`，FR 从 `0.0188` 升到 `0.5447`。这说明攻击目标能显著放大 CL 过程中的遗忘和退化。

## Signal-level attack smoke

也验证了 RTTDP-style signal poisoning 路径可以执行：

### PGD-NHE

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --gpu -1 \
  --output-root experiments/rttdp_brainuicl_runs/smoke_pgd_nhe_1sub \
  --max-subjects 1 \
  --ssl-epoch 1 \
  --incremental-epoch 1 \
  --batch 2 \
  --num-worker 0 \
  --attack-mode pgd_nhe \
  --pgd-steps 1 \
  --pgd-eps-scale 0.10 \
  --run-attack-only
```

结果：

```text
subject 64
old ACC = 0.6766
old MF1 = 0.6489
buffer added = 24
```

### PGD-BLE

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --gpu -1 \
  --output-root experiments/rttdp_brainuicl_runs/smoke_pgd_ble_1sub \
  --max-subjects 1 \
  --ssl-epoch 1 \
  --incremental-epoch 1 \
  --batch 2 \
  --num-worker 0 \
  --attack-mode pgd_ble \
  --pgd-steps 1 \
  --pgd-eps-scale 0.10 \
  --run-attack-only
```

结果：

```text
subject 64
old ACC = 0.6533
old MF1 = 0.6260
buffer added = 21
```

这些 smoke 说明 signal-level RTTDP 路径已经接入完整 CL 流程。当前 PGD 只用 1 step 和较小 eps，所以退化不如 `model_nhe` 强；如果后续要更贴近 RTTDP 的数据投毒结论，应扩大 PGD steps、eps 和 subject 数。

## 已完成 full 49-subject run

完整 49-subject、10+10 epoch clean vs attack 已跑完：

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

输出目录：

```text
experiments/rttdp_brainuicl_runs/full49_model_nhe_seed4321/
  run.log
  split.json
  comparison.json
  comparison_report.md
  clean/
  attack_model_nhe/
```

new-individual 顺序：

```text
64, 89, 1, 27, 60, 5, 52, 42, 80, 26, 91, 22, 61, 85, 17, 36, 98,
33, 55, 86, 54, 84, 49, 3, 2, 10, 87, 15, 95, 57, 70, 11, 47, 65,
66, 96, 83, 9, 46, 31, 4, 19, 99, 63, 81, 25, 97, 20, 13
```

最终 stability 指标：

| variant | ACC | MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|
| clean | 0.7120 | 0.6825 | 0.6970 | 0.6726 | 0.0136 |
| attack_model_nhe | 0.2191 | 0.0974 | 0.2137 | 0.1033 | 0.6881 |

最终 plasticity 汇总：

| variant | initial ACC | before ACC | after ACC | initial MF1 | before MF1 | after MF1 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.6464 | 0.6154 | 0.6148 | 0.5568 | 0.5428 | 0.5567 |
| attack_model_nhe | 0.6464 | 0.2181 | 0.2055 | 0.5568 | 0.0997 | 0.0857 |

结论：在同一新个体顺序、同样 10+10 epoch 设置下，`model_nhe` 白盒代理攻击把最终 old/generalization ACC 从 `0.7120` 降到 `0.2191`，MF1 从 `0.6825` 降到 `0.0974`，FR 从 `0.0136` 升到 `0.6881`。这说明在放宽为 white-box、无 budget 的设置下，代理目标足以让 BrainUICL 的 continual update 出现严重灾难遗忘/性能退化。

攻击分支后期的 buffer 基本停在 `1030`，新增伪标签数量为 `0`。这说明模型输出已经退化到无法通过置信度过滤，后续个体不能再稳定补充高置信样本，CL 的自训练闭环被破坏。

## 后续 PGD full run 命令

更接近 RTTDP signal poisoning 的 full run 尚未完整跑完，命令如下：

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --gpu 0 \
  --output-root experiments/rttdp_brainuicl_runs/full49_pgd_nhe_seed4321 \
  --max-subjects 0 \
  --ssl-epoch 10 \
  --incremental-epoch 10 \
  --batch 8 \
  --num-worker 4 \
  --attack-mode pgd_nhe \
  --pgd-steps 5 \
  --pgd-eps-scale 0.25 \
  --pgd-random-start
```

PGD 版本会比 `model_nhe` 慢很多，因为每个 new subject batch 都要做多步反传来构造 poisoned EEG/EOG 输入。

## 当前结论与限制

- `full49_model_nhe_seed4321` 已完成完整 clean vs attack 对比。
- 当前最强结论来自 `model_nhe`，它是放宽约束下的 white-box/no-budget 强代理攻击，用于验证上界风险。
- `pgd_nhe/pgd_ble` 更接近 RTTDP 原始“输入样本投毒”路径；目前只做了 smoke，完整 PGD 需要额外运行。
- 原始 BrainUICL CL 入口没有被这次 RTTDP 迁移修改；攻击流程保存在独立 `experiments/` runner 中。
