# BrainUICL 复现报告

生成日期：2026-07-03  
代码仓库：`/home/undefined/Desktop/bci/papers/TTAP/BrainUICL`  
论文 PDF：`/home/undefined/Desktop/bci/papers/TTAP/TTA/2025ICLR-BrainUICL: An Unsupervised Individual Continual Learning Framework for EEG Applications.pdf`

数据格式、脚本参数、指标和方法机制的详细解释见 `REPRODUCTION_QA.md`。

## 1. 复现结论

本地已经完成两轮 ISRUC 复现：

1. **10-subject 最小完整流程**：用于快速验证下载、预处理、预训练、continual adaptation、伪标签过滤、buffer 增长、最终分析是否能跑通。这个结果不能当作论文级别指标，只说明工程链路可用。
2. **98-subject 完整流程**：使用 BrainUICL ISRUC 代码实际会用到的 subgroup-I subject，即 `1..100` 中排除 `8,40` 后的 98 人。预训练和 49 个 new individual 的 continual learning 已完整跑完。

完整 run 的总体表现比 10-sub run 稳定得多：98-sub 的 stream 平均泛化指标 `AAA=0.6953`、`AAF1=0.6717`，但最终一步的泛化 `ACC=0.6199`、`MF1=0.5829` 因最后几个 subject 带来下降。这个现象和 BrainUICL 的问题设定一致：不同个体 EEG 分布差异明显，部分新个体会对通用模型造成冲击。

## 2. 环境与存储

Python 环境：

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python
```

已验证：

```text
GPU: NVIDIA GeForce RTX 4070 SUPER
CUDA available: True
torch: 2.9.1+cu130
```

大文件没有放在论文/代码目录，而是放在：

```text
/home/undefined/Disk/ai-storage/BrainUICL/
  downloads/isruc/subgroupI_rar/       # ISRUC subgroup-I rar
  raw/isruc/group1/                    # 解压后的 .rec + 标注 txt
  processed/isruc_group1_npy_float32/  # BrainUICL 训练输入
  model_parameter/                     # checkpoint；repo 内 model_parameter 是软链接
  logs/                                # 训练日志
  envs/brainuicl/                      # Python 环境
```

当前实际占用：

```text
downloads:       6.8G
raw:             8.0K  # extracted raw has been removed after reproduction
processed:       7.7G
model_parameter: 2.6G
logs:           ~1.0M
```

当前 `/home/undefined/Disk/ai-storage` 可用空间约 `48G`。如果后续只需要重新跑训练，不需要重新预处理，可以只保留 `processed/`、`model_parameter/`、`logs/`。`raw/` 已清理；如果确认不再需要离线重建数据，也可以删除 `downloads/` 再节省约 `6.8G`。

## 3. 数据集形态

本次没有下载官方提供的 extracted `.mat` 大包；该路线 100 subject 约 `44.5GB`。实际使用的是 BrainUICL 代码匹配的 `.rar -> .rec + hypnogram -> .npy` 路线，98 个 subject 的 rar 下载总量约 `7.29GB`。

处理后的数据：

```text
/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32/<subject>/data/<seq_id>.npy
/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32/<subject>/label/<seq_id>.npy
```

完整数据统计：

```text
subjects: 98
sequences: 4276
files: 8552 npy files = 4276 data + 4276 label
per-subject sequences: min 36, max 52, mean 43.63
```

单个样本：

```text
data shape:  (20, 8, 3000)
label shape: (20,)
```

含义：

```text
20   = 一个 sequence 含 20 个 sleep epoch
8    = 通道数，前 2 个 EOG，后 6 个 EEG
3000 = 每个 30 秒 epoch 以 100Hz 采样得到 3000 点
```

标签是五分类睡眠分期：

```text
0 W, 1 N1, 2 N2, 3 N3, 4 REM
```

## 4. 模型和训练流程

ISRUC 分支的模型可以理解为三段：

```text
EEG/EOG feature extractor -> Transformer encoder -> sleep-stage classifier
```

具体结构：

1. EEG 和 EOG 分别走 1D-CNN block；ISRUC 中 EEG 为 6 通道，EOG 为 2 通道。
2. 两路特征各自池化到 512 维，然后拼接并线性融合回 512 维。
3. 20 个 epoch 的 sequence 输入 3 层、8-head 的 attention encoder。
4. MLP classifier 输出 5 类睡眠分期。

Continual learning 阶段：

1. 先用 source/pretrain subject 有监督训练基础模型。
2. 对每个 new individual，先用 CPC 自监督方式适配 teacher 模型。
3. teacher 对当前 subject 产生伪标签；置信度阈值为 `0.9`，一个 sequence 至少 `15/20` 个 epoch 高置信才进入 buffer。
4. 用新 subject 的高置信伪标签和已有 buffer 做 joint fine-tuning。
5. 每适配一个 individual 后，在 old/generalization set 上评估稳定性，在当前 individual 上评估适应性。

## 5. 本地代码修复

原始 GitHub 代码不能直接在当前环境完整运行，已做最小必要修复：

```text
main.py
  - 修复 bool 参数解析，--is_pretrain false 不再被 Python bool("false") 误判为 True
  - 自动识别已经准备好的 subject，仍遵守 ISRUC 排除 8、40 的逻辑
  - pretrain 阶段不再调用只适用于 continual 阶段的 analysis()
  - BCI2000 trainer 改为懒加载，避免 ISRUC run 被 BCI2000 代码错误阻塞

dataloader/data_loader.py
  - 数据改为按需加载，避免一次性把 full ISRUC 全载入内存
  - buffer dataset 小样本场景允许 replacement sampling
  - 修复索引边界问题

model/pretrain_net.py / trainer/trainer_bci2000.py
  - 修复语法/缩进错误

model/incremental_algorithm.py
  - 当没有高置信伪标签时返回 0 loss，避免 CrossEntropy 空张量崩溃

trainer/pretrainer.py
  - 修复小 batch 下 dev loss 统计和 classification report label 问题
```

## 6. 两次实验命令

10-subject 预训练：

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python main.py \
  --file_path /home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32 \
  --is_pretrain true \
  --pretrain_epoch 100 \
  --batch 16 \
  --num_worker 0 \
  --gpu 0
```

10-subject continual：

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python main.py \
  --file_path /home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32 \
  --is_pretrain false \
  --ssl_epoch 10 \
  --incremental_epoch 10 \
  --batch 16 \
  --num_worker 0 \
  --gpu 0
```

98-subject 完整预训练：

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python main.py \
  --file_path /home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32 \
  --is_pretrain true \
  --pretrain_epoch 100 \
  --batch 16 \
  --num_worker 4 \
  --gpu 0
```

98-subject 完整 continual：

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python main.py \
  --file_path /home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32 \
  --is_pretrain false \
  --ssl_epoch 10 \
  --incremental_epoch 10 \
  --batch 16 \
  --num_worker 4 \
  --gpu 0
```

## 7. 结果对比

| 指标 | 10-subject 最小完整流程 | 98-subject 完整流程 |
|---|---:|---:|
| 使用 subject 数 | 10 | 98 |
| split: train / val / old / new | 2 / 1 / 2 / 5 | 24 / 6 / 19 / 49 |
| pretrain best epoch | 92 | 18 |
| pretrain best ACC | 0.7043 | 0.6637 |
| pretrain best MF1 | 0.5833 | 0.6254 |
| final generalization ACC | 0.7453 | 0.6199 |
| final generalization MF1 | 0.5350 | 0.5829 |
| final AAA | 0.6521 | 0.6953 |
| final AAF1 | 0.4285 | 0.6717 |
| final FR | 0.0404 | 0.1175 |
| final buffer length | 206 | 2361 |
| incremental initial ACC | 0.4744 | 0.6464 |
| incremental before ACC | 0.3905 | 0.6259 |
| incremental after ACC | 0.4278 | 0.6321 |
| incremental initial MF1 | 0.3884 | 0.5591 |
| incremental before MF1 | 0.2991 | 0.5540 |
| incremental after MF1 | 0.3473 | 0.5711 |

解释：

- `final generalization ACC/MF1` 是最后一个 new individual 适配完成后的 old/generalization set 表现。
- `AAA/AAF1` 是整个 continual stream 上的平均泛化表现，比最后一步指标更能反映整体稳定性。
- `FR` 是 forgetting rate，越低表示遗忘越小。
- `incremental before/after` 是各 new individual 在适配前后的平均表现。

## 8. 10-subject 结果详情

使用 subject：

```text
1,2,3,4,5,6,7,9,10,11
```

split：

```text
Train: 7,9
Val:   4
Old:   1,6
New:   2,3,5,10,11
```

continual new order：

```text
11,5,10,3,2
```

高置信伪标签加入 buffer 的 sequence 数：

```text
43,17,12,14,28
```

核心曲线：

```text
Generalization ACC:  [0.7767, 0.5052, 0.6256, 0.5395, 0.7203, 0.7453]
Generalization MF1:  [0.4967, 0.2856, 0.3690, 0.3759, 0.5089, 0.5350]
Generalization AAA:  [0.7767, 0.6410, 0.6359, 0.6118, 0.6335, 0.6521]
Generalization AAF1: [0.4967, 0.3912, 0.3838, 0.3818, 0.4072, 0.4285]
Generalization FR:   [0.0000, 0.3496, 0.1946, 0.3054, 0.0726, 0.0404]
```

10-sub 说明的问题：

- 工程流程已经跑通，但样本和 subject 太少，split 极不稳定。
- final ACC 看起来高，但 MF1 明显偏低，说明类别不均衡下 ACC 容易乐观。
- continual adaptation 有一定收益：new individual 平均 ACC 从 before `0.3905` 到 after `0.4278`，MF1 从 `0.2991` 到 `0.3473`。
- 这轮更适合作为 smoke test，不适合和论文表格直接比较。

## 9. 98-subject 完整结果详情

split：

```text
Train:
7,16,18,23,24,28,30,34,35,37,38,41,45,48,50,53,69,71,74,78,79,82,93,94

Val:
12,21,29,58,76,77

Old:
6,14,32,39,43,44,51,56,59,62,67,68,72,73,75,88,90,92,100

New:
1,2,3,4,5,9,10,11,13,15,17,19,20,22,25,26,27,31,33,36,42,46,47,49,52,54,55,57,60,61,63,64,65,66,70,80,81,83,84,85,86,87,89,91,95,96,97,98,99
```

实际 continual 顺序：

```text
64,89,1,27,60,5,52,42,80,26,91,22,61,85,17,36,98,33,55,86,54,84,49,3,2,10,87,15,95,57,70,11,47,65,66,96,83,9,46,31,4,19,99,63,81,25,97,20,13
```

高置信伪标签加入 buffer 的 sequence 数：

```text
24,35,24,37,33,31,32,29,29,47,36,13,26,35,23,31,8,39,31,33,33,32,28,25,28,27,29,20,28,36,33,18,26,31,27,23,30,1,29,20,25,25,30,29,28,17,30,14,13
```

统计：

```text
Initial buffer length: 1030
Pseudo-labeled sequences added: 1331
Final buffer length: 2361
```

完整 run 的关键结果：

```text
Pretrain best epoch: 18
Pretrain best ACC:   0.6636882129277567
Pretrain best MF1:   0.6254126511315609

Final generalization ACC:  0.619940119760479
Final generalization MF1:  0.5828740330980121
Final AAA:                 0.6952958083832335
Final AAF1:                0.6717326993649914
Final FR:                  0.11746654164180383

Incremental initial ACC:   0.6463871796739017
Incremental before ACC:    0.6258723152398316
Incremental after ACC:     0.6320863664635606
Incremental initial MF1:   0.5591476675308866
Incremental before MF1:    0.5540002379103041
Incremental after MF1:     0.57105910396596
```

完整 run 说明的问题：

- 98-sub 的平均稳定性明显强于 10-sub：`AAF1` 从 `0.4285` 提升到 `0.6717`。
- final ACC 不是最好的整体代表，因为最后两个 subject `20,13` 后 old/generalization set 明显下降：ACC 从 `0.7189 -> 0.6508 -> 0.6199`。
- 个体适配的平均收益是正的，但不大：ACC `0.6259 -> 0.6321`，MF1 `0.5540 -> 0.5711`。
- 一些 subject 是强 domain shift/outlier，例如 `1,5,22,26,98,4,13` 等，在当前顺序下会显著影响模型稳定性。
- CPC contrastive 阶段单独看时有时会造成 old set 表现下跌，joint training 和 buffer 负责把模型拉回到更稳的状态。

## 10. 日志与 checkpoint

日志：

```text
/home/undefined/Disk/ai-storage/BrainUICL/logs/pretrain_isruc_10sub_seed4321.log
/home/undefined/Disk/ai-storage/BrainUICL/logs/brainuicl_isruc_10sub_seed4321.log
/home/undefined/Disk/ai-storage/BrainUICL/logs/pretrain_isruc_full98_seed4321.log
/home/undefined/Disk/ai-storage/BrainUICL/logs/brainuicl_isruc_full98_seed4321.log
```

checkpoint：

```text
10-sub backup:
/home/undefined/Disk/ai-storage/BrainUICL/model_parameter/ISRUC_10sub_seed4321

98-sub current:
/home/undefined/Disk/ai-storage/BrainUICL/model_parameter/ISRUC
```

更多底层记录，包括每个 epoch loss、classification report、每个 individual 的 initial/before/after 指标，都保存在上述日志中。
