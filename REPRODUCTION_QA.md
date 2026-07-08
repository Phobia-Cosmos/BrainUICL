# BrainUICL 复现问答与方法理解

生成日期：2026-07-03  
仓库：`/home/undefined/Desktop/bci/papers/TTAP/BrainUICL`

本文回答复现过程中容易混淆的概念，重点解释 ISRUC 数据、预处理、训练划分、指标、buffer、CPC 和 BrainUICL 方法假设。最终实验指标见 `REPRODUCTION_REPORT.md`。

## 1. 数据格式

### 原始下载的数据是什么格式？

我们下载的是 ISRUC Sleep subgroup-I 每个 subject 一个 `.rar` 压缩包，例如：

```text
/home/undefined/Disk/ai-storage/BrainUICL/downloads/isruc/subgroupI_rar/1.rar
```

以 subject 1 为例，压缩包里包含：

```text
1/1.rec
1/1_1.txt
1/1_1.xlsx
1/1_2.txt
1/1_2.xlsx
```

`.rar` 压缩的是原始 PSG 记录文件 `.rec` 和专家标注 hypnogram 文件，不是 `.mat`，也不是我们后面生成的 `.npy`。

### 解压后的 raw 是什么？

解压后的 raw 目录是：

```text
/home/undefined/Disk/ai-storage/BrainUICL/raw/isruc/group1/<subject>/
```

每个 subject 主要包含：

- `<sid>.rec`：整晚 PSG 多通道生理信号记录。
- `<sid>_1.txt` / `<sid>_1.xlsx`：第一位睡眠专家的逐 30 秒 sleep stage 标注。
- `<sid>_2.txt` / `<sid>_2.xlsx`：第二位睡眠专家的逐 30 秒 sleep stage 标注。

我们本地脚本和原仓库 preprocessing 逻辑都使用 `_1.txt`，也就是第一位专家标注。

### `.rec` 的作用是什么？

`.rec` 是连续整晚 PSG 信号的主体文件。它保存多通道时间序列，例如 subject 1 读取出来是：

```text
sampling rate: 200Hz
duration:      26400 seconds
channels:      19
```

其中包括 EEG、EOG、EMG、SaO2 等多类 PSG 通道。BrainUICL 只选其中 8 个通道用于 ISRUC 实验：2 个 EOG + 6 个 EEG。

### EDF 是什么？`.rec` 和 EDF 是什么关系？

EDF 是 European Data Format，是常见的生物医学时间序列格式。MNE 提供 `mne.io.read_raw_edf()` 读取 EDF/EDF+。ISRUC 的 `.rec` 文件可以按 EDF 类格式读取；本地脚本把 `.rec` 临时软链接成 `.edf`，只是为了让 MNE 按 EDF reader 顺利解析。

这不是把数据转换了一遍，而是同一个文件换了一个入口名。

### 为什么一个 subject 有多个 txt label 文件？

ISRUC 官方说明 PSG recordings 由两位 human experts 视觉标注。因此同一个 subject 有 `_1` 和 `_2` 两套 hypnogram。二者代表两位专家的 sleep stage scoring。我们的复现只用 `_1.txt`，和原仓库 `preprocessing/isruc_processing.py` 保持一致。

### 处理后的 `.npy` 是什么？

处理后目录是：

```text
/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32/<subject>/data/<seq_id>.npy
/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32/<subject>/label/<seq_id>.npy
```

单个 `data/*.npy`：

```text
shape = (20, 8, 3000)
```

含义：

- `20`：一个 sequence 包含 20 个 sleep epoch。
- `8`：通道数，前 2 个是 EOG，后 6 个是 EEG。
- `3000`：一个 30 秒 epoch 以 100Hz 采样，得到 `30 * 100 = 3000` 个时间点。

单个 `label/*.npy`：

```text
shape = (20,)
```

每个位置对应 sequence 中一个 30 秒 epoch 的 sleep stage：

```text
0 W, 1 N1, 2 N2, 3 N3, 4 REM
```

原始 label 中的 `5` 被合并为 `4`，即把旧 R/REM 编码统一到 REM 类。

### 采样点是什么？

这些点是 EEG/EOG 通道在时间上的信号幅值。直观理解：每个通道每 0.01 秒记录一个电位/生理信号强度值，30 秒就是 3000 个点。它不是图像像素，而是多通道一维时间序列。

MNE 读 EDF 时通常会按 EDF header 的 calibration 把原始整数转换为物理单位。对 EEG/EOG 来说，语义上是电位变化；具体单位以 EDF header/MNE 解释为准。

### 一个 epoch 都是 30 秒吗？

对 ISRUC sleep staging 是的。睡眠分期标准通常以 30 秒为一个 scoring epoch。代码中也是：

```python
duration = 30.0
tmax = 30.0 - 1.0 / raw.info["sfreq"]
```

注意这只针对 ISRUC。论文里 FACED 是 10 秒片段，Physionet-MI 是 4 秒片段。

### 为什么重采样到 100Hz？

原始 ISRUC subject 1 是 200Hz。论文设置中 ISRUC 被 resampled to 100Hz。100Hz 对睡眠分期常用频段足够，Nyquist 频率为 50Hz，可以覆盖论文提到的 0.3-35Hz，同时减少计算和存储。重采样后一个 30 秒 epoch 就固定为 3000 点。

### band-pass filtered 在代码哪里？

本地实际运行的 `scripts/prepare_isruc_group1.py` 没有显式调用 `raw.filter(0.3, 35)`。代码只做了：

- 读取 `.rec`
- 添加 30 秒 annotations
- 切成 epochs
- resample 到 100Hz
- pick 2 EOG + 6 EEG
- 保存 `.npy`

论文说 ISRUC recordings are band-pass filtered `(0.3Hz-35Hz)`。这可能是作者数据准备中已有的处理，或者是论文描述但开源 preprocessing 未显式实现。按“本地代码事实”，我们这次复现没有额外加 band-pass filter。

### `.rar -> .rec + hypnogram -> .npy` 和 `.mat` 路线会不会有数据区别？

理论上，如果 `.mat` 是从同一份 `.rec`、同一套 hypnogram、同样通道、同样滤波和重采样流程导出的，那么二者应表示同一底层记录。

实践上可能有细微差别：

- `.mat` 可能已经做过通道抽取、单位转换、滤波或重采样。
- `.mat` 可能默认保存 float64，`.npy` 本地保存 float32。
- `.mat` 可能包含更多通道或中间字段。
- hypnogram 选 expert 1 还是 expert 2 也会影响 label。

我们这次没有使用 `.mat`，而是按 BrainUICL 仓库的 raw route 复现。

### 为什么 `.mat` 会大很多？

原因通常是存储形式不同：

- `.rar` 里的 `.rec` 是较紧凑的原始记录容器，并且外层有 RAR 压缩。
- `.mat` extracted channels 常常是 MATLAB 数组，可能是 float64、未压缩或压缩效率低，还可能包含更多字段/通道。
- 我们的 `.npy` 是 float32，但不是压缩格式；它只保留 8 个通道和 100Hz 后的数据。

所以 `.mat` 大不代表信息更多，也不代表训练一定更好，主要是封装和数据类型不同。

### 为什么官方还提供 `.mat` 路径？

`.mat` 对 MATLAB 用户和不想处理 EDF/REC 的研究者更方便。它通常是“提取后的通道数据”，避免用户自己解析 `.rec`、选择通道和处理 header。但代价是文件大、灵活性低、可能不完全等同于你自己的 preprocessing。

## 2. 预处理脚本

### `scripts/prepare_isruc_group1.py` 的作用是什么？

这是本地为复现写的 end-to-end ISRUC subgroup-I 准备脚本。它做四件事：

1. 下载 subject 的 `.rar`，如果本地已有则跳过。
2. 用 7-Zip 解压到 raw 目录。
3. 用 MNE 读取 `.rec`，结合 `_1.txt` 标注切 30 秒 epoch。
4. 选 8 个通道、重采样到 100Hz、按 20 个 epoch 一组保存 `.npy`。

它还支持：

- `--subjects` 只处理部分 subject。
- `--force-*` 强制重做某一步。
- `--delete-archive` 处理后删除 `.rar`。
- `--delete-raw` 处理后删除 raw。
- `--keep-final-sequence` 不模仿原作者 `seq_num - 1` 的最后一组丢弃行为。

### `scripts/download_isruc_archives.py` 和 prepare 脚本有什么区别？

`download_isruc_archives.py` 只负责下载 `.rar`，而且是 ranged parallel download：

- 一个文件拆成多个 byte ranges。
- 多线程下载并拼接。
- 更适合大量 subject 的稳定下载。

`prepare_isruc_group1.py` 是完整准备流程，能下载、解压、预处理，但下载方式是逐 subject 的普通 curl。我们完整复现时实际用了两个脚本配合：先用 downloader 批量拿全 `.rar`，再用 prepare 解压和生成 `.npy`。

### `preprocessing/` 目录什么时候被使用？

原仓库的 `preprocessing/isruc_processing.py` 是作者给出的原始预处理脚本示例，但它有硬编码路径，例如 `/data/datasets2/ISRUC_extracted/group1/`，不能直接适配本机目录。

本次复现没有直接运行 `preprocessing/isruc_processing.py`，而是用 `scripts/prepare_isruc_group1.py` 复刻其核心逻辑，并增加：

- 可配置路径。
- 可指定 subject。
- 自动下载/解压。
- 与当前 MNE 版本兼容。
- 可恢复执行。

`preprocessing/edf.py` 是原仓库自带的 EDF reader 副本，主要供原 `isruc_processing.py` 使用；本地新脚本直接使用安装环境中的 `mne.io.read_raw_edf()`。

## 3. 存储安排

### 为什么 downloads 和 raw 可以优先删除？

训练 `main.py` 只读取 processed `.npy`：

```text
/home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32
```

因此：

- `downloads/` 只用于重新解压或重新预处理。
- `raw/` 只用于重新生成 `.npy`。
- `processed/` 才是训练必需输入。
- `model_parameter/` 是训练结果 checkpoint。
- `logs/` 是实验记录。

如果空间紧张，优先删 `raw/`，因为 `.rar` 还可以重新解压；再考虑删 `downloads/`，因为官方可重新下载但耗时。不要删 `processed/`，除非你准备重新预处理。

## 4. 运行命令参数

### `PYTHONUNBUFFERED=1` 的作用是什么？

它让 Python 标准输出不缓冲，日志会即时写到终端或 `tee` 文件。长时间训练时，如果不加这个变量，print 可能延迟很久才落盘，不利于观察进度。

它不影响训练结果，只影响日志刷新。

### `ssl_epoch` 是什么？

`ssl_epoch` 是每个 new individual 到来时，guiding/teacher model 进行 CPC self-supervised fine-tuning 的 epoch 数。论文和我们都用 10。

它不是 pretrain epoch。流程是：

```text
pretrain_epoch:       源域有监督预训练 M0
ssl_epoch:            每个新个体上 CPC 自监督适配 guiding model
incremental_epoch:    每个新个体上 student/incremental model joint fine-tuning
```

### `batch` 的作用是什么？

`batch` 是 DataLoader 每次送入模型的 sequence 数。ISRUC 中一个 item 是一个 `(20, 8, 3000)` sequence，所以 `batch=16` 表示每步处理 16 个 sequence。

batch 影响：

- GPU 显存占用。
- 梯度估计稳定性。
- 训练速度。
- DataLoader 读取压力。

论文超参表里 batch 是 32。本地为了 4070 稳定和少踩显存问题，完整 run 用了 16。

### full command 为什么用 `num_worker=4`？

`num_worker` 是 PyTorch DataLoader 的子进程数，用于并行读取 `.npy`。完整 98-subject 数据文件多，`num_worker=4` 可以减少 GPU 等待数据的时间。

不是必须。`num_worker=0` 更容易 debug，但 full run 会更慢。

### pretrain 为什么不在 GPU 完成？

本地 pretrain 实际是在 GPU 上完成的。代码里 `trainer/pretrainer.py` 使用：

```python
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
model.to(device)
data.to(device)
```

如果观察时 GPU 利用率不高，通常是因为：

- 数据读取或 validation 阶段在 CPU 侧较多。
- 模型不算特别大。
- batch 较小。
- 日志打印和 sklearn metric 计算在 CPU。

预处理阶段则是 CPU/MNE/磁盘 IO 为主，不属于 GPU 训练。

## 5. 训练数据划分

### train / validation / old / new / generalization 是什么关系？

论文设定把 subject 级别数据分成三类：

```text
pretraining/source set : 用于训练初始模型 M0
incremental/new set    : 按个体顺序到来，用无标签方式 continual adaptation
generalization set     : 每轮 adaptation 后测试稳定性
```

本地代码在 `main.py` 中进一步把 pretraining/source set 分成 train 和 validation：

```text
train          : 有标签源域训练，用于 pretrain，也作为 Strue buffer
validation     : 有标签源域验证，用于选择 pretrain 最佳 epoch
old_task_idx   : 代码名叫 old，实际对应论文 generalization/test set
new_task_idx   : 论文 incremental / continual individual flow
```

这些集合按 subject 划分，互不重叠。

### 98-subject 是如何划分的？

BrainUICL ISRUC 使用 `1..100` 排除 `8,40` 后的 98 个 subject。代码比例近似论文 `3:5:2`：

```text
new/incremental: 50% -> 49 subjects
old/generalization: 20% -> 19 subjects
pretraining total: 30% -> 30 subjects
```

pretraining 的 30 个 subject 又被代码分为：

```text
train: 24 subjects
val:    6 subjects
```

因此完整 run split 是：

```text
Train: 24
Val:    6
Old/generalization: 19
New/incremental:    49
```

### pretrain 会把 processed 目录下所有数据一起预训练吗？

不会。

`processed/` 下所有 subject 只是候选集合。`main.py` 先扫描有哪些 subject，然后按 seed 划分。pretrain 阶段只用 `train_idx` 训练，只用 `val_idx` 验证；`old_task_idx` 和 `new_task_idx` 不参与 pretrain。

完整 run 中，虽然 processed 有 98 个 subject，但 pretrain 只训练 24 个 subject。

### 为什么 pretrain 还要选一个最好的 epoch？

pretrain 跑 100 epoch，每个 epoch 后在 validation subject 上评估。代码选择 validation ACC 最高的 epoch，把这个 epoch 的模型保存成 M0。

意义：

- 避免最后一个 epoch 过拟合 train subject。
- 让后续 continual learning 从验证表现最好的 source model 开始。
- 这个 epoch 没有特殊生理意义，只是“在 val split 上表现最好”。

当前代码按 best validation ACC 选，不按 MF1 选。

### train / val / old 之间是什么关系？

它们是不同 subject：

- train：训练 M0，并作为真实标签 buffer `Strue`。
- val：只用于 pretrain 模型选择。
- old/generalization：不参与训练，用于每轮 continual 后测试稳定性。

old 在代码名里叫 old task，但在论文语义上更接近 generalization/test set。

### 论文是否针对不同 continual new order 做实验？

是。论文说明：固定 train/incremental/generalization 的划分，只随机打乱 continual individual flow 的输入顺序，重复 5 次，报告均值和方差/置信区间。

我们本地完整 run 是 seed 4321 下的一次 new order，不是 5 个 order 的统计均值。因此不能把本地单次结果当作论文表格的严格复现。

## 6. subset statistics 和 sequence

### subset statistics 如何解读？

例子：

```text
subject, sequences, label_counts(W,N1,N2,N3,REM)
1, 43, [264, 73, 174, 231, 118]
```

含义：

- subject 1 被切成 43 个 sequence。
- 每个 sequence 有 20 个 30 秒 epoch。
- 总 epoch 数是 `43 * 20 = 860`。
- label_counts 统计这 860 个 epoch 中各睡眠阶段数量。

所以 subject 1：

```text
W:   264 epochs
N1:   73 epochs
N2:  174 epochs
N3:  231 epochs
REM: 118 epochs
```

label_counts 可以看出严重类别不均衡，例如某些 subject 的 REM 很少甚至没有。

### 为什么每个 subject 的 sequence 数不同？

主要原因：

- 每个 subject 的整晚记录时长不同。
- 有的记录有效 epoch 数不同。
- 代码按 20 个 epoch 组成一个 sequence，不足 20 的尾部会丢弃。
- 为模仿作者原 preprocessing，默认还会保存 `seq_num - 1`，也就是再丢掉最后一个完整 sequence。

例如 subject 1 raw 是 26400 秒：

```text
26400 / 30 = 880 epoch
880 / 20 = 44 sequence
作者逻辑保存 44 - 1 = 43 sequence
```

不同个体睡眠时长和有效记录不同，所以 sequence 数不同。

## 7. buffer 和伪标签

### buffer length 代表什么？

`Buffer_Length` 是当前 replay storage 中 sequence 数量，不是 epoch 数，也不是字节数。

完整 run：

```text
Initial buffer length: 1030
Final buffer length:   2361
```

解释：

- 初始 1030 来自 train source subjects 的真实标签 sequence。
- 后续每个 new individual 适配完成后，会把高置信伪标签 sequence 加入 buffer。
- 完整 run 共加入 1331 个伪标签 sequence，所以最终 `1030 + 1331 = 2361`。

一个 sequence 有 20 个 epoch，因此 1030 个 sequence 对应 20600 个 30 秒 epoch 标签。

### 伪标签过滤是哪一步？

代码里有两层过滤：

1. `model/incremental_algorithm.py` 的 `BufferPseudoLabelFinetune.update()`：
   - teacher/guiding model 对当前新 subject 预测。
   - softmax 最大概率 `> 0.9` 的 epoch 才参与 new subject 的 pseudo-label loss。

2. `trainer/trainer.py` 的 `buffer_single_merge()`：
   - 当前 subject 适配完成后，用当前模型重新预测整个 subject。
   - 如果一个 sequence 里至少 15/20 个 epoch 的 max softmax probability `>= 0.9`，就保存该 sequence 的伪标签到 checkpoint 目录，并把该 sequence 加入 future buffer。

第一层用于当前轮训练，第二层用于后续 replay。

### 置信度是如何计算的？

对每个 epoch 的 5 类 logits 做 softmax：

```text
prob = softmax(logits)
confidence = max(prob)
pseudo_label = argmax(prob)
```

如果 `confidence >= 0.9`，认为这个 epoch 的伪标签可信。

### 进入 buffer 代表什么意思？

进入 buffer 不是把数据复制一份到新目录，而是：

- 原始 data path 仍指向 processed 目录里的 `data/<seq_id>.npy`。
- 伪 label 保存到 `model_parameter/.../individual_<num>/label/<seq_id>.npy`。
- 运行时把 data path 和 pseudo label path 追加到 `args.train_path`。

后续新 subject 到来时，DataLoader 会从 buffer 中抽样，与当前新 subject 一起 joint training。

### 高置信度加入的是 sequence id，为什么不需要 subject id？

单看 `label/34.npy` 这种文件名确实没有 subject id。但运行时不是只靠文件名定位数据，而是保存一对路径：

```text
data path  = processed/.../<subject>/data/<seq_id>.npy
label path = model_parameter/.../individual_<num>/label/<seq_id>.npy
```

subject id 已经包含在 data path 里；label path 的 `individual_<num>` 对应该轮 continual order 的第几个新个体。运行期间二者成对存在于 `args.train_path`，所以不需要在 label 文件名里再写 subject id。

缺点是：如果只离线看 `label/34.npy`，确实无法单独知道原 subject，必须结合日志中的 new order 或运行时 path 关系。

### 为什么 Initial buffer length 有 1030 这么多？

因为它不是 subject 数，而是 source train set 的 sequence 数。完整 run 的 train split 有 24 个 subject，每个 subject 大约 36-52 个 sequence，所以总和是 1030。

## 8. 指标解释

### ACC 是什么？

ACC 是 Accuracy，所有 epoch 中预测正确的比例：

```text
ACC = correct predictions / total predictions
```

在睡眠分期里，它统计 W/N1/N2/N3/REM 五类整体正确率。

### MF1 是什么？

MF1 是 Macro-F1。先分别计算每个类别的 F1，再对 5 类做不加权平均：

```text
MF1 = mean(F1_W, F1_N1, F1_N2, F1_N3, F1_REM)
```

它比 ACC 更能反映类别不均衡下的真实性能。睡眠数据中 N2/W 往往多，N1/REM 可能少；如果模型只预测大类，ACC 可能还可以，但 MF1 会很差。

### 我们该如何解释 ACC 和 MF1？

建议：

- ACC 解释为总体正确率。
- MF1 解释为对五个睡眠阶段“平均公平”的分类能力。
- 如果 ACC 高但 MF1 低，说明模型偏向多数类，小类识别差。
- 对 ISRUC sleep staging，更应该同时报告二者，不能只看 ACC。

### AAA 和 AAF1 是什么？

AAA 是 Average Anytime Accuracy。每适配一个 new individual 后，都会在 generalization set 上测一次 ACC；AAA 是到当前时刻为止这些 ACC 的平均。

AAF1 同理，是每轮 generalization MF1 的累计平均。

代码中：

```python
AAA = mean(ACC curve so far)
AAF1 = mean(MF1 curve so far)
```

它们衡量 continual stream 整个过程的稳定性，而不是只看最后一步。

### 和论文结果差别大不大？

论文 Table 2/3 中 ISRUC BrainUICL 大约是：

```text
Final/mean stability: AAA 74.0-74.1%, AAF1 72.0-72.1%
Plasticity after:     ACC 75.1%, MF1 70.0%
```

我们完整 98-sub 单次 run 是：

```text
AAA  69.53%
AAF1 67.17%
After ACC 63.21%
After MF1 57.11%
```

差距是存在的，尤其 plasticity after 指标低很多。主要原因可能包括：

- 我们只跑了一个 continual order；论文是 5 个随机 order 统计。
- 本地 batch 用 16，论文超参表是 32。
- 本地 preprocessing 没有显式 band-pass filter。
- 开源代码存在多处语法/运行问题，本地做了最小修复，不保证和作者私有实验代码完全一致。
- 最后几个 subject 在本地 order 中造成明显下降，single-order final 指标会受顺序影响。

因此本地结果可以说明完整流程跑通和方法趋势，但不是论文表格的严格数值复现。

## 9. checkpoint

### 模型 checkpoint 是什么？

checkpoint 是训练后保存的模型参数。BrainUICL ISRUC 模型分三块保存：

```text
feature_extractor_parameter_4321.pkl
feature_encoder_parameter_4321.pkl
sleep_classifier_parameter_4321.pkl
```

pretrain checkpoint 是初始模型 M0：

```text
model_parameter/ISRUC/Pretrain/
```

continual checkpoint 是每个 new individual 适配后的模型 Mi：

```text
model_parameter/ISRUC/EpochNum_2_cpc/individual_<i>/
```

这里的 `.pkl` 实际是 PyTorch `state_dict`，不是完整 Python 对象，也不包含 optimizer 状态。因此它能用于加载模型权重，但不是严格意义上的完整训练恢复包。

## 10. 模型结构

### 为什么两路特征先池化到 512，再拼接回去？

ISRUC 输入有两种模态：

- EEG：脑电活动，6 通道。
- EOG：眼动信号，2 通道。

代码先用两个独立 Conv1D block 分别提取 EEG 和 EOG 特征。每路经过 `AdaptiveAvgPool1d(1)`，把每个 30 秒 epoch 的时间维压缩成一个 512 维向量。

然后：

```text
EEG feature: 512
EOG feature: 512
concat:     1024
linear:     1024 -> 512
```

这样做的好处：

- EEG/EOG 前期用各自 CNN，保留模态差异。
- 后期融合成统一 512 维，方便 Transformer 在 20 个 epoch 的时间序列上建模。
- linear fusion 可以学习两种模态的加权组合，而不是简单平均。

### 线性融合逻辑是什么？

`self.fusion = nn.Linear(1024, 512)`。

它对拼接后的 `[EEG_feature, EOG_feature]` 做可学习线性变换：

```text
fused = W * concat(EEG, EOG) + b
```

模型可以通过训练学习哪些 EEG/EOG 维度更重要。它不是手工指定权重。

## 11. CPC、guiding model 和自监督适配

### CPC 是什么？

CPC 是 Contrastive Predictive Coding。核心思想是：用前面时间步的 latent representation 预测未来时间步的 representation。

在 BrainUICL 中，一个 sequence 有 20 个 epoch。模型先提取每个 epoch 的 latent feature。CPC 随机选一个时间点 `t`，用 `t` 之前的上下文生成 context vector，再预测未来 `t+1, t+2, t+3` 的 feature。真实未来 feature 是正样本，batch 中其他样本提供负样本，通过 contrastive loss 训练。

它不需要人工标签，所以适合 new individual 无标签到来的场景。

### 为什么不用聚类伪标签？

论文给出的理由是 EEG 信号 SNR 低，cluster-based pseudo-label 对 EEG 不够有效。更具体地说：

- EEG 噪声、伪迹和个体差异都很强。
- 同一睡眠阶段在不同 subject 上分布可能不重合。
- 不同 sleep stage 的特征分布也可能重叠。
- 聚类假设“类别天然形成清晰簇”，但 EEG latent space 不一定满足。

因此论文选择利用 EEG 的时间序列结构，用 CPC 做自监督适配。

### 自监督 fine-tuning 方法有哪些？

BrainUICL 代码只实现 CPC。但 EEG/time-series 中常见自监督 FT 思路包括：

- CPC / predictive coding：预测未来 latent。
- TS-TCC 类 temporal contrastive learning：时间和上下文一致性对比。
- SimCLR/MoCo 风格 augmentation contrastive：不同增强视图拉近。
- BYOL/Barlow Twins/VICReg：非负样本或冗余约束的表征学习。
- masked signal modeling：遮蔽片段后重建。
- autoencoder/denoising autoencoder：重建原信号或去噪。
- forecasting：预测未来原始信号或特征。
- temporal order prediction：判断片段顺序。
- frequency-domain pretext：频带遮蔽、频谱重建或频带一致性。

本论文选 CPC，是因为它直接利用 sequential nature of EEG。

### 为什么要适配 teacher/guiding model？

new individual 没有标签，直接训练 student 很容易被错误伪标签带偏。BrainUICL 做法是：

1. 从最新 incremental model `Mi-1` 复制一个 guiding model `Mg`。
2. 用当前 subject 的无标签数据对 `Mg` 做 CPC 自监督适配。
3. 用适配后的 `Mg` 给当前 subject 生成更贴近该 subject 分布的伪标签。
4. student/incremental model 再结合 buffer 和伪标签训练。

这样可以把“为当前 subject 生成标签”的模型和“长期累积的 incremental model”分开，降低直接漂移风险。

### 论文是否假设 guiding model 可以 fit 新个体？这个假设可靠吗？

是，论文明确表达了这个假设：CPC fine-tuned guiding model 可以 initially fit 当前 incremental individual，从而产生更高质量的 pseudo labels。

这个假设不总是成立。对 outlier subject、强噪声 subject、睡眠结构异常 subject，CPC 适配后仍可能给出错误高置信预测。BrainUICL 用两种方式缓解：

- 置信度阈值 `0.9` 过滤低质量伪标签。
- DCB 用更多真实标签 source samples replay，减少伪标签噪声积累。
- CEA 每两轮对齐历史状态，减少对单个 subject 过拟合。

但这些是缓解，不是理论保证。

### guiding model 优化后生成的 label 是针对 subject 还是 sequence？

模型面对的是一个 current subject 的所有 sequence，但预测粒度是 epoch。

具体是：

- 对当前 subject 的每个 sequence 预测 20 个 epoch 的标签。
- 每个 epoch 是五分类 sleep stage。
- 如果一个 sequence 足够高置信，就把长度 20 的 pseudo-label sequence 存下来。

所以 label 不是 subject-level 标签，而是 epoch-level 标签，按 sequence 文件保存。

## 12. Dynamic Confident Buffer 和 CEA

### 8:2 选择 Strue 和 Spseudo 合理吗？

论文做了 ratio ablation，结论是大多数情况下 `Strue:Spseudo = 8:2` 更好地平衡 stability 和 plasticity。

直观解释：

- `Strue` 是 pretrain source train set 的真实标签，可靠，能抑制伪标签噪声。
- `Spseudo` 是过往 incremental individuals 的高置信伪标签，有噪声但提供新个体多样性。
- 只用 `Strue` 会更保守，可能缺少新个体分布多样性。
- 只用 `Spseudo` 噪声积累风险大。

本地代码在 `BuildBufferDataset` 中按 `args.train_len` 区分初始真实 source buffer 和后续追加的 pseudo buffer，目标就是近似 8:2 抽样。

### 增量模型在不同 epoch 之间会发生很大偏移吗？

可能会，尤其在新 subject 和历史分布差异很大时。每个 fine-tuning epoch 都会更新 feature extractor / encoder / classifier，latent feature distribution 会变化。

BrainUICL 的 CEA 就是针对这个问题：每两 epoch 对齐一次当前模型与之前 epoch 的 buffer hidden features，避免模型为了当前 subject 偏离太远。

### 不同 epoch 学到的特征是不一样的吗？

是。同一个模型在 fine-tuning 过程中参数不断更新，所以同一批 buffer samples 在不同 epoch 经过网络得到的 hidden features 会不同。CEA 用 KL divergence 约束这些 feature distribution 的变化，避免 abrupt representation change。

### CEA 的 every two epochs 是什么含义？

论文和代码都设置 alignment interval = 2。含义是每隔 2 个 fine-tuning epoch，用之前保存的 buffer feature 和当前 feature 做 KL 对齐。

论文 appendix 中也做了 interval ablation，结论是每 2 epoch 通常更好地平衡 stability 和 plasticity。

## 13. 实验代码如何验证

建议按四层验证：

### 1. 数据验证

检查：

```bash
find processed/isruc_group1_npy_float32 -mindepth 1 -maxdepth 1 -type d | wc -l
find processed/isruc_group1_npy_float32 -mindepth 3 -maxdepth 3 -name '*.npy' | wc -l
```

还要抽样检查：

- data shape 是否是 `(20, 8, 3000)`。
- label shape 是否是 `(20,)`。
- label 是否只包含 `0..4`。
- 每个 subject 的 data/label 文件数是否一致。

### 2. 代码可运行性验证

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python -m compileall -q main.py dataloader model trainer utils scripts
```

然后跑 1 epoch smoke test：

- pretrain 1 epoch。
- continual `ssl_epoch=1 incremental_epoch=1`。

目标不是指标好，而是确认完整路径不会崩。

### 3. split 与泄漏验证

查看日志开头的：

```text
Train Idx
Val Idx
Old Task Idx
New Task Idx
```

确认四组 subject 无交集。pretrain 只能用 Train；Val 只选 best epoch；Old/generalization 只评估；New/incremental 只在 continual 阶段按无标签流程适配。

### 4. 结果验证

检查：

- pretrain 是否保存 `model_parameter/ISRUC/Pretrain/*.pkl`。
- continual 是否生成 49 个 `individual_<i>` checkpoint。
- `Buffer_Length` 是否随高置信 pseudo sequence 增长。
- `Generalization ACC/MF1/AAA/AAF1/FR` 是否在日志末尾完整输出。
- 固定 seed 重跑时 split 和 new order 是否一致。

更严格的论文复现需要跑 5 个不同 continual new order，并报告均值/方差。

## 14. 本地完整 run 的关键解释

### 为什么 10-sub run 不能作为论文级结果？

10-sub run 只有：

```text
train 2, val 1, old 2, new 5
```

subject 太少，split 波动极大，类别分布也更偏。因此它只能证明：

- 数据准备没问题。
- 训练/continual/伪标签/buffer/分析流程能跑通。

不能拿来和论文 Table 2/3 比数值。

### 为什么 full run final ACC 会下降，但 AAA 还可以？

final ACC 只看最后一个 model state。我们的 new order 最后两个 subject `20,13` 后，generalization ACC 从 `0.7189 -> 0.6508 -> 0.6199` 下降明显。

AAA/AAF1 是整个 stream 平均，反映全过程稳定性，所以比 final step 更稳健。对 continual learning，AAA/AAF1 通常比最后一步指标更合理。

## 15. 参考来源

- 本地论文 PDF：`/home/undefined/Desktop/bci/papers/TTAP/TTA/2025ICLR-BrainUICL: An Unsupervised Individual Continual Learning Framework for EEG Applications.pdf`
- BrainUICL README：`/home/undefined/Desktop/bci/papers/TTAP/BrainUICL/README.md`
- ISRUC 官方数据页：https://sleeptight.isr.uc.pt/?page_id=48
- ISRUC 官方说明页：https://sleeptight.isr.uc.pt/
- ISRUC subject details 页面：https://sleeptight.isr.uc.pt/?page_id=57
