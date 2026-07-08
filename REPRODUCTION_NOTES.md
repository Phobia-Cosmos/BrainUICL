# BrainUICL ISRUC Reproduction Notes

Final organized results for both the 10-subject run and the full 98-subject run are in
`REPRODUCTION_REPORT.md`. This file is the working note from the reproduction process and
contains some interim subset-era storage numbers.

## Storage Layout

Large files are kept outside the paper/code directory:

```text
/home/undefined/Disk/ai-storage/BrainUICL/
  downloads/isruc/subgroupI_rar/       # official ISRUC subgroup-I .rar archives
  raw/isruc/group1/                    # extracted .rec + hypnogram files
  processed/isruc_group1_npy_float32/  # BrainUICL input: <subject>/data/*.npy and label/*.npy
  model_parameter/                     # checkpoints; repo model_parameter is a symlink here
  logs/
  tools/7zip/7zz                       # user-space 7-Zip binary
  envs/brainuicl/                      # Python env that reuses ai-storage CUDA PyTorch
```

The repo has:

```text
model_parameter -> /home/undefined/Disk/ai-storage/BrainUICL/model_parameter
```

## Dataset Size

Measured with official `Content-Length` headers on 2026-07-02:

```text
ISRUC subgroup-I raw .rar, 100 subjects:                7,473,353,967 bytes
ISRUC subgroup-I raw .rar, used by BrainUICL (no 8,40): 7,287,979,603 bytes
ISRUC extracted-channel .mat, 100 subjects:            44,546,719,845 bytes
```

The code path in this repo uses the `.rar -> .rec + *_1.txt -> .npy` route, not the `.mat` route.
With float32 `.npy`, the expected storage for the 98 used subjects is roughly:

```text
downloads:     ~6.8 GiB
extracted raw: ~10-11 GiB
processed npy: ~7-8 GiB
total kept:    ~25-30 GiB plus checkpoints/logs
```

The current `/home/undefined/Disk/ai-storage` partition has about 70 GiB free, so this layout is safe.

## Environment

Use:

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python
```

Verified:

```text
torch 2.9.1+cu130
CUDA available: True
GPU: NVIDIA GeForce RTX 4070 SUPER
numpy 2.5.0
pandas 3.0.3
scikit-learn 1.9.0
mne 1.12.1
```

## Prepare Data

Prepare all subjects used by the paper code:

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  scripts/prepare_isruc_group1.py \
  --download-root /home/undefined/Disk/ai-storage/BrainUICL/downloads/isruc/subgroupI_rar \
  --raw-root /home/undefined/Disk/ai-storage/BrainUICL/raw/isruc/group1 \
  --output-root /home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32 \
  --tmp-root /home/undefined/Disk/ai-storage/BrainUICL/tmp
```

Prepare the minimal complete-workflow subset used in the local run:

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  scripts/prepare_isruc_group1.py \
  --subjects 1,2,3,4,5,6,7,9,10,11 \
  --download-root /home/undefined/Disk/ai-storage/BrainUICL/downloads/isruc/subgroupI_rar \
  --raw-root /home/undefined/Disk/ai-storage/BrainUICL/raw/isruc/group1 \
  --output-root /home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32 \
  --tmp-root /home/undefined/Disk/ai-storage/BrainUICL/tmp
```

The script is resumable: existing archives, extracted raw files, and processed `.npy` outputs are skipped by default.

Subject 8 is skipped because the original BrainUICL ISRUC code excludes subjects 8 and 40.

## Smoke Test Commands

Pretrain for one epoch on the 5-subject subset:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python main.py \
  --file_path /home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32 \
  --is_pretrain true \
  --pretrain_epoch 1 \
  --batch 4 \
  --num_worker 0 \
  --gpu 0
```

Run one incremental smoke-test epoch:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python main.py \
  --file_path /home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32 \
  --is_pretrain false \
  --ssl_epoch 1 \
  --incremental_epoch 1 \
  --batch 4 \
  --num_worker 0 \
  --gpu 0
```

Both smoke tests completed successfully on 2026-07-02.

## 10-Subject Minimal Complete Run

The smallest run that exercises pretraining, old-task evaluation, continual SSL adaptation, joint training,
pseudo-label filtering, buffer growth, and final analysis used these ISRUC subjects:

```text
1,2,3,4,5,6,7,9,10,11
```

Current storage for this subset:

```text
downloads:     937M  /home/undefined/Disk/ai-storage/BrainUICL/downloads/isruc/subgroupI_rar
extracted raw: 1.5G  /home/undefined/Disk/ai-storage/BrainUICL/raw/isruc/group1
processed npy: 825M  /home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32
checkpoints:   284M  /home/undefined/Disk/ai-storage/BrainUICL/model_parameter
logs:          328K  /home/undefined/Disk/ai-storage/BrainUICL/logs
```

Processed data layout and shape:

```text
<output-root>/<subject>/data/<sequence_id>.npy
<output-root>/<subject>/label/<sequence_id>.npy
```

Each `data/*.npy` has shape `(20, 8, 3000)`:

```text
20   = sequence length
8    = channels: first 2 EOG, last 6 EEG
3000 = 30 seconds sampled at 100 Hz
```

Each `label/*.npy` has shape `(20,)`, with labels mapped to five sleep stages:

```text
0 W, 1 N1, 2 N2, 3 N3, 4 REM
```

Current 10-subject subset statistics:

```text
subject, sequences, label_counts(W,N1,N2,N3,REM)
1, 43, [264, 73, 174, 231, 118]
2, 47, [239, 103, 344, 157, 97]
3, 46, [130, 162, 246, 173, 209]
4, 47, [28, 65, 426, 214, 207]
5, 42, [261, 108, 265, 164, 42]
6, 43, [685, 16, 60, 99, 0]
7, 45, [134, 171, 204, 234, 157]
9, 47, [121, 173, 341, 159, 146]
10, 41, [298, 90, 308, 96, 28]
11, 48, [331, 96, 267, 179, 87]
TOTAL, 449, [2491, 1057, 2635, 1706, 1091]
```

Pretraining command:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python main.py \
  --file_path /home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32 \
  --is_pretrain true \
  --pretrain_epoch 100 \
  --batch 16 \
  --num_worker 0 \
  --gpu 0 \
  2>&1 | tee /home/undefined/Disk/ai-storage/BrainUICL/logs/pretrain_isruc_10sub_seed4321.log
```

Pretraining result:

```text
Train subjects:    7,9
Validation subject: 4
Old-task subjects: 1,6
New-task subjects: 2,3,5,10,11
Buffer length:     92
Best epoch:        92
Best validation ACC: 0.7042553191489361
Best validation MF1: 0.5833362976549468
```

Continual adaptation command:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python main.py \
  --file_path /home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32 \
  --is_pretrain false \
  --ssl_epoch 10 \
  --incremental_epoch 10 \
  --batch 16 \
  --num_worker 0 \
  --gpu 0 \
  2>&1 | tee /home/undefined/Disk/ai-storage/BrainUICL/logs/brainuicl_isruc_10sub_seed4321.log
```

Continual adaptation result:

```text
New-task order: 11,5,10,3,2
Final buffer length: 206
Confident pseudo-labeled sequences added per individual: 43,17,12,14,28

Generalization ACC Curve: [0.7767441860465116, 0.5052325581395349, 0.6255813953488372, 0.5395348837209303, 0.7203488372093023, 0.7453488372093023]
Generalization MF1 Curve: [0.4966937319954165, 0.28562936091542307, 0.3689566353657819, 0.3759086465692766, 0.5088735998554144, 0.5349552548732697]
Generalization AAA Curve: [0.7767441860465116, 0.6409883720930233, 0.6358527131782946, 0.6117732558139535, 0.6334883720930232, 0.6521317829457364]
Generalization AAF1 Curve: [0.4966937319954165, 0.3911615464554198, 0.38375990942554045, 0.3817970937114745, 0.4072123949402625, 0.4285028715957637]
Generalization FR Curve: [0.0, 0.34955089820359275, 0.19461077844311384, 0.3053892215568862, 0.0726047904191617, 0.04041916167664669]

Incremental individual Initial ACC:          0.47435617158801435
Incremental individual Before Adaptation ACC: 0.3904844070477179
Incremental individual After Adaptation ACC:  0.42777469175098126
Incremental individual Initial MF1:          0.38835897833343747
Incremental individual Before Adaptation MF1: 0.299091750660634
Incremental individual After Adaptation MF1:  0.34729781678083993
```

Per-new-subject adaptability, stored as `[initial, before adaptation, after joint adaptation]`:

```text
subject 11 ACC [0.44583333333333336, 0.44583333333333336, 0.5510416666666667]
subject 5  ACC [0.5404761904761904, 0.3357142857142857, 0.3107142857142857]
subject 10 ACC [0.46463414634146344, 0.3926829268292683, 0.37926829268292683]
subject 3  ACC [0.47934782608695653, 0.375, 0.4489130434782609]
subject 2  ACC [0.44148936170212766, 0.4031914893617021, 0.44893617021276594]
```

This is a runnable workflow reproduction, not a paper-level numerical reproduction. The subset has only
10 subjects, so the train/validation/old/new splits are much smaller than the paper setting and the curves
are noisier. The useful signal is that the full code path runs on CUDA and shows the intended continual
learning behavior: new individuals are adapted one by one, confident pseudo-labeled samples enter the
buffer, and stability/plasticity metrics are produced.

## Full Run Commands

After all 98 subjects are prepared, run pretraining:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python main.py \
  --file_path /home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32 \
  --is_pretrain true \
  --pretrain_epoch 100 \
  --batch 16 \
  --num_worker 4 \
  --gpu 0 | tee /home/undefined/Disk/ai-storage/BrainUICL/logs/pretrain_isruc_seed4321.log
```

Then run BrainUICL continual adaptation:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python main.py \
  --file_path /home/undefined/Disk/ai-storage/BrainUICL/processed/isruc_group1_npy_float32 \
  --is_pretrain false \
  --ssl_epoch 10 \
  --incremental_epoch 10 \
  --batch 16 \
  --num_worker 4 \
  --gpu 0 | tee /home/undefined/Disk/ai-storage/BrainUICL/logs/brainuicl_isruc_seed4321.log
```
