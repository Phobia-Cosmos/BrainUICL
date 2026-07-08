# BrainUICL Distribution Trajectory

生成日期：2026-07-07

## 目的

同一批 clean EEG/EOG 输入分别送入 Pretrain、clean CL、attack CL checkpoints，观察表征空间如何变化。这样可以区分：

- 原始输入分布是否不同；
- 特征提取后分布是否开始明显变化；
- clean CL 和 attack CL 对同一输入的表征漂移是否不同。

## Subjects

- source_train: [7, 16, 18, 23]
- old_generalization: [6, 14, 32, 39]
- new_order: [64, 89, 1, 27]

## 输出图像

```text
raw_signal_tsne.png
feature_tsne_by_checkpoint.png
centroid_shift_from_pretrain.png
old_new_distance_by_checkpoint.png
```

## Centroid shift from pretrain

| state/group | L2 distance |
|---|---:|
| attack_10:new_order | 31.7904 |
| attack_10:old_generalization | 29.7296 |
| attack_10:source_train | 62.3398 |
| attack_25:new_order | 24.2216 |
| attack_25:old_generalization | 26.6845 |
| attack_25:source_train | 61.4907 |
| attack_49:new_order | 23.2720 |
| attack_49:old_generalization | 19.3032 |
| attack_49:source_train | 23.5894 |
| clean_10:new_order | 25.7784 |
| clean_10:old_generalization | 26.3125 |
| clean_10:source_train | 58.6294 |
| clean_25:new_order | 25.1508 |
| clean_25:old_generalization | 25.4245 |
| clean_25:source_train | 59.9095 |
| clean_49:new_order | 23.3314 |
| clean_49:old_generalization | 23.3954 |
| clean_49:source_train | 49.1309 |
| pretrain:new_order | 0.0000 |
| pretrain:old_generalization | 0.0000 |
| pretrain:source_train | 0.0000 |

## Old/New distance

| state | old-new centroid distance |
|---|---:|
| pretrain | 18.3609 |
| clean_10 | 23.9944 |
| clean_25 | 24.9056 |
| clean_49 | 22.8168 |
| attack_10 | 25.1749 |
| attack_25 | 22.5212 |
| attack_49 | 25.6501 |

## 如何解读

- `raw_signal_tsne.png` 只看输入统计，不受 checkpoint 影响；如果 clean/PGD 在这里很近，说明原始数据空间偏移不明显。
- `feature_tsne_by_checkpoint.png` 看模型提取后的 embedding；同一输入在不同 checkpoint 下位置变化，说明变化来自模型表征而不是输入本身。
- `centroid_shift_from_pretrain.png` 量化 CL 过程中表征中心相对 pretrain 的漂移；attack 曲线如果明显更大，说明攻击主要拉偏模型表征。
- `old_new_distance_by_checkpoint.png` 看 old/new subject 表征距离是否被 CL 或 attack 放大。
