# LeRobot 数据处理与使用指南

本文档说明如何在 TRaDA 项目中处理和使用 LeRobot 格式的机器人数据。

---

## 1. 数据格式概述

LeRobot 支持两种数据格式：

| 格式 | 说明 | 使用场景 |
|------|------|----------|
| **v2.1** | 每个episode一个parquet文件和视频文件 | 数据合并、混合操作（操作简单） |
| **v3.0** | 所有episode合并为一个parquet文件和视频文件 | 训练（lerobot-train要求） |

**推荐流程**：在 v2.1 格式下进行数据处理（合并、混合），然后转换为 v3.0 格式进行训练。

---

## 2. 数据目录结构

### 2.1 原始分散数据（scattered）
```
~/projects/TRaDA-data-real/scattered/
├── Agilex_Cobot_Magic_insert_wedge_fwd_1897_000001/  # 单个episode目录
│   ├── data/chunk-000/episode_000000.parquet
│   ├── videos/chunk-000/
│   │   ├── observation.images.image_top/episode_000000.mp4
│   │   ├── observation.images.image_left/episode_000000.mp4
│   │   └── observation.images.image_right/episode_000000.mp4
│   └── meta/info.json
├── Agilex_Cobot_Magic_insert_wedge_fwd_1897_000002/
└── ...
```

### 2.2 合并后数据（merged_v21）
```
~/projects/TRaDA-data-real/merged_v21/
└── Agilex_Cobot_Magic_insert_wedge_fwd_1897/
    ├── data/chunk-000/
    │   ├── episode_000000.parquet  # episode 0
    │   ├── episode_000001.parquet  # episode 1
    │   └── ...
    ├── videos/chunk-000/
    │   ├── observation.images.image_top/
    │   │   ├── episode_000000.mp4
    │   │   └── ...
    │   └── ...
    └── meta/
        ├── info.json
        ├── episodes.jsonl
        ├── tasks.jsonl
        └── episodes_stats.jsonl
```

### 2.3 混合数据（mixed_v21）
```
~/projects/TRaDA-data-real/mixed_v21/
└── mixed_fwd_bwdrev_9_1/  # 9:1 比例混合
    ├── data/chunk-000/
    ├── videos/chunk-000/
    └── meta/
```

---

## 3. 数据处理脚本

### 3.1 合并分散数据：`merge_datasets_v21.py`

将分散的episode目录合并为统一数据集。

**用法**：
```bash
cd /home/rogerspyke/projects/TRaDA/third_party/lerobot

# 仅合并（输出v2.1格式）
uv run python scripts/merge_datasets_v21.py

# 合并并转换为v3.0格式
uv run python scripts/merge_datasets_v21.py --convert-to-v30
```

**配置**：编辑脚本底部的 `DATASET_CONFIGS`：
```python
DATASET_CONFIGS = {
    "bwd": {
        "source_dir": Path.home() / "projects/TRaDA-data-real/scattered/Agilex_Cobot_Magic_insert_wedge_bwd_1910",
        "output_name": "Agilex_Cobot_Magic_insert_wedge_bwd_1910",
        "task_description": "Pick up the red triangular wedge from the white square mold and place it on the table.",
        "episode_prefix": "Agilex_Cobot_Magic_insert_wedge_bwd_1910_",
    },
}
```

### 3.2 混合多个数据集：`create_mixed_dataset_v21.py`

按比例混合多个数据集，用于多任务训练或数据增强。

**用法**：
```bash
# 混合 fwd 和 bwd-rev 数据集，比例 9:1
uv run python scripts/create_mixed_dataset_v21.py \
    --datasets \
        ~/projects/TRaDA-data-real/merged_v21/Agilex_Cobot_Magic_insert_wedge_fwd_1897 \
        ~/projects/TRaDA-data-real/merged_v21/Agilex_Cobot_Magic_insert_wedge_bwd_1910-rev \
    --ratios 9 1 \
    --output-dir ~/projects/TRaDA-data-real/mixed_v21 \
    --output-name mixed_fwd_bwdrev_9_1 \
    --seed 42

# 默认自动转换为v3.0格式，可用 --convert-to-v30 控制
```

**参数说明**：
| 参数 | 说明 |
|------|------|
| `--datasets` | 源数据集路径列表 |
| `--ratios` | 混合比例，如 `9 1` 表示 90%:10% |
| `--output-dir` | 输出目录 |
| `--output-name` | 输出数据集名称 |
| `--seed` | 随机种子（默认42） |
| `--convert-to-v30` | 转换为v3.0格式（默认启用） |

---

## 4. 训练

使用 `lerobot-train` 命令训练模型：

```bash
# 基本训练
uv run lerobot-train \
    --dataset.repo_id=local \
    --dataset.root ~/projects/TRaDA-data-real/mixed_v21/mixed_fwd_bwdrev_9_1 \
    --policy.type=act \
    --output_dir=outputs/train/act_mixed_9_1

# 完整参数示例
uv run lerobot-train \
    --dataset.repo_id=local \
    --dataset.root ~/projects/TRaDA-data-real/mixed_v21/mixed_fwd_bwdrev_9_1 \
    --policy.type=act \
    --training.batch_size=8 \
    --training.steps=50000 \
    --training.log_freq=500 \
    --training.save_freq=10000 \
    --output_dir=outputs/train/act_mixed_9_1 \
    --device=cuda
```

---

## 5. 完整流水线示例

### 场景：训练一个多任务模型，混合 fwd 和 bwd-rev 任务

**步骤 1：合并分散数据**
```bash
cd /home/rogerspyke/projects/TRaDA/third_party/lerobot

# 合并所有分散的episode目录
uv run python scripts/merge_datasets_v21.py --convert-to-v30
```

输出：
- `~/projects/TRaDA-data-real/merged_v21/Agilex_Cobot_Magic_insert_wedge_fwd_1897/` (v3.0)
- `~/projects/TRaDA-data-real/merged_v21/Agilex_Cobot_Magic_insert_wedge_bwd_1910/` (v3.0)

**步骤 2：创建混合数据集**
```bash
# 创建不同比例的混合数据集
for ratio in "9 1" "7 3" "5 5" "3 7" "1 9"; do
    name="mixed_fwd_bwdrev_${ratio// /_}"
    uv run python scripts/create_mixed_dataset_v21.py \
        --datasets \
            ~/projects/TRaDA-data-real/merged_v21/Agilex_Cobot_Magic_insert_wedge_fwd_1897 \
            ~/projects/TRaDA-data-real/merged_v21/Agilex_Cobot_Magic_insert_wedge_bwd_1910-rev \
        --ratios $ratio \
        --output-dir ~/projects/TRaDA-data-real/mixed_v21 \
        --output-name $name \
        --seed 42
done
```

输出：
- `mixed_fwd_bwdrev_9_1/` - 90% fwd + 10% bwd-rev
- `mixed_fwd_bwdrev_7_3/` - 70% fwd + 30% bwd-rev
- `mixed_fwd_bwdrev_5_5/` - 50% fwd + 50% bwd-rev
- `mixed_fwd_bwdrev_3_7/` - 30% fwd + 70% bwd-rev
- `mixed_fwd_bwdrev_1_9/` - 10% fwd + 90% bwd-rev

**步骤 3：训练模型**
```bash
# 训练 9:1 混合数据集
uv run lerobot-train \
    --dataset.repo_id=local \
    --dataset.root ~/projects/TRaDA-data-real/mixed_v21/mixed_fwd_bwdrev_9_1 \
    --policy.type=act \
    --training.steps=50000 \
    --output_dir=outputs/train/act_mixed_9_1 \
    --device=cuda

# 训练 5:5 混合数据集（均衡混合）
uv run lerobot-train \
    --dataset.repo_id=local \
    --dataset.root ~/projects/TRaDA-data-real/mixed_v21/mixed_fwd_bwdrev_5_5 \
    --policy.type=act \
    --training.steps=50000 \
    --output_dir=outputs/train/act_mixed_5_5 \
    --device=cuda
```

**步骤 4：评估模型**
```bash
# 使用 lerobot-eval 评估
uv run lerobot-eval \
    -p outputs/train/act_mixed_9_1/pretrained_model \
    eval.n_episodes=10 \
    device=cuda
```

---

## 6. 常见问题

### Q: 为什么要在 v2.1 格式下进行混合？
A: v2.1 格式每个 episode 是独立的文件，混合操作只需要复制和重命名文件，非常简单。v3.0 格式所有 episode 合并在一个文件中，混合需要拆分和重新合并，操作复杂。

### Q: 训练时需要什么格式？
A: 当前版本的 lerobot-train 只支持 v3.0 格式。脚本默认会自动转换。

### Q: 如何验证数据集是否正确？
A: 使用 Python 加载数据集：
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id='local',
    root='/path/to/dataset',
)
print(f"Episodes: {dataset.num_episodes}")
print(f"Frames: {len(dataset)}")
```

### Q: 混合数据集的 episode 数量如何计算？
A: 对于数据集 i，采样数量 = floor(该数据集总episodes × ratio_i / sum(ratios))。例如 9:1 混合两个各100 episodes 的数据集，结果为 90 + 10 = 100 episodes。

---

## 7. 数据集信息

当前可用数据集：

| 数据集 | Episodes | Frames | 任务描述 |
|--------|----------|--------|----------|
| `Agilex_Cobot_Magic_insert_wedge_fwd_1897` | 100 | 51,658 | 将楔子插入模具 |
| `Agilex_Cobot_Magic_insert_wedge_fwd_1897-rev` | 100 | ~51,658 | 插入任务的逆向 |
| `Agilex_Cobot_Magic_insert_wedge_bwd_1910` | 100 | 42,313 | 从模具取出楔子 |
| `Agilex_Cobot_Magic_insert_wedge_bwd_1910-rev` | 100 | ~42,313 | 取出任务的逆向 |

数据特征：
- 3个摄像头：top, left, right
- 动作维度：14（双臂关节 + 夹爪）
- 状态维度：26（双臂关节 + 夹爪 + 位姿）
- 帧率：30 FPS
