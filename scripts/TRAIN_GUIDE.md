# LeRobot ACT Training Guide for TRaDA Project

## Overview

This guide provides step-by-step instructions for training ACT (Action Chunking with Transformers) models on TRaDA datasets using `lerobot-train`.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with >= 8GB VRAM (tested on RTX 4070 12GB)
- Sufficient disk space for datasets and checkpoints

### Software Requirements
```bash
cd /home/rogerspyke/projects/TRaDA/third_party/lerobot
uv sync --extra training
```

## Available Datasets

### Mixed Datasets (v3.0 format, ready for training)
```
/home/rogerspyke/projects/TRaDA-data-real/mixed/
├── mixed_fwd_bwdrev_1_9/   # 10% fwd + 90% bwd-rev, 43,131 frames
├── mixed_fwd_bwdrev_3_7/   # 30% fwd + 70% bwd-rev, 45,323 frames
├── mixed_fwd_bwdrev_5_5/   # 50% fwd + 50% bwd-rev, 47,381 frames
├── mixed_fwd_bwdrev_7_3/   # 70% fwd + 30% bwd-rev, 49,107 frames
└── mixed_fwd_bwdrev_9_1/   # 90% fwd + 10% bwd-rev, 50,608 frames
```

All datasets have:
- 100 episodes each
- v3.0 format (required for lerobot-train)
- Valid video files (verified)
- Task instruction: "Pick up the red triangular wedge from the table and insert it into the white square mold."

## Training Command

### Basic Training
```bash
cd /home/rogerspyke/projects/TRaDA/third_party/lerobot

uv run lerobot-train \
    --dataset.repo_id=local \
    --dataset.root=/home/rogerspyke/projects/TRaDA-data-real/mixed/mixed_fwd_bwdrev_9_1 \
    --dataset.use_imagenet_stats=false \
    --policy.type=act \
    --policy.push_to_hub=false \
    --output_dir=/home/rogerspyke/projects/TRaDA-data-real/models/act_mixed_9_1 \
    --steps=100000 \
    --save_freq=20000 \
    --batch_size=8 \
    --num_workers=4 \
    --tolerance_s=0.05
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset.repo_id` | Dataset identifier | "local" for local datasets |
| `--dataset.root` | Path to v3.0 dataset | Required |
| `--dataset.use_imagenet_stats` | ImageNet normalization | `false` (stats missing for video keys) |
| `--policy.type` | Policy architecture | `act` |
| `--steps` | Total training steps | `100000` |
| `--save_freq` | Checkpoint frequency | `20000` |
| `--batch_size` | Batch size | `8` (reduce if OOM) |
| `--tolerance_s` | Video timestamp tolerance | `0.01` |

### ACT-Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--policy.chunk_size` | Action prediction chunk | `100` |
| `--policy.optimizer_lr` | Learning rate | `1e-5` |
| `--policy.vision_backbone` | Vision encoder | `resnet18` |

## Creating New Mixed Datasets

Use `create_mixed_dataset_safe.py` to create new mixed datasets with safe video handling:

```bash
uv run python scripts/create_mixed_dataset_safe.py \
    --datasets \
        /home/rogerspyke/projects/TRaDA-data-real/merged/Agilex_Cobot_Magic_insert_wedge_fwd_1897 \
        /home/rogerspyke/projects/TRaDA-data-real/merged/Agilex_Cobot_Magic_insert_wedge_bwd_1910-rev \
    --ratios 5 5 \
    --output-dir /home/rogerspyke/projects/TRaDA-data-real/mixed \
    --output-name mixed_fwd_bwdrev_5_5 \
    --seed 42 \
    --crf 18 \
    --preset fast
```

**Note:** This script uses a robust two-pass video concatenation approach to avoid the corrupted H.264 stream issue that occurs with stream copy mode.

## Resume Training

```bash
uv run lerobot-train \
    --config_path=<output_dir>/checkpoints/<checkpoint_dir>/train_config.json \
    --resume=true
```

## Troubleshooting

### Video Decoding Errors
If you see `[h264] no frame!` errors, the video files are corrupted. Re-create the dataset using `create_mixed_dataset_safe.py`.

### Timestamp Tolerance Errors
If you see `FrameTimestampError`, increase tolerance: `--tolerance_s=0.05`

### Missing Stats Error
If you see `KeyError: 'observation.images.image_top'`, disable ImageNet stats: `--dataset.use_imagenet_stats=false`
