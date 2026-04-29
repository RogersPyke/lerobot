#!/bin/bash
# ============================================================================
# Parallel ACT Training Script for TRaDA Mixed Datasets
# ============================================================================
#
# This script runs parallel training on two GPUs for four datasets:
#   - mixed_fwd_bwdrev_9_1 (90% forward + 10% backward-reversed)
#   - mixed_fwd_bwdrev_7_3 (70% forward + 30% backward-reversed)
#   - mixed_fwd_bwdrev_3_7 (30% forward + 70% backward-reversed)
#   - mixed_fwd_bwdrev_1_9 (10% forward + 90% backward-reversed)
#
# Configuration:
#   - 30,000 training steps per model
#   - Checkpoint every 10,000 steps
#   - Batch size: 4 (per GPU) - reduced for 12GB VRAM
#   - Two RTX 4070 GPUs (12GB each)
#
# Usage:
#   cd /home/rogerspyke/projects/TRaDA/third_party/lerobot
#   chmod +x scripts/run_parallel_training.sh
#   ./scripts/run_parallel_training.sh
#
# ============================================================================

set -e

# Configuration
STEPS=30000
SAVE_FREQ=10000
BATCH_SIZE=4
NUM_WORKERS=4
TOLERANCE_S=0.05

# Paths
MIXED_DIR="/home/rogerspyke/projects/TRaDA-data-real/mixed"
OUTPUT_BASE="/home/rogerspyke/projects/TRaDA-models"
LOG_DIR="/home/rogerspyke/projects/TRaDA-models/logs"

# Datasets (4 models: mixed_9_1, mixed_7_3, mixed_3_7, mixed_1_9)
# Schedule:
#   Round 1: GPU 0 -> mixed_9_1, GPU 1 -> mixed_7_3 (parallel)
#   Round 2: GPU 0 -> mixed_3_7, GPU 1 -> mixed_1_9 (parallel)
DATASETS=(
    "mixed_fwd_bwdrev_9_1"
    "mixed_fwd_bwdrev_7_3"
    "mixed_fwd_bwdrev_3_7"
    "mixed_fwd_bwdrev_1_9"
)

# Create directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$LOG_DIR"

# Get timestamp for this training session
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Parallel ACT Training - Started at $TIMESTAMP"
echo "=========================================="
echo "Configuration:"
echo "  Steps: $STEPS"
echo "  Save frequency: $SAVE_FREQ"
echo "  Batch size: $BATCH_SIZE"
echo "  GPUs: 0, 1"
echo "  Datasets: ${DATASETS[@]}"
echo "=========================================="

# Function to run training on a specific GPU
run_training() {
    local GPU_ID=$1
    local DATASET_NAME=$2
    local DATASET_PATH="$MIXED_DIR/$DATASET_NAME"
    local OUTPUT_DIR="$OUTPUT_BASE/${DATASET_NAME}_act_${TIMESTAMP}"
    local LOG_FILE="$LOG_DIR/${DATASET_NAME}_gpu${GPU_ID}_${TIMESTAMP}.log"

    echo "[GPU $GPU_ID] Starting training on $DATASET_NAME"
    echo "[GPU $GPU_ID] Output: $OUTPUT_DIR"
    echo "[GPU $GPU_ID] Log: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU_ID uv run lerobot-train \
        --dataset.repo_id=local \
        --dataset.root="$DATASET_PATH" \
        --dataset.use_imagenet_stats=false \
        --policy.type=act \
        --policy.push_to_hub=false \
        --output_dir="$OUTPUT_DIR" \
        --steps=$STEPS \
        --save_freq=$SAVE_FREQ \
        --batch_size=$BATCH_SIZE \
        --num_workers=$NUM_WORKERS \
        --tolerance_s=$TOLERANCE_S \
        --seed=42 \
        2>&1 | tee "$LOG_FILE"

    local EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[GPU $GPU_ID] Completed: $DATASET_NAME"
    else
        echo "[GPU $GPU_ID] FAILED: $DATASET_NAME (exit code: $EXIT_CODE)"
    fi

    return $EXIT_CODE
}

# Training schedule:
# Round 1: GPU 0 -> mixed_9_1, GPU 1 -> mixed_7_3 (parallel)
# Round 2: GPU 0 -> mixed_3_7, GPU 1 -> mixed_1_9 (parallel)

echo ""
echo "=== Round 1: Training mixed_9_1 (GPU 0) and mixed_7_3 (GPU 1) ==="
run_training 0 "${DATASETS[0]}" &
PID_0=$!
run_training 1 "${DATASETS[1]}" &
PID_1=$!

wait $PID_0
EXIT_0=$?
wait $PID_1
EXIT_1=$?

echo ""
echo "=== Round 1 Complete ==="
[ $EXIT_0 -eq 0 ] && echo "  GPU 0 (mixed_9_1): SUCCESS" || echo "  GPU 0 (mixed_9_1): FAILED"
[ $EXIT_1 -eq 0 ] && echo "  GPU 1 (mixed_7_3): SUCCESS" || echo "  GPU 1 (mixed_7_3): FAILED"

echo ""
echo "=== Round 2: Training mixed_3_7 (GPU 0) and mixed_1_9 (GPU 1) ==="
run_training 0 "${DATASETS[2]}" &
PID_0=$!
run_training 1 "${DATASETS[3]}" &
PID_1=$!

wait $PID_0
EXIT_0=$?
wait $PID_1
EXIT_1=$?

echo ""
echo "=== Round 2 Complete ==="
[ $EXIT_0 -eq 0 ] && echo "  GPU 0 (mixed_3_7): SUCCESS" || echo "  GPU 0 (mixed_3_7): FAILED"
[ $EXIT_1 -eq 0 ] && echo "  GPU 1 (mixed_1_9): SUCCESS" || echo "  GPU 1 (mixed_1_9): FAILED"

echo ""
echo "=========================================="
echo "All training completed at $(date)"
echo "=========================================="
echo "Models saved to: $OUTPUT_BASE"
echo "Logs saved to: $LOG_DIR"