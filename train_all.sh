#!/bin/bash

# This script runs the complete GDN training pipeline:
# 1. Stage 1 training (self-supervised forecasting)
# 2. Stage 2 training (fault detection with fault-injected data)

set -e

echo "============================================"
echo "Training Pipeline"
echo "============================================"
echo ""

# Configuration
RAW_DATA_PATH="data/carOBD/obdiidata"
SHARED_DATASET_DIR="data/shared_dataset"
CHECKPOINT_DIR="checkpoints"

# Stage 1 configuration
STAGE1_EPOCHS=140
STAGE1_BATCH_SIZE=32
STAGE1_LR=1e-3

# Stage 2 configuration
STAGE2_EPOCHS=120
STAGE2_BATCH_SIZE=32
STAGE2_LR=5e-4

# ============================================
# Step 1: Create shared dataset
# ============================================
echo "Step 1: Creating shared dataset..."
python data/create_shared_dataset.py \
    --raw-data-path "$RAW_DATA_PATH" \
    --output-dir "$SHARED_DATASET_DIR"

if [ $? -ne 0 ]; then
    echo "  ERROR: Shared dataset creation failed"
    exit 1
fi

echo "  Shared dataset created successfully"
echo ""

# ============================================
# Step 2: Stage 1 Training
# ============================================
echo "Step 2: Training Stage 1 (self-supervised forecasting)..."
echo ""

# Check if shared dataset exists
if [ ! -f "$SHARED_DATASET_DIR/train.npz" ]; then
    echo "  ERROR: Shared dataset not found. Please run Step 1 first."
    exit 1
fi

python training/train_stage1.py \
    --data_path "$SHARED_DATASET_DIR" \
    --epochs "$STAGE1_EPOCHS" \
    --batch_size "$STAGE1_BATCH_SIZE" \
    --lr "$STAGE1_LR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_name stage1_best

if [ $? -ne 0 ]; then
    echo "  ERROR: Stage 1 training failed"
    exit 1
fi

echo "  Stage 1 training completed successfully"
echo ""

# ============================================
# Step 3: Stage 2 Training
# ============================================
echo "Step 3: Training Stage 2 (fault detection)..."
echo ""

# Check if Stage 1 checkpoint exists
STAGE1_CHECKPOINT="$CHECKPOINT_DIR/stage1_best_forecast_stage1_best.pt"
if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "  ERROR: Stage 1 checkpoint not found at $STAGE1_CHECKPOINT. Please run Step 2 first."
    exit 1
fi

python training/train_stage2.py \
    --stage1_checkpoint "$STAGE1_CHECKPOINT" \
    --data_path "$SHARED_DATASET_DIR" \
    --epochs "$STAGE2_EPOCHS" \
    --batch_size "$STAGE2_BATCH_SIZE" \
    --lr "$STAGE2_LR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_name stage2_clean_best

if [ $? -ne 0 ]; then
    echo "  ERROR: Stage 2 training failed"
    exit 1
fi

echo "  Stage 2 training completed successfully"
echo ""

# ============================================
# Summary
# ============================================
echo "============================================"
echo "Training Pipeline Complete"
echo "============================================"
echo ""
echo "All checkpoints saved to: $CHECKPOINT_DIR"
echo "Ready for evaluation:"
echo "  - GDN-only: python llm/evaluation/evaluate_gdn_kg_llm.py --mode gdn_only"
echo "  - LLM baseline: python llm/evaluation/evaluate_llm_baseline.py"
echo "  - GDN-KG-LLM: python llm/evaluation/evaluate_gdn_kg_llm.py"
echo ""
