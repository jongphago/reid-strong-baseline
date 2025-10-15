#!/bin/bash
# Test W&B integration with minimal training

echo "======================================"
echo "W&B Integration Test"
echo "======================================"

# Check if WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY is not set"
    echo "Please set it with: export WANDB_API_KEY='your_api_key_here'"
    echo "Or login with: wandb login"
    echo ""
fi

# Check if data exists
if [ ! -d "./data/market1501" ]; then
    echo "Error: Market1501 dataset not found at ./data/market1501"
    echo "Please download and extract the dataset first"
    exit 1
fi

echo "Starting test training (5 epochs)..."
echo ""

python tools/train.py \
    --config_file='configs/test_wandb.yml' \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('market1501')" \
    OUTPUT_DIR "./log/test_wandb"

echo ""
echo "======================================"
echo "Test completed!"
echo "Check W&B dashboard for results"
echo "======================================"

