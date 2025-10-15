#!/bin/bash
# Test W&B Sweep integration

echo "======================================"
echo "W&B Sweep Integration Test"
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

echo "Starting sweep test (3 runs, 2 epochs each)..."
echo "This will take approximately 20-30 minutes"
echo ""

.venv/bin/python tools/sweep.py \
    --config_file configs/test_wandb.yml \
    --sweep-config sweeps/sweep_test.yaml

echo ""
echo "======================================"
echo "Sweep test completed!"
echo "Check W&B dashboard for results"
echo "======================================"

