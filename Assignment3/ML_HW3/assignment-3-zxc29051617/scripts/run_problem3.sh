#!/bin/bash

# Usage:
#   bash scripts/run_problem3.sh PUBLIC_DIR PRIVATE_DIR
#
# PUBLIC_DIR  = 資料集 (train/test)
# PRIVATE_DIR = 存放 checkpoint / submission 的地方 (不能上傳到 GitHub)

PUBLIC_DIR="$1"
PRIVATE_DIR="$2"

if [ -z "$PUBLIC_DIR" ] || [ -z "$PRIVATE_DIR" ]; then
    echo "❌ Usage: bash scripts/run_problem3.sh PUBLIC_DIR PRIVATE_DIR"
    exit 1
fi

echo "==========================================="
echo "  Problem 3 - SSL Pretrain + Fine-tune"
echo "  PUBLIC_DIR = $PUBLIC_DIR"
echo "  PRIVATE_DIR = $PRIVATE_DIR"
echo "==========================================="

# Make sure PRIVATE_DIR exists
mkdir -p "$PRIVATE_DIR"

# ---------------------------------------------------------
# 1. SSL Pretrain
# ---------------------------------------------------------
echo "==========================================="
echo "  [1] SSL Pretrain (Masked Autoencoder)"
echo "==========================================="

uv run train.py \
    --config problem3_pretrain \
    train_dataset.data_dir="${PUBLIC_DIR}/train" \
    valid_dataset.data_dir="${PUBLIC_DIR}/train" \
    checkpoint.path="${PRIVATE_DIR}/ssl_pretrain_transunet.pth"

# ---------------------------------------------------------
# 2. Fine-tune using SSL weight
# ---------------------------------------------------------
echo "==========================================="
echo "  [2] Fine-tune with SSL-pretrained weights"
echo "==========================================="

uv run train.py \
    --config problem3_finetune \
    trainer.pretrain_path="${PRIVATE_DIR}/ssl_pretrain_transunet.pth" \
    train_dataset.data_dir="${PUBLIC_DIR}/train" \
    valid_dataset.data_dir="${PUBLIC_DIR}/train" \
    checkpoint.path="${PRIVATE_DIR}/problem3_finetune_best.pth"

# ---------------------------------------------------------
# 3. Generate submission.csv
# ---------------------------------------------------------
echo "==========================================="
echo "  [3] Generating submission CSV"
echo "==========================================="

uv run generate_submission.py \
    --config problem3_finetune \
    --public_dir "${PUBLIC_DIR}/test" \
    --private_dir "${PRIVATE_DIR}"

echo "==========================================="
echo "  Problem 3 - Done!"
echo "  Submission saved at ${PRIVATE_DIR}/submission.csv"
echo "==========================================="
