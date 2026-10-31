#!/bin/bash

# 檢查是否有傳入配置名稱參數 ($1)
if [ -z "$1" ]; then
    echo "錯誤: 請提供要執行的配置檔案名稱 (例如: problem1_bce_aug)"
    exit 1
fi

# 執行訓練指令 (採用最直接的方式)
uv run train.py --config "$1"