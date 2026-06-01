#!/bin/bash

pip install gdown

gdown "https://drive.google.com/uc?id=1ZYSu6QA-RS-X-lN0GO_cUs-0_qJ8_PUx" -O problem1_best.pth

# 3. 執行預測
uv run generate_submission.py --config cnn_unet --public_dir "${1}" --private_dir "${2}"