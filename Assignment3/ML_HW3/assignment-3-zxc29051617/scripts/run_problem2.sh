#!/bin/bash

pip install gdown

gdown "https://drive.google.com/uc?id=1CSClGTbQbWxj6o70dj2YeONvVpsXyQ02" -O problem2.pth

uv run generate_submission.py --config problem2 --public_dir "${1}" --private_dir "${2}"