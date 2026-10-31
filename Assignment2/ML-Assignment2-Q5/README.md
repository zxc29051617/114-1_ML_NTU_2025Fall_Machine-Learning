# Machine Learning Assignment 2 — Q5: Who Play This Game

This README explains how to **set up and launch the official container environment** for **Assignment 2 – Q5**.

git clone --recursive https://github.com/gcobs104628/ML-Assignment2-Q5.git

https://www.kaggle.com/competitions/machine-learning-class-fall-2025-assignment-2-q-5  在 "Data" 頁面中下載test_set和train_set

### 請確保您的專案目錄（ML-Assignment2-Q5）包含以下結構：
```
檔案結構
請確保您的專案目錄（ML-Assignment2-Q5）包含以下結構：

ML-Assignment2-Q5/
│
├── .githooks/              
├── build/                  
├── minizero/               
├── scripts/                
│   ├── start-container.sh  # Script：啟動 Docker 容器
│   └── build.sh            # Script：編譯遊戲引擎
├── style_detection/       
├── test_set/               # 測試資料集
├── train_set/              # 訓練資料集
│
├── .autopep8               
├── .clang-format           
├── .gitignore              
├── .gitmodules             
├── CMakeLists.txt          
├── conf.cfg                # 專案的設定檔
├── Q5.py                   # Script：執行預測並產生提交檔案
├── train.py                # Script：執行模型訓練
└── README.md               # 本說明文件

```
---

## Environment Setup
### Enter the Project Directory
Make sure you are inside the cloned assignment repository.
```bash
# 啟動 WSL
wsl

# 進入專案資料夾（依存放位置調整）
cd ML-Assignment2-Q5
```
### Start the Container.
When successful, you will see a prompt like:  root@...:/workspace
```bash
./scripts/start-container.sh
```
### Build the Game Engine (Inside the Container)
```bash
./scripts/build.sh go
```

# Training
```bash
python train.py
```
訓練完成後，模型權重（checkpoints）將會儲存到 weights/ 資料夾中，模型。

# Q5.py
```bash
python Q5.py \
  --model_path weights/combined_epoch1.pt \
  --test_dir test_set \
  --output_csv submission.csv \
  --device cuda \
  --n_frames 10 \
  --move_stride 1 \
  --tta 1
```
使用訓練好的模型，對 test_set 中的資料進行預測，以產生 submission.csv。
# Kaggle 提交
產生的 submission.csv 檔案提交到 Kaggle 競賽頁面  
https://www.kaggle.com/competitions/machine-learning-class-fall-2025-assignment-2-q-5