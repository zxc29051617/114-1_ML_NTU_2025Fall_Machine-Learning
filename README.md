# NTU Machine Learning 2025 Fall

這個 repository 收錄臺大 114-1 機器學習（CSIE5043）課程的講義、四份作業與期末專題。

作業涵蓋傳統機器學習、圍棋棋力與棋風辨識、醫學影像分割，以及 GAN、diffusion 等生成模型；
期末專題研究免調參的模型合併方法 EMR-Merging。

## 專案結構

```text
.
├── README.md
├── notes.md              # 課程筆記
├── Lecture *.pdf         # 課程講義
├── Assignment1/          # 傳統機器學習與圍棋棋力預測
├── Assignment2/          # 圍棋棋風辨識（Who Play This Game）
├── Assignment3/          # 醫學影像分割
├── Assignment4/          # 生成模型：GAN、DDPM、DDIM、MeanFlow
└── Final/                # 期末專題：EMR-Merging 模型合併
```

## Assignment 1：傳統機器學習與圍棋棋力預測

路徑：[`Assignment1/`](Assignment1)

- Q1–Q4：perceptron、MLP、linear / logistic regression、SVR、SVM、decision tree、random forest 等基礎方法
  （`ML_Assignment1_Q1_Q4_r13945041.ipynb`）。
- Q5：從圍棋對局的 policy、rank 與 KataGo 相關特徵，用 XGBoost 預測棋力等級（`train.py`、`Q5.py`）。

```bash
cd Assignment1
pip install numpy pandas scikit-learn xgboost lightgbm joblib tqdm scipy
python train.py --train_dir train_set --out_dir weights
python Q5.py --model_path weights/xgb_rank_model.pkl --test_dir test_set --output_csv submission.csv --chunk_size 500
```

## Assignment 2：圍棋棋風辨識（Who Play This Game）

路徑：[`Assignment2/ML-Assignment2-Q5/`](Assignment2/ML-Assignment2-Q5)

從 SGF 對局學習每位棋手的棋風 embedding，再比對 query games 與 candidate players，找出是誰下的棋。
程式建立在 MiniZero（`minizero/`）與棋風偵測模型（`style_detection/`）上。

- `train.py`：訓練棋風 embedding 模型（設定在 `conf.cfg`）
- `Q5.py`：計算 candidate 與 query 的相似度並輸出 submission

在 container 中執行：

```bash
cd Assignment2/ML-Assignment2-Q5
./scripts/start-container.sh
./scripts/build.sh go          # 在 container 內編譯圍棋引擎
python train.py
python Q5.py --model_path weights/combined_epoch1.pt --test_dir test_set --output_csv submission.csv --device cuda
```

## Assignment 3：醫學影像分割

路徑：[`Assignment3/ML_HW3/assignment-3-zxc29051617/`](Assignment3/ML_HW3/assignment-3-zxc29051617)

訓練分割模型產生二元 mask，比較三種做法：

- Problem 1：Simple-CNN UNet
- Problem 2：TransUNet（Transformer + UNet）
- Problem 3：先自監督預訓練，再 fine-tune 分割

實驗結果與分析寫在該資料夾的 README。

```bash
cd Assignment3/ML_HW3/assignment-3-zxc29051617
uv sync
python train.py --config problem1_bce
python generate_submission.py --config problem1_bce --public_dir dataset/public --private_dir dataset/private
```

## Assignment 4：生成模型

路徑：[`Assignment4/ML4/assignment-4-zxc29051617/`](Assignment4/ML4/assignment-4-zxc29051617)

在 2D checkerboard 資料上實作並比較 GAN、DDPM、DDIM，加分題實作 MeanFlow；以 Energy Distance 與
2-Wasserstein Distance 評估生成的分佈。

```bash
cd Assignment4/ML4/assignment-4-zxc29051617
uv sync
uv run train.py --method gan          # 或 ddpm
uv run train_meanflow.py
uv run generate_sample.py --method ddim
```

## Final Project：EMR-Merging

路徑：[`Final/merge_vit_final/`](Final/merge_vit_final)

以 [EMR-Merging](https://arxiv.org/abs/2405.17461)（NeurIPS 2024）為基礎：把多個 task-specific 模型的 task vector
合併成一個，再用每個任務的 mask 與 rescaler 還原各任務需要的方向與大小。主要實驗在 Vision Transformer
（`merge_vit/`），並和 Task Arithmetic、TIES-Merging、AdaMerging 比較；另外嘗試 hierarchical clustering、
SVD refinement、soft mask、trimmed-mean rescaling 等修改。

```bash
cd Final/merge_vit_final/merge_vit/src
python main_task_arithmetic.py
python main_ties_merging.py
python main_emr_merging.py
```

需要先準備 `checkpoints/ViT-B-32/` 的預訓練與各資料集 fine-tuned 權重，以及 `merge_vit/data/` 的資料集。
`Final/FP-emr-merging/` 是另一份版本，多了 SVD、ablation 與參數搜尋等實驗腳本（`merge_vit/` 下的 `eval_ablation.py`、`find_best_r.py` 等）。
