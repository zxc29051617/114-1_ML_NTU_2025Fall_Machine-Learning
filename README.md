# NTU Machine Learning 2025 Fall

這個 repository 收錄臺大 114-1 機器學習（李濬屹老師，CSIE5043）課程的講義、四份作業與期末專題。

作業涵蓋傳統機器學習、圍棋棋力與棋風辨識、醫學影像分割，以及 GAN、diffusion 等生成模型；
期末專題是小組改良免訓練的模型合併方法 EMR-Merging。

## 專案結構

```text
.
├── README.md
├── notes.md              # 課程筆記
├── lectures/             # 課程講義
├── Assignment1/          # 傳統機器學習與圍棋棋力預測
├── Assignment2/          # 圍棋棋風辨識（Who Play This Game）
├── Assignment3/          # 醫學影像分割
├── Assignment4/          # 生成模型：GAN、DDPM、DDIM、MeanFlow
└── Final/                # 期末專題：Layer-adaptive EMR 模型合併
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

路徑：[`Assignment3/`](Assignment3)

訓練分割模型產生二元 mask，比較三種做法：

- Problem 1：Simple-CNN UNet
- Problem 2：TransUNet（Transformer + UNet）
- Problem 3：先自監督預訓練，再 fine-tune 分割

實驗結果與分析寫在該資料夾的 README。

```bash
cd Assignment3
uv sync
python train.py --config problem1_bce
python generate_submission.py --config problem1_bce --public_dir dataset/public --private_dir dataset/private
```

## Assignment 4：生成模型

路徑：[`Assignment4/`](Assignment4)

在 2D checkerboard 資料上實作並比較 GAN、DDPM、DDIM，加分題實作 MeanFlow；以 Energy Distance 與
2-Wasserstein Distance 評估生成的分佈。

```bash
cd Assignment4
uv sync
uv run train.py --method gan          # 或 ddpm
uv run train_meanflow.py
uv run generate_sample.py --method ddim
```

## Final Project：Layer-adaptive EMR

路徑：[`Final/`](Final)

小組期末專題，組員是 Zheng-Yu Chen、Bo-Wei Lee、Yu-Lun Hung、Hsien-Ting Wang。
題目是 Layer-adaptive EMR: Four Modular and Stackable Enhancements for Robust Training-free Model Merging，
海報在 [`Final/FP-emr-merging/png/method-main.jpg`](Final/FP-emr-merging/png/method-main.jpg)。

以 [EMR-Merging](https://arxiv.org/abs/2405.17461)（NeurIPS 2024）為基礎，不用再訓練，把 8 個 fine-tuned 的 ViT-B/32
合併成一個模型。EMR-Merging 所有層共用同一個 unified task vector、二元 mask 和 rescaler，我們加了四個可以疊加的修改：

- Similarity-Based Hierarchical Clustering：先合併最相似的兩個任務，再和其他任務合併（`--cluster`）
- Low-rank Elect：用 SVD 只保留前幾個方向，濾掉雜訊（`--lora 4`）
- Importance-aware Soft Masks：用 sigmoid 算出的連續 mask 取代二元 mask（`--soft_mask_alpha 14`）
- Layer-wise Robust Scalar Rescale：每一層用 trimmed mean 算自己的 rescaler（`--trimmean 0.1`）

8 個任務的平均準確率：EMR-Merging 88.7%，只加 clustering 是 89.11%，最好；四個一起用是 87.3%，
主要是 soft mask 濾掉太多參數。

- `Final/FP-emr-merging/`：小組最後版本的程式（組員的 repo `zychen1204/FP-emr-merging`），
  另外改了 `torch.load` 等幾處，讓程式能在新版 PyTorch 上執行
- `Final/experiments/`：SVD rank 搜尋與 trimmed mean 的實驗腳本，
  依序執行 `extract_task_vectors.py`、`find_best_r.py`、`run_experiment_final.py`

```bash
cd Final/FP-emr-merging/merge_vit
python src/main_emr_merging.py --cluster
python src/main_emr_merging.py --lora 4 --trimmean 0.1 --cluster --soft_mask_alpha 14
python src/main_task_arithmetic.py
python src/main_ties_merging.py
```

執行前要把預訓練和各資料集 fine-tuned 的權重放進 `merge_vit/checkpoints/ViT-B-32/`，資料集放進 `merge_vit/data/`，
放法見 `merge_vit/checkpoints/README.md` 和 `merge_vit/data/README.md`。
