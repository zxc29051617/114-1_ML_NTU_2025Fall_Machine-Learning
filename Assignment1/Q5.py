# Q5.py
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
import argparse
from train import geometric_mean, parse_file

def extract_features(test_dir: Path, chunk_size: int = 500) -> pd.DataFrame:
    """Aggregate features; each file may produce multiple chunk-rows, each with the same id."""
    feats = []
    for file in sorted(test_dir.glob("*.txt")):
        df = parse_file(file, chunk_size=chunk_size)   # <-- 重要：和訓練時相同的 chunk_size
        df["id"] = file.stem                           # <-- 這裡直接把 id 貼到每個 chunk
        feats.append(df)
    return pd.concat(feats, ignore_index=True)

def main(args):
    model_path = Path(args.model_path)
    test_dir = Path(args.test_dir)
    output_csv = Path(args.output_csv)

    print(f"[Info] Loading model from {model_path}")
    model = joblib.load(model_path)

    print(f"[Info] Extracting test features from {test_dir}")
    df_test = extract_features(test_dir, chunk_size=args.chunk_size)

    # 特徵矩陣（去掉 id）
    X_test = df_test.drop(columns=["id"])

    print(f"[Info] Predicting ranks per chunk ...")
    # 取機率，稍後對同一 id 平均
    prob_chunk = model.predict_proba(X_test)  # shape: (num_chunks, 9)
    classes = model.classes_                  # 應該是 array([0,1,...,8])

    # 組成含 id 的 DataFrame
    prob_cols = [f"c{int(c)+1}" for c in classes]  # c1..c9
    proba_df = pd.DataFrame(prob_chunk, columns=prob_cols)
    proba_df["id"] = df_test["id"].values

    # 對同一 id 的多個 chunk「平均機率」
    avg = proba_df.groupby("id")[prob_cols].mean()

    # 取最大機率的類別（注意 classes 是 0..8，段位要 +1）
    pred_idx = np.argmax(avg.values, axis=1)
    pred_rank = (classes[pred_idx] + 1).astype(int)

    # 依照 id 的數字排序輸出（Kaggle 通常用數字順序）
    ids_sorted = sorted(avg.index, key=lambda x: int(x))
    id_to_rank = {k: v for k, v in zip(avg.index.tolist(), pred_rank.tolist())}
    submission = pd.DataFrame({
        "id": ids_sorted,
        "rank": [id_to_rank[i] for i in ids_sorted]
    })
    submission.to_csv(output_csv, index=False)
    print(f"[Success] submission.csv generated at {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kaggle submission for Go Rank Prediction")
    parser.add_argument("--model_path", type=str, default="./weights/xgb_rank_model.pkl")
    parser.add_argument("--test_dir", type=str, default="./test_set")
    parser.add_argument("--output_csv", type=str, default="submission.csv")
    parser.add_argument("--chunk_size", type=int, default=500, help="must match training chunk size")
    args = parser.parse_args()
    main(args)
