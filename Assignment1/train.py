import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from typing import List
from tqdm import tqdm
import argparse

# Feature Extraction 

def geometric_mean(arr: List[float]) -> float:
    arr = np.array(arr)
    arr = arr[arr > 0]
    if len(arr) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(arr))))

def parse_file(file_path: Path, chunk_size: int = 500) -> pd.DataFrame:
    """
    Parse a single training file (.txt) and return multiple aggregated feature rows (split B/W + chunking)
    Each chunk (e.g., 500 moves) becomes one training sample.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() != ""]

    feats = []  # collect per-chunk DataFrames
    i = 0
    prev_lead = None

    # temporary storage for one chunk
    feats_policy, feats_rank = [], []
    strength_list, winrate_list, lead_list, uncert_list, loss_list = [], [], [], [], []

    move_count = 0

    while i < len(lines):
        line = lines[i]
        if "[" in line and "]" in line:
            color = line[0]  # B or W

            # 1. Policy (9)
            p = list(map(float, lines[i + 1].replace("%", "").split()))
            if len(p) != 9:
                i += 1
                continue
            # 2. Value (9) - skip reading but move index correctly
            _ = lines[i + 2]
            # 3. Rank model (9)
            r = list(map(float, lines[i + 3].split()))
            # 4. Strength
            s = float(lines[i + 4])
            # 5. KataGo
            katago = lines[i + 5].split()
            if len(katago) >= 3:
                winrate = float(katago[0].replace("%", "")) / 100.0
                lead = float(katago[1])
                uncert = float(katago[2])
                if color == "W":
                    winrate = 1.0 - winrate
                    lead = -lead
                if prev_lead is not None:
                    loss_list.append(abs(lead - prev_lead))
                prev_lead = lead
                winrate_list.append(winrate)
                lead_list.append(lead)
                uncert_list.append(uncert)
            # accumulate
            feats_policy.append(p)
            feats_rank.append(r)
            strength_list.append(s)
            move_count += 1

            if move_count >= chunk_size:
                feats.append(_aggregate_chunk(
                    feats_policy, feats_rank, strength_list,
                    winrate_list, lead_list, uncert_list, loss_list
                ))
                # reset
                feats_policy, feats_rank = [], []
                strength_list, winrate_list, lead_list, uncert_list, loss_list = [], [], [], [], []
                move_count = 0
                prev_lead = None

        i += 1

    if move_count > 0:
        feats.append(_aggregate_chunk(
            feats_policy, feats_rank, strength_list,
            winrate_list, lead_list, uncert_list, loss_list
        ))

    return pd.DataFrame(feats)


def _aggregate_chunk(feats_policy, feats_rank, strength_list,
                     winrate_list, lead_list, uncert_list, loss_list) -> dict:
    """Helper to aggregate one chunk of features"""
    feats_policy = np.array(feats_policy) if len(feats_policy) else np.zeros((1, 9))
    feats_rank = np.array(feats_rank) if len(feats_rank) else np.zeros((1, 9))
    feat_dict = {}
    # Priors: geometric mean
    for j in range(9):
        feat_dict[f"gm_policy_{j+1}D"] = geometric_mean(feats_policy[:, j])
    # Rank model mean
    for j in range(9):
        feat_dict[f"mean_rank_{j+1}D"] = np.mean(feats_rank[:, j])
    # Strength
    feat_dict["mean_strength"] = np.mean(strength_list) if strength_list else 0
    feat_dict["std_strength"] = np.std(strength_list) if strength_list else 0
    # Loss
    feat_dict["mean_loss"] = np.mean(loss_list) if loss_list else 0
    feat_dict["std_loss"] = np.std(loss_list) if loss_list else 0
    # Winrate/Lead/Uncert
    feat_dict["mean_winrate"] = np.mean(winrate_list) if winrate_list else 0
    feat_dict["mean_lead"] = np.mean(lead_list) if lead_list else 0
    feat_dict["std_lead"] = np.std(lead_list) if lead_list else 0
    feat_dict["mean_uncert"] = np.mean(uncert_list) if uncert_list else 0
    return feat_dict

# Main 
def main(args):
    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_feats = []
    all_labels = []

    print(f"[Info] Parsing training data from {train_dir} ...")

    for rank in range(1, 10):
        fpath = train_dir / f"log_{rank}D_policy_train.txt"
        if not fpath.exists():
            print(f"[Warn] Missing file: {fpath}")
            continue
        df = parse_file(fpath)
        df["label"] = rank
        all_feats.append(df)
        all_labels += [rank]

    df_all = pd.concat(all_feats, ignore_index=True)
    X = df_all.drop(columns=["label"])
    y = df_all["label"] - 1  # 0–8 classes

    print(f"[Info] Training samples: {len(df_all)}  Features: {X.shape[1]}")

    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        objective="multi:softprob",
        num_class=9,
        eval_metric="mlogloss",
        random_state=42,
    )


    model.fit(X, y)
    y_pred = model.predict(X)
    acc = (y_pred == y).mean()
    print(f"[Info] Training Accuracy: {acc:.4f}")

    joblib.dump(model, out_dir / "xgb_rank_model.pkl")
    print(f"[Info] Model saved to {out_dir / 'xgb_rank_model.pkl'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Go Rank Predictor")
    parser.add_argument("--train_dir", type=str, default="./data/train",
                        help="Directory containing log_XD_policy_train.txt files")
    parser.add_argument("--out_dir", type=str, default="./weights",
                        help="Output directory for model weights")
    args = parser.parse_args()
    main(args)
