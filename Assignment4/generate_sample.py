import os
import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

# 引入原本的模組
from src.diffusion import DDPMScheduler, DDIMScheduler, DDPMPipeline, Denoiser, DDIMPipeline
from src.gan import GANGenerator, GANPipeline
from src.metric import cal_2_wasserstein_dist, cal_energy_distance
from src.utils import set_random_seed, get_device

# [新增] 引入 MeanFlow 相關模組
# 假設你把 MeanFlow 的類別和取樣函式放在 train_meanflow.py 和 inference_meanflow.py
# 如果你放在 src 裡，請自行調整 import 路徑
try:
    from train_meanflow import MeanFlowDenoiser
    from inference_meanflow import meanflow_sample
except ImportError:
    print("⚠️ Warning: MeanFlow modules not found. Ensure train_meanflow.py exists.")
    MeanFlowDenoiser = None
    meanflow_sample = None


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim", "gan", "meanflow"], # [新增] 加入 meanflow 選項
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    
    # 讀取 Config (meanflow 暫時共用 ddpm 的 config 或是預設值)
    config_path = Path("configs", f"{args.method}.yaml")
    if not config_path.exists() and args.method in ["ddim", "meanflow"]:
        config_path = Path("configs", "ddpm.yaml")
        
    config = OmegaConf.load(config_path)

    # 載入 Ground Truth (用於計算分數)
    target = np.load(Path("data", "checkerboard.npy"))
    target = target.astype(np.float32)
    device = get_device()

    # 初始化 pipe 或 samples 變數
    pipe = None
    samples = None

    # === 分支處理不同模型 ===
    
    if args.method in ["ddpm", "ddim"]:
        print(f"Loading Diffusion Model for {args.method.upper()}...")
        model = Denoiser(hidden_dim=config.model.hidden_dim).to(device)
        
        ckpt_path = Path("checkpoints", "ddpm.pth")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到權重: {ckpt_path}")
            
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

        if args.method == "ddpm":
            scheduler = DDPMScheduler(
                num_timesteps=config.scheduler.num_timesteps,
                beta_start=config.scheduler.beta_start,
                beta_end=config.scheduler.beta_end,
                device=device
            )
            pipe = DDPMPipeline(model, scheduler, device)
            print(f"Generating samples using {args.method.upper()}...")
            samples = pipe.sample(sample_size=5000).cpu().numpy()

        elif args.method == "ddim":
            scheduler = DDIMScheduler(
                num_timesteps=config.scheduler.num_timesteps,
                beta_start=config.scheduler.beta_start,
                beta_end=config.scheduler.beta_end,
                device=device,
                eta=0.0 
            )
            pipe = DDIMPipeline(model, scheduler, num_inference_steps=50, device=device)
            print(f"Generating samples using {args.method.upper()}...")
            samples = pipe.sample(sample_size=5000).cpu().numpy()

    elif args.method == "gan":
        print("Loading GAN Model...")
        model = GANGenerator(hidden_dim=config.generator.hidden_dim, z_dim=2).to(device)
        
        ckpt_path = Path("checkpoints", "gan.pth")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到權重: {ckpt_path}")
            
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and "generator" in checkpoint:
            model.load_state_dict(checkpoint["generator"])
        else:
            model.load_state_dict(checkpoint)
        
        pipe = GANPipeline(generator=model, device=device)
        print(f"Generating samples using {args.method.upper()}...")
        samples = pipe.sample(sample_size=5000).cpu().numpy()

    elif args.method == "meanflow":
        # [新增] MeanFlow 生成邏輯
        if MeanFlowDenoiser is None:
            raise ImportError("無法載入 MeanFlow 模組，請確認 train_meanflow.py 存在。")
            
        print("Loading MeanFlow Model...")
        # 這裡的 hidden_dim 可以寫死 128 或從 config 讀
        model = MeanFlowDenoiser(hidden_dim=128).to(device)
        
        ckpt_path = Path("checkpoints", "meanflow.pth")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到權重: {ckpt_path}. 請先執行 train_meanflow.py")
            
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        print(f"Generating samples using {args.method.upper()} (1-step)...")
        # 直接呼叫我們之前寫好的 sampling function
        # 注意：meanflow_sample 回傳的已經是 numpy array
        samples = meanflow_sample(model, 5000, device)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    # === 評估與存檔區 (所有方法共用) ===
    
    # 確保 samples 存在
    if samples is None:
        raise RuntimeError("Sample generation failed.")

    # 計算分數
    wd = cal_2_wasserstein_dist(samples, target)
    ed = cal_energy_distance(samples, target)
    print(f"[{args.method.upper()}] WD: {wd:.4f} | ED: {ed:.4f}")

    os.makedirs("results", exist_ok=True)
    
    # 1. 儲存 .npy
    np.save(Path("results", f"{args.method}_generated_sample.npy"), samples)
    
    # 2. 儲存 .csv (Kaggle Submission)
    csv_path = Path("results", f"{args.method}_submission.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "x", "y"])
        for i, (x, y) in enumerate(samples):
            writer.writerow([f"{i+1:04d}", x, y])
            
    print(f"✅ Submission CSV saved to {csv_path}")

    # 3. 繪圖
    plt.figure(figsize=(6, 6), dpi=150)
    plt.scatter(target[:, 0], target[:, 1], s=5, alpha=0.5, label="GT", c='gray', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, label=args.method.upper(), c='blue', marker='.')
    plt.xlim(-4.5, 4.5)
    plt.ylim(-4.5, 4.5)
    plt.legend()
    plt.title(f"{args.method.upper()} (WD: {wd:.4f})")
    plt.savefig(Path("results", f"{args.method}.png"))
    plt.close()
    print(f"✅ Plot saved to results/{args.method}.png")