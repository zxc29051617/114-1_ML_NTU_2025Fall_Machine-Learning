from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
import wandb 

# === 1. 引入 GAN 的工具 ===
from src.gan import GANGenerator, GANDiscriminator, GANPipeline, GANTrainer

# === 2. 引入 DDPM 的工具 (這裡補上了！) ===
from src.diffusion import DDPMScheduler, DDPMPipeline, Denoiser, DiffusionTrainer

from src.utils import set_random_seed, get_device
from src.metric import cal_2_wasserstein_dist, cal_energy_distance

SAMPLE_SIZE = 5000

def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="ddpm",
        choices=["ddpm", "gan"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.method}.yaml"))

    # 初始化 WandB
    wandb.init(
        entity="r13945041-national-taiwan-university",  
        project="ml_hw4",                                
        name=f"{args.method}_run",
        config=OmegaConf.to_container(config)
    )

    # 讀取資料
    target_dist = np.load(Path("data", "checkerboard.npy")).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(target_dist))
    dataloader = DataLoader(dataset, **config.dataloader)
    device = get_device()

    # ==========================================
    #  這裡就是主要的判斷區塊：GAN vs DDPM
    # ==========================================

    if args.method == "ddpm":
        # === DDPM 的初始化 (填滿這裡) ===
        print("Initializing DDPM...")
        
        # 1. 建立排程器 (Scheduler) - 負責管數學公式
        scheduler = DDPMScheduler(
            num_timesteps=config.scheduler.num_timesteps,
            beta_start=config.scheduler.beta_start,
            beta_end=config.scheduler.beta_end,
            device=device
        )
        
        # 2. 建立模型 (Denoiser) - 負責預測雜訊
        model = Denoiser(
            hidden_dim=config.model.hidden_dim
        ).to(device)
        
        # 3. 建立訓練器 (Trainer) - 裡面已經包好了 Pipeline
        trainer = DiffusionTrainer(
            model=model,
            scheduler=scheduler,
            lr=config.trainer.lr,
            device=device
        )
        
    elif args.method == "gan":
        # === GAN 的初始化 (保持不變) ===
        print("Initializing GAN...")
        
        generator = GANGenerator(
            hidden_dim=config.generator.hidden_dim,
            z_dim=2
        ).to(device)
        
        discriminator = GANDiscriminator(
            hidden_dim=config.discriminator.hidden_dim
        ).to(device)
        
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            lr=config.trainer.lr,
            device=device
        )
        
        # 額外建立 pipeline 供最後評估用
        pipeline = GANPipeline(generator, device)
        
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # ==========================================
    #  開始訓練 (不管哪種方法都呼叫 fit)
    # ==========================================
    trainer.fit(
        dataloader, 
        epochs=config.trainer.epochs, 
        log_every=config.trainer.log_every
    )
    
    # ==========================================
    #  訓練結束後的最終評估 (Optional)
    # ==========================================
    print("Calculating final metrics...")
    
    # 根據方法不同，呼叫不同的採樣方式
    if args.method == "gan":
        fake_data = pipeline.sample(SAMPLE_SIZE).cpu().numpy()
    elif args.method == "ddpm":
        # DDPM 的 Trainer 裡我們有存 self.pipe，可以直接用
        fake_data = trainer.pipe.sample(SAMPLE_SIZE).cpu().numpy()

    # 從真實資料隨機抽樣來比較
    idx = np.random.choice(len(target_dist), SAMPLE_SIZE, replace=False)
    real_data = target_dist[idx]
    
    wd = cal_2_wasserstein_dist(real_data, fake_data)
    ed = cal_energy_distance(real_data, fake_data)
    
    print(f"Final WD: {wd:.4f}, ED: {ed:.4f}")
    wandb.log({"Final_WD": wd, "Final_ED": ed})