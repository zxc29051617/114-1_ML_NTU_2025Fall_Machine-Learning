import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange

import wandb
import matplotlib.pyplot as plt
from .metric import cal_2_wasserstein_dist, cal_energy_distance
import numpy as np


class GANGenerator(nn.Module):
    def __init__(self, hidden_dim: int = 128, z_dim: int = 2):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # 輸出 (x, y)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GANDiscriminator(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LeakyReLU(0.2), # GAN 建議用 LeakyReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1) # 輸出 Logits，不加 Sigmoid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GANPipeline:
    def __init__(self, generator: nn.Module, device: str = "cpu"):
        self.generator = generator
        self.device = device

    @torch.no_grad()
    def sample(self, sample_size: int = 5000) -> torch.Tensor:
        self.generator.eval()
        # 假設 z_dim 固定為 2
        z = torch.randn(sample_size, 2, device=self.device)
        return self.generator(z)


class GANTrainer:
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.G = generator
        self.D = discriminator
        self.opt_G = Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999)) # GAN 常用 beta1=0.5
        self.opt_D = Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCEWithLogitsLoss() # 內建 Sigmoid + BCELoss
        self.device = device

    def fit(self, dataloader: DataLoader, epochs: int, log_every: int) -> None:
        self.G.train()
        self.D.train()
        
        # 建立存圖的資料夾
        os.makedirs("results/gan", exist_ok=True)
        
        # 預先載入全部真實資料用來算分 (轉成 numpy)
        all_real_data = dataloader.dataset.tensors[0].numpy()
        
        print("Start Training...")
        for epoch in trange(epochs):
            for (real_data, ) in dataloader:
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # === 1. Train Discriminator ===
                self.opt_D.zero_grad()
                pred_real = self.D(real_data)
                loss_real = self.criterion(pred_real, torch.ones_like(pred_real))
                
                z = torch.randn(batch_size, self.G.z_dim, device=self.device) # 注意: 這裡改用 self.G.z_dim 比較保險
                fake_data = self.G(z)
                pred_fake = self.D(fake_data.detach())
                loss_fake = self.criterion(pred_fake, torch.zeros_like(pred_fake))
                
                loss_D = loss_real + loss_fake
                loss_D.backward()
                self.opt_D.step()
                
                # === 2. Train Generator ===
                self.opt_G.zero_grad()
                pred_fake_G = self.D(fake_data)
                loss_G = self.criterion(pred_fake_G, torch.ones_like(pred_fake_G))
                loss_G.backward()
                self.opt_G.step()

            # === 3. 記錄 Loss ===
            log_dict = {
                "epoch": epoch,
                "loss_G": loss_G.item(), 
                "loss_D": loss_D.item()
            }

            # === 4. 每 log_every (100) epochs 做一次評估與畫圖 ===
            if (epoch + 1) % log_every == 0:
                self.G.eval()
                with torch.no_grad():
                    # 生成 2500 個點來畫圖
                    z_vis = torch.randn(2500, self.G.z_dim, device=self.device)
                    fake_vis = self.G(z_vis).cpu().numpy()
                    
                    # A. 計算分數 (使用部分真實資料來加速計算)
                    # 隨機抽 2500 筆真實資料
                    idx = np.random.choice(len(all_real_data), 2500, replace=False)
                    real_sample = all_real_data[idx]
                    
                    wd = cal_2_wasserstein_dist(real_sample, fake_vis)
                    ed = cal_energy_distance(real_sample, fake_vis)
                    
                    log_dict["WD"] = wd
                    log_dict["ED"] = ed
                    
                    # B. 畫圖
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.scatter(fake_vis[:, 0], fake_vis[:, 1], s=1, alpha=0.5, label="Fake")
                    ax.set_xlim(-4, 4) # 根據 checkerboard 範圍固定視窗
                    ax.set_ylim(-4, 4)
                    ax.set_title(f"Epoch {epoch+1} | WD: {wd:.4f}")
                    plt.close(fig) # 關閉視窗避免吃記憶體
                    
                    # 存檔到本地 (做 GIF 用)
                    fig.savefig(f"results/gan/epoch_{epoch+1}.png")
                    
                    # 傳送到 WandB (直接看圖)
                    log_dict["Generated Distribution"] = wandb.Image(f"results/gan/epoch_{epoch+1}.png")
                
                self.G.train() # 切換回訓練模式

            wandb.log(log_dict)

            # 儲存模型 checkpoint
            if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(
                    {"generator": self.G.state_dict(), "discriminator": self.D.state_dict()},
                    Path("checkpoints", "gan.pth")
                )
