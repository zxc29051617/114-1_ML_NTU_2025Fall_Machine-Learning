import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np
import wandb
import matplotlib.pyplot as plt

# 引用算分工具 (假設 metric.py 在 src 資料夾外，或是根據你的結構調整 import)
try:
    from metric import cal_2_wasserstein_dist, cal_energy_distance
except ImportError:
    from .metric import cal_2_wasserstein_dist, cal_energy_distance


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DDPMScheduler:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cuda",
    ):
        self.T = num_timesteps
        self.device = device
        
        # 1. 計算 Betas (Linear Schedule)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        
        # 2. 計算 Alphas
        self.alphas = 1.0 - self.betas
        
        # 3. 計算 Alpha Bars (累乘)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)

    def forward(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # 公式: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        
        # 為了讓維度對齊 (Batch, 1)，我們用 gather 或索引
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1, 1)
        
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def reverse(self, sample: torch.Tensor, t: int, eps: torch.Tensor) -> torch.Tensor:
        # 這是 Sampling 用的一個 Step
        # 公式: x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps) + sigma_t * z
        
        alpha_t = self.alphas[t]
        beta_t = self.betas[t]
        alpha_bar_t = self.alpha_bars[t]
        
        # 1. 計算均值 Mean
        coef_eps = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (sample - coef_eps * eps)
        
        # 2. 如果不是最後一步 (t=0)，就加上隨機雜訊
        if t > 0:
            noise = torch.randn_like(sample)
            sigma_t = torch.sqrt(beta_t) # 或使用 posterior variance
            return mean + sigma_t * noise
        else:
            return mean


class DDPMPipeline:
    def __init__(
        self,
        model: nn.Module,
        scheduler: DDPMScheduler,
        device: str = "cuda",
    ):
        self.model = model
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def sample(self, sample_size: int = 2500) -> torch.Tensor:
        self.model.eval()
        
        # 1. 從純雜訊開始 (x_T)
        x = torch.randn(sample_size, 2).to(self.device)
        
        # 2. 一步一步往回推 (從 T-1 到 0)
        for t in range(self.scheduler.T - 1, -1, -1):
            # 建立由 t 組成的 batch
            t_tensor = torch.full((sample_size,), t, device=self.device, dtype=torch.long)
            
            # 預測雜訊
            predicted_noise = self.model(x, t_tensor)
            
            # 算出上一步
            x = self.scheduler.reverse(x, t, predicted_noise)
            
        return x


class Denoiser(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        # 時間編碼器 (把數字 t 變成向量)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 輸入座標編碼
        self.input_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU()
        )
        
        # 主網路 (Concatenate 後處理)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # 接在一起所以是 *2
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # 輸出預測的雜訊 noise_x, noise_y
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 1. 處理時間
        t_emb = self.time_mlp(t)
        # 2. 處理座標
        x_emb = self.input_mlp(x)
        # 3. 接起來
        h = torch.cat([x_emb, t_emb], dim=1)
        return self.net(h)


class DiffusionTrainer:
    def __init__(
        self,
        model: nn.Module,
        scheduler: DDPMScheduler, # 注意這裡多傳入 scheduler 方便 fit 使用
        lr: float = 1e-3,
        device: str = "cuda",
    ):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.device = device
        # 建立 Pipeline 用於評估
        self.pipe = DDPMPipeline(model, scheduler, device)

    def fit(self, dataloader: DataLoader, epochs: int = 10000, log_every: int = 100):
        self.model.train()
        os.makedirs("results/ddpm", exist_ok=True)
        
        # 預先載入真實資料用於評估
        all_real_data = dataloader.dataset.tensors[0].numpy()

        print("Start Training DDPM...")
        for epoch in trange(epochs):
            loss_total = 0
            count = 0
            for x0, in dataloader:
                x0 = x0.to(self.device)
                batch_size = x0.size(0)
                
                # === 1. 隨機採樣時間 t ===
                t = torch.randint(0, self.scheduler.T, (batch_size,), device=self.device).long()
                
                # === 2. 產生雜訊 epsilon ===
                noise = torch.randn_like(x0)
                
                # === 3. 加噪 (Forward) 得到 x_t ===
                x_t = self.scheduler.forward(x0, t, noise)
                
                # === 4. 預測雜訊 ===
                pred_noise = self.model(x_t, t)
                
                # === 5. 算 Loss (MSE) ===
                loss = self.criterion(pred_noise, noise)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss_total += loss.item()
                count += 1
            
            avg_loss = loss_total / count
            log_dict = {"epoch": epoch, "loss": avg_loss}
            
            # === 評估與畫圖 (每 log_every 次) ===
            if (epoch + 1) % log_every == 0:
                # 採樣
                samples = self.pipe.sample(sample_size=2500).cpu().numpy()
                
                # 算分
                idx = np.random.choice(len(all_real_data), 2500, replace=False)
                real_sample = all_real_data[idx]
                wd = cal_2_wasserstein_dist(real_sample, samples)
                ed = cal_energy_distance(real_sample, samples)
                
                log_dict["WD"] = wd
                log_dict["ED"] = ed
                
                # 畫圖
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, label="Generated")
                ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
                ax.set_title(f"DDPM Epoch {epoch+1} | WD: {wd:.4f}")
                plt.close(fig)
                fig.savefig(f"results/ddpm/epoch_{epoch+1}.png")
                
                log_dict["Generated Distribution"] = wandb.Image(f"results/ddpm/epoch_{epoch+1}.png")
                
                # 存檔
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(self.model.state_dict(), Path("checkpoints", "ddpm.pth"))
                
            wandb.log(log_dict)
        
        # 訓練結束存最後一次
        torch.save(self.model.state_dict(), Path("checkpoints", "ddpm.pth"))

class DDIMScheduler(DDPMScheduler):
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cuda",
        eta: float = 0.0, # <--- 關鍵參數: 0 代表 DDIM (Deterministic), 1 代表 DDPM
    ):
        super().__init__(num_timesteps, beta_start, beta_end, device)
        self.eta = eta

    def step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        pred_noise: torch.Tensor,
    ) -> torch.Tensor:
        # === DDIM 的核心公式 ===
        
        # 1. 取得 alpha_bar (累積的保留率)
        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_t_prev = self.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(self.device)
        
        # 2. 計算 sigma (雜訊的多寡)
        # 如果 eta=0 (DDIM)，這裡算出來會是 0 -> 代表過程完全沒有隨機性
        sigma_t = self.eta * torch.sqrt(
            (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
        )
        
        # 3. 預測 x0 (這一步是 DDIM 能跳步的關鍵：它先猜原本的圖長怎樣)
        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        
        # 4. 計算指向 x_{t-1} 的方向 (Direction pointing to x_t_prev)
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * pred_noise
        
        # 5. 組合出 x_{t_prev}
        # 新的圖 = (保留部分 * 預測的原圖) + (變化方向) + (隨機雜訊)
        noise = torch.randn_like(x_t)
        x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        
        return x_prev


class DDIMPipeline:
    def __init__(self, model: nn.Module, scheduler: DDIMScheduler, num_inference_steps: int = 50, device="cuda"):
        self.model = model
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.device = device

    @torch.no_grad()
    def sample(self, sample_size: int = 1) -> torch.Tensor:
        self.model.eval()
        
        # 1. 設定時間步 (Timesteps)
        # DDPM 是 999, 998, ... 0 (走 1000 步)
        # DDIM 是 999, 799, 599 ... 0 (只走 50 步，大步跳躍)
        timesteps = torch.linspace(
            self.scheduler.T - 1, 0, self.num_inference_steps
        ).long().to(self.device)
        
        # 2. 從純雜訊開始
        x = torch.randn(sample_size, 2).to(self.device)
        
        # 3. 開始倒數 (跳著走)
        for i, t in enumerate(timesteps):
            # 算出下一步的時間點 (prev_t)
            # 例如現在 t=999, 下一步可能就是 t=799
            prev_t = timesteps[i+1] if i < len(timesteps) - 1 else -1
            
            # 建立 batch
            t_tensor = torch.full((sample_size,), t, device=self.device, dtype=torch.long)
            
            # 預測雜訊
            predicted_noise = self.model(x, t_tensor)
            
            # 使用 DDIM step 更新狀態
            x = self.scheduler.step(x, t, prev_t, predicted_noise)
            
        return x