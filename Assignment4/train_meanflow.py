import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call, jvp # PyTorch 2.0+ 核心功能
import numpy as np
import os
import wandb
from tqdm import tqdm
from pathlib import Path

# 引用你原本寫好的 Positional Embedding，或是直接複製過來也可以
# 這裡假設你的 diffusion.py 在 src 資料夾下，或是同層級
try:
    from src.diffusion import SinusoidalPositionEmbeddings
except ImportError:
    # 萬一 import 失敗，這裡有一個備用的定義
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

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. MeanFlow 專用模型 (輸入 x, r, t) ===
class MeanFlowDenoiser(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # 處理 t 和 r 的 Embedding
        # 我們將 (t) 和 (t-r) 的 embedding 加起來作為時間條件
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        self.input_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU()
        )
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2) # 輸出 velocity u
        )

    def forward(self, x, r, t):
        # x: (B, 2)
        # r, t: (B,)
        
        # 1. Embed 時間
        # 根據論文 ablation，使用 t 和 t-r 效果最好
        t_emb = self.time_mlp(t)
        delta_emb = self.time_mlp(t - r)
        
        # 融合條件 (相加)
        time_cond = t_emb + delta_emb
        
        # 2. Embed 輸入
        x_emb = self.input_mlp(x)
        
        # 3. Concatenate
        h = torch.cat([x_emb, time_cond], dim=-1)
        
        return self.net(h)

# === 2. MeanFlow Loss (Algorithm 1) ===
def compute_meanflow_loss(model, x, device):
    batch_size = x.shape[0]
    
    # 1. Sample t, r ~ Uniform(0, 1)
    t = torch.rand(batch_size, device=device)
    r = torch.rand(batch_size, device=device)
    
    # 強制 r <= t (論文限制)
    mask = r > t
    r_new = torch.where(mask, t, r)
    t_new = torch.where(mask, r, t)
    r, t = r_new, t_new
    
    # 2. Sample noise e
    e = torch.randn_like(x)
    
    # 3. Interpolation z_t = (1-t)x + te
    t_expand = t.view(-1, 1)
    z = (1 - t_expand) * x + t_expand * e
    
    # 4. Instantaneous Velocity v = e - x
    v = e - x
    
    # === 5. 計算 JVP (Jacobian-Vector Product) ===
    # 目標: 計算 du/dt
    # 輸入變量 (z, r, t) 的切線向量 (tangents) 為 (v, 0, 1)
    
    # 準備 functional call
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    
    def func_model(z_in, r_in, t_in):
        return functional_call(model, (params, buffers), (z_in, r_in, t_in))

    tangent_z = v
    tangent_r = torch.zeros_like(r)
    tangent_t = torch.ones_like(t)
    
    # 執行 jvp
    # u_theta: 模型預測的平均速度
    # dudt: u 對時間的全導數
    u_theta, dudt = jvp(
        func_model, 
        (z, r, t), 
        (tangent_z, tangent_r, tangent_t)
    )
    
    # === 6. Loss Target ===
    # u_target = v - (t - r) * du/dt
    t_r_diff = (t - r).view(-1, 1)
    u_target = v - t_r_diff * dudt
    
    # 7. Loss (target 必須 detach)
    loss = torch.mean((u_theta - u_target.detach()) ** 2)
    
    return loss

# === 3. Training Loop ===
if __name__ == "__main__":
    device = get_device()
    
    # 讀取資料
    target_data = np.load(Path("data", "checkerboard.npy")).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(target_data))
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = MeanFlowDenoiser(hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # WandB 設定 (可選)
    # wandb.init(project="meanflow-checkerboard")
    
    epochs = 2000 # MeanFlow 訓練通常比 GAN 穩定，比 DDPM 快
    
    print("🚀 Start Training MeanFlow...")
    model.train()
    
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        total_loss = 0
        for (x_batch,) in dataloader:
            x_batch = x_batch.to(device)
            
            optimizer.zero_grad()
            loss = compute_meanflow_loss(model, x_batch, device)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        pbar.set_description(f"Loss: {avg_loss:.4f}")
        
        # wandb.log({"loss": avg_loss})
    
    # 存檔
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), Path("checkpoints", "meanflow.pth"))
    print("✅ Model saved to checkpoints/meanflow.pth")