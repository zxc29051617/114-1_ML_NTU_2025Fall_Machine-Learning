import torch
import matplotlib.pyplot as plt
import os
from src.diffusion import DDIMScheduler, DDIMPipeline, Denoiser

# 1. 設定環境
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("assets", exist_ok=True) # 確保 assets 資料夾存在

# 2. 載入模型 (用 DDPM 練好的權重)
model = Denoiser(hidden_dim=128).to(device)
ckpt_path = "checkpoints/ddpm.pth"

if not os.path.exists(ckpt_path):
    print("找不到權重檔！請確認路徑。")
    exit()

model.load_state_dict(torch.load(ckpt_path, map_location=device))
print("模型載入成功，準備生成 DDIM 圖片...")

# 3. 設定 DDIM Scheduler (50步, eta=0.0)
scheduler = DDIMScheduler(device=device, eta=0.0)
pipe = DDIMPipeline(model, scheduler, num_inference_steps=50, device=device)

# 4. 生成圖片
samples = pipe.sample(2500).cpu().numpy()

# 5. 畫圖並存檔
plt.figure(figsize=(5, 5))
plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title("DDIM Generation (Steps=50)")
plt.tight_layout()

# 存到 assets 資料夾
save_path = "assets/ddim_final.png"
plt.savefig(save_path)
print(f"✅ 成功！圖片已儲存為: {save_path}")