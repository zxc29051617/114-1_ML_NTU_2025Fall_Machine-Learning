import torch
import numpy as np
import os
from src.diffusion import DDIMScheduler, DDIMPipeline, Denoiser
from src.metric import cal_2_wasserstein_dist, cal_energy_distance

# 設定
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Denoiser(hidden_dim=128).to(device)
ckpt_path = "checkpoints/ddpm.pth"

if not os.path.exists(ckpt_path):
    print("找不到模型權重！")
    exit()

model.load_state_dict(torch.load(ckpt_path, map_location=device))

# 真實資料
real_data = np.load("data/checkerboard.npy")
idx = np.random.choice(len(real_data), 2500, replace=False)
real_sample = real_data[idx]

# 測試不同的 eta (固定步數為 50)
etas = [0.0, 0.25, 0.5, 0.75, 1.0]
steps = 50 

print(f"{'Eta':<5} | {'WD':<8} | {'ED':<8}")
print("-" * 30)

for eta in etas:
    # 這裡會用到你寫好的 eta 參數
    scheduler = DDIMScheduler(device=device, eta=eta)
    pipe = DDIMPipeline(model, scheduler, num_inference_steps=steps, device=device)
    samples = pipe.sample(2500).cpu().numpy()
    
    wd = cal_2_wasserstein_dist(real_sample, samples)
    ed = cal_energy_distance(real_sample, samples)
    print(f"{eta:<5} | {wd:.4f}   | {ed:.4f}")