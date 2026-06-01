import torch
import numpy as np
import os
from src.diffusion import DDIMScheduler, DDIMPipeline, Denoiser, DDPMScheduler, DDPMPipeline
from src.metric import cal_2_wasserstein_dist, cal_energy_distance

# 1. 設定環境
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. 載入模型
model = Denoiser(hidden_dim=128).to(device)
ckpt_path = "checkpoints/ddpm.pth"

if not os.path.exists(ckpt_path):
    print("找不到模型權重！請確認你有先跑過 train.py -m ddpm")
    exit()

model.load_state_dict(torch.load(ckpt_path, map_location=device))
print("成功載入模型！")

# 3. 準備真實資料 (用來算分)
real_data = np.load("data/checkerboard.npy")
idx = np.random.choice(len(real_data), 2500, replace=False)
real_sample = real_data[idx]

print("\n" + "="*50)
print(f"{'Method':<10} | {'Steps':<5} | {'WD':<8} | {'ED':<8}")
print("="*50)

# === DDPM (基準線) ===
scheduler_ddpm = DDPMScheduler(device=device)
pipe_ddpm = DDPMPipeline(model, scheduler_ddpm, device=device)
samples = pipe_ddpm.sample(2500).cpu().numpy()
wd = cal_2_wasserstein_dist(real_sample, samples)
ed = cal_energy_distance(real_sample, samples)
print(f"{'DDPM':<10} | {1000:<5} | {wd:.4f}   | {ed:.4f}")

# === DDIM (各種步數) ===
scheduler_ddim = DDIMScheduler(device=device, eta=0.0)
steps_list = [1000, 500, 100, 50, 10, 1]

for steps in steps_list:
    pipe_ddim = DDIMPipeline(model, scheduler_ddim, num_inference_steps=steps, device=device)
    samples = pipe_ddim.sample(2500).cpu().numpy()
    
    wd = cal_2_wasserstein_dist(real_sample, samples)
    ed = cal_energy_distance(real_sample, samples)
    
    print(f"{'DDIM':<10} | {steps:<5} | {wd:.4f}   | {ed:.4f}")

print("="*50)