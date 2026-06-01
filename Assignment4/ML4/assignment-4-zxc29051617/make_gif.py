# 記得先跑 uv add imageio
import imageio.v2 as imageio
import os
import glob
import re

def try_int(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ 用來讓檔名依照數字排序 (epoch_1, epoch_2, ... epoch_10) """
    return [try_int(c) for c in re.split('([0-9]+)', s)]

def make_gif(folder_path="results/gan", output_name="gan_training.gif"):
    # 1. 找圖片
    images = glob.glob(os.path.join(folder_path, "*.png"))
    
    # 2. 排序 (確保是從第 1 epoch 到最後)
    images.sort(key=alphanum_key)
    
    if not images:
        print(f"找不到圖片！請確認 {folder_path} 資料夾裡面有 .png 檔案")
        return

    print(f"找到 {len(images)} 張圖片，正在製作 GIF...")
    
    # 3. 讀取並製作 GIF
    frames = []
    for filename in images:
        frames.append(imageio.imread(filename))
        
    # duration 是每張圖停留幾秒 (0.1秒 = 10fps)
    imageio.mimsave(output_name, frames, duration=100, loop=0)
    print(f"成功！GIF 已儲存為: {output_name}")

if __name__ == "__main__":
    make_gif(folder_path="results/gan", output_name="gan_evolution.gif")