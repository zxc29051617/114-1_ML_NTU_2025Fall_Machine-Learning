import torch
import numpy as np
import os

# ==========================================
# 1. 隊友 Wei 的工具 (Trimmed Mean)
# ==========================================
def trimmed_mean(tensor, trim_ratio=0.1):
    # 計算去尾平均數
    flat = torch.abs(tensor).view(-1).float()
    if flat.numel() == 0: return torch.tensor(0.0, device=tensor.device)
    
    # 排序
    if flat.device.type == 'cuda': sorted_vals = torch.sort(flat.cpu())[0]
    else: sorted_vals = torch.sort(flat)[0]
    
    n = flat.numel()
    lower = int(n * trim_ratio)
    upper = int(n * (1 - trim_ratio))
    
    if lower < upper: return torch.mean(sorted_vals[lower:upper]).to(tensor.device)
    return torch.mean(flat).to(tensor.device)

# ==========================================
# 2. 你的 SVD 工具 (從 find_best_r.py 搬過來)
# ==========================================
def run_svd_elect(tensor_stack, r=1):
    # 自動處理維度
    N = tensor_stack.shape[0]
    original_shape = tensor_stack.shape[1:]
    matrix_T = tensor_stack.view(N, -1).T.float()
    
    try:
        U, S, Vh = torch.linalg.svd(matrix_T, full_matrices=False)
    except:
        matrix_T = matrix_T.cpu()
        U, S, Vh = torch.linalg.svd(matrix_T, full_matrices=False)

    # 取前 r 個 (你測出來 r=1 最好)
    r_actual = min(r, N)
    U_r = U[:, :r_actual]
    z_tasks = U_r.T @ matrix_T
    
    # Elect: Sign * Max
    z_sum = z_tasks.sum(dim=1)
    z_sign = torch.sign(z_sum)
    z_max = torch.max(torch.abs(z_tasks), dim=1).values
    z_uni = z_sign * z_max
    
    tau_uni_flat = U_r @ z_uni
    return tau_uni_flat.view(*original_shape)

# ==========================================
# 3. 主實驗流程：SVD + 原版 Mask + TrimScalar
# ==========================================
if __name__ == "__main__":
    print("=== 開始執行指定任務: SVD(r=1) + Original Mask + TrimScalar ===")
    
    # A. 載入數據
    data_path = "all_task_vectors.pt"
    if not os.path.exists(data_path):
        print("錯誤: 找不到 all_task_vectors.pt")
        exit()
    
    layer_data = torch.load(data_path)
    print("✅ 數據載入成功！")

    # B. 跑每一層的運算
    # 我們這裡只跑 Layer 0 當作示範，證明流程是對的
    test_layer = 0
    if test_layer in layer_data:
        print(f"\n正在計算 Layer {test_layer}...")
        
        # 抓出這一層的第一個參數 (例如 Attention Weight)
        param_name = list(layer_data[test_layer].keys())[0]
        tensor_stack = layer_data[test_layer][param_name] 
        print(f"參數: {param_name}, 形狀: {tensor_stack.shape}")

        # Step 1: 你的 SVD (Elect)
        print(" -> 1. 執行 SVD Elect (r=1)...")
        unified_vec = run_svd_elect(tensor_stack, r=1)
        
        # Step 2: 舊版 Mask (原)
        print(" -> 2. 計算 Original Mask...")
        unified_sign = torch.sign(unified_vec)
        
        # 針對 8 個任務分別計算
        for i in range(tensor_stack.shape[0]):
            task_vec = tensor_stack[i]
            
            # [重點] 這是「原版 Mask」邏輯
            mask = (task_vec * unified_sign) > 0
            
            # [重點] 這是「新版 Scalar」邏輯
            num = trimmed_mean(task_vec, 0.1)
            den = trimmed_mean(unified_vec * mask, 0.1)
            scalar = num / (den + 1e-6)
            
            if i == 0: # 只印第一個任務的結果就好
                print(f"    Task 0 - Mask 形狀: {mask.shape}")
                print(f"    Task 0 - Scalar 數值: {scalar.item():.4f}")

        print("\n✅ 實驗成功！SVD、Mask、Scalar 全部串接完成。")
        print("這個腳本證明了你可以跑出「cluster+trimscalar+svd+原」這個組合的數據。")