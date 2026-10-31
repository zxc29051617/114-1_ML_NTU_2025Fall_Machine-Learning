import torch
import numpy as np
import os

# 1. 載入數據
print("正在載入 all_task_vectors.pt ...")
try:
    data_path = "all_task_vectors.pt" 
    layer_data_map = torch.load(data_path)
    print("資料載入成功！")
except FileNotFoundError:
    print("找不到 all_task_vectors.pt，請先執行 extract_task_vectors.py 產出檔案！")
    exit()

def run_svd_elect(tensor_stack, r):
    """
    對堆疊好的 Tensor 進行 Low-rank Elect
    tensor_stack: [N_tasks, ...] 任意形狀的堆疊參數
    r: int (Rank)
    """
    # 1. 獲取維度資訊
    N = tensor_stack.shape[0] # 任務數量 (e.g., 8)
    original_feature_shape = tensor_stack.shape[1:] # 扣除任務維度後的原始形狀 (e.g., [2304, 768] 或 [768])
    
    # 2. 轉成 2D 矩陣以進行 SVD
    # 無論原始形狀是 3D [8, 2304, 768] 還是 2D [8, 768]
    # view(N, -1) 都會把它變成 [8, 特徵總數]
    # 然後轉置變成 [特徵總數, 8]，每一行代表一個任務
    matrix_T = tensor_stack.view(N, -1).T.float() 
    
    # 3. 執行 SVD
    try:
        U, S, Vh = torch.linalg.svd(matrix_T, full_matrices=False)
    except:
        # 記憶體不足時切換到 CPU
        matrix_T = matrix_T.cpu()
        U, S, Vh = torch.linalg.svd(matrix_T, full_matrices=False)

    # 4. 取前 r 個特徵 (Low-rank)
    # Rank 不能超過任務數量 (N=8)
    r_actual = min(r, N) 
    
    U_r = U[:, :r_actual] # [Feature_Dim, r]
    
    # 投影 (Project)
    z_tasks = U_r.T @ matrix_T # [r, 8]
    
    # 5. 在低維空間進行 Elect (選最大)
    # 這裡的邏輯是：選出 8 個任務在主要特徵上的共識
    z_sum = z_tasks.sum(dim=1)
    z_sign = torch.sign(z_sum)
    z_max = torch.max(torch.abs(z_tasks), dim=1).values
    z_uni = z_sign * z_max # [r]
    
    # 6. 還原 (Back-project)
    tau_uni_flat = U_r @ z_uni # [Feature_Dim]
    
    # 7. 轉回原本形狀
    # 使用 *original_feature_shape 自動解包形狀
    tau_uni = tau_uni_flat.view(*original_feature_shape)
    
    return tau_uni

# ===========================
# 主實驗迴圈
# ===========================

candidates_r = [1, 2, 3, 4, 5, 6, 7, 8] 

print(f"開始測試不同的 r 值: {candidates_r}")

for r in candidates_r:
    print(f"\n--- Testing Rank r = {r} ---")
    
    total_similarity = 0
    layer_count = 0
    
    # 為了避免碰到奇怪的參數（有些 scalar 參數可能導致問題），
    # 我們這裡加一個過濾機制，只測 "weight" 類型的參數，通常比較準
    test_layers = [0, 5, 11] 
    
    for layer_id in test_layers:
        if layer_id not in layer_data_map: continue
        
        # 遍歷該層所有參數，只找一個代表性的 Weight 來測
        target_tensor = None
        for name, tensor in layer_data_map[layer_id].items():
            if 'weight' in name and tensor.dim() >= 2: # 只測維度>=2的權重矩陣
                target_tensor = tensor
                break
        
        if target_tensor is None:
            # 如果這層沒找到合適的，就隨便拿第一個，但要確保它是 tensor
            target_tensor = list(layer_data_map[layer_id].values())[0]

        # 跑 SVD
        tau_uni = run_svd_elect(target_tensor, r)
        
        # 計算相似度
        sim_sum = 0
        for i in range(target_tensor.shape[0]):
            original = target_tensor[i]
            v1 = tau_uni.flatten()
            v2 = original.flatten()
            
            # 避免除以 0
            norm1 = torch.norm(v1)
            norm2 = torch.norm(v2)
            if norm1 > 0 and norm2 > 0:
                cos_sim = torch.dot(v1, v2) / (norm1 * norm2)
                sim_sum += cos_sim.item()
            
        avg_sim = sim_sum / target_tensor.shape[0]
        total_similarity += avg_sim
        layer_count += 1
        print(f"  Layer {layer_id} Avg Cosine Similarity: {avg_sim:.4f}")

    if layer_count > 0:
        final_score = total_similarity / layer_count
        print(f"===> Rank {r} 的總平均相似度: {final_score:.4f}")