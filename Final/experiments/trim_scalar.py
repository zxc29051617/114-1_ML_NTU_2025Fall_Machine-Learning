import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

DEFAULT_TRIM_RATIO = 0.1
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Task Clustering
# =============================================================================
def compute_task_similarity(task_vectors: list, 
                            layer_name: Optional[str] = None,
                            device: Optional[str] = None) -> torch.Tensor:
    """
    計算任務向量間的餘弦相似度矩陣
    
    使用指定層的參數作為任務的代表向量，計算兩兩間的餘弦相似度。
    餘弦相似度範圍為[-1, 1]，值越大表示越相似。
    
    Args:
        task_vectors: TaskVector 物件列表
        layer_name: 用於計算相似度的層名稱，None 則使用第一層
        device: 運算裝置，None 則自動選擇
        
    Returns:
        similarity_matrix: [num_tasks, num_tasks] 的相似度矩陣
        
    Raises:
        ValueError:如果task_vectors為空或格式錯誤
        
    Example:
        >>> from task_vectors import TaskVector
        >>> tvs = [TaskVector(pre, ft1), TaskVector(pre, ft2)]
        >>> sim_matrix = compute_task_similarity(tvs)
        >>> print(sim_matrix)
        tensor([[1.0000, 0.8523],
                [0.8523, 1.0000]])
    """
    # 參數檢查
    if not task_vectors:
        raise ValueError("task_vectors不能為空")
    
    if not hasattr(task_vectors[0], 'vector'):
        raise ValueError("task_vectors必須是TaskVector物件列表")
    
    # 設定裝置
    if device is None:
        device = DEFAULT_DEVICE
    
    # 選擇代表層（預設用第一層）
    if layer_name is None:
        layer_name = list(task_vectors[0].vector.keys())[0]
    
    if layer_name not in task_vectors[0].vector:
        raise ValueError(f"找不到層: {layer_name}")
    
    # 提取並攤平每個任務的向量
    vectors = []
    for i, tv in enumerate(task_vectors):
        try:
            vec = tv.vector[layer_name].view(-1).float().to(device)
            vectors.append(vec)
        except Exception as e:
            raise ValueError(f"處理第 {i} 個 task vector 時出錯: {e}")
    
    # 堆疊成矩陣 [num_tasks, vector_dim]
    vectors_matrix = torch.stack(vectors)
    
    # L2正規化（使向量長度為1）
    vectors_normalized = F.normalize(vectors_matrix, p=2, dim=1)
    
    # 計算餘弦相似度（正規化向量的內積）
    similarity_matrix = torch.mm(vectors_normalized, vectors_normalized.t())
    
    return similarity_matrix


def find_most_similar_pair(similarity_matrix: torch.Tensor) -> Tuple[int, int, float]:
    """
    從相似度矩陣中找出最相似的任務對
    
    Args:
        similarity_matrix: [num_tasks, num_tasks]的相似度矩陣
        
    Returns:
        task_i:第一個任務的索引
        task_j:第二個任務的索引  
        similarity:這兩個任務的相似度值
        
    Raises:
        ValueError:如果矩陣太小（少於2個task）
        
    Example:
        >>> i, j, sim = find_most_similar_pair(sim_matrix)
        >>> print(f"Task {i} and Task {j}: similarity = {sim:.4f}")
    """
    num_tasks = similarity_matrix.shape[0]
    
    if num_tasks < 2:
        raise ValueError("至少需要 2 個任務才能找相似對")
    
    # 複製矩陣，避免修改原始資料
    sim = similarity_matrix.clone()
    
    # 將對角線設為-∞（排除自己和自己比較）
    sim.fill_diagonal_(-float('inf'))
    
    # 找最大值的位置
    max_idx = torch.argmax(sim).item()
    
    task_i = max_idx // num_tasks
    task_j = max_idx % num_tasks
    similarity = similarity_matrix[task_i, task_j].item()
    
    return task_i, task_j, similarity


def cluster_tasks(task_vectors: list, 
                  dataset_names: Optional[List[str]] = None,
                  layer_name: Optional[str] = None,
                  device: Optional[str] = None,
                  verbose: bool = True) -> Tuple[int, int]:
    """
    對任務進行分群，找出最相似的一對
    
    Args:
        task_vectors:TaskVector物件列表
        dataset_names:資料集名稱列表（用於顯示，可選）
        layer_name:用於計算相似度的層名稱，None 則使用第一層
        device:運算裝置，None 則自動選擇
        verbose:是否印出過程訊息
        
    Returns:
        (task_i, task_j):最相似的兩個任務索引
        
    Example:
        >>> # 基本用法
        >>> i, j = cluster_tasks(task_vectors, dataset_names)
        
        >>> # 不顯示訊息
        >>> i, j = cluster_tasks(task_vectors, verbose=False)
        
        >>> # 使用特定層計算相似度
        >>> i, j = cluster_tasks(task_vectors, layer_name='visual.proj')
    """
    if verbose:
        print("[Clustering] 計算任務相似度...")
    
    # 計算相似度矩陣
    sim_matrix = compute_task_similarity(task_vectors, layer_name, device)
    
    # 找最相似的一對
    task_i, task_j, similarity = find_most_similar_pair(sim_matrix)
    
    if verbose:
        if dataset_names:
            name_i = dataset_names[task_i]
            name_j = dataset_names[task_j]
            print(f"[Clustering] 最相似的任務對: {name_i} & {name_j}")
        else:
            print(f"[Clustering] 最相似的任務對: Task {task_i} & Task {task_j}")
        print(f"[Clustering] 相似度:{similarity:.4f}")
    
    return task_i, task_j


def get_similarity_matrix(task_vectors: list,
                          dataset_names: Optional[List[str]] = None,
                          device: Optional[str] = None) -> Tuple[torch.Tensor, Optional[List[str]]]:
    """
    取得完整的相似度矩陣（方便分析或視覺化）
    
    Args:
        task_vectors: TaskVector 物件列表
        dataset_names: 資料集名稱列表（可選）
        device: 運算裝置
        
    Returns:
        similarity_matrix: 相似度矩陣
        dataset_names: 資料集名稱（如果有提供）
        
    Example:
        >>> sim_matrix, names = get_similarity_matrix(tvs, ds_names)
        >>> # 可以用來畫 heatmap
        >>> import seaborn as sns
        >>> sns.heatmap(sim_matrix.cpu(), xticklabels=names, yticklabels=names)
    """
    sim_matrix = compute_task_similarity(task_vectors, device=device)
    return sim_matrix, dataset_names



# =============================================================================
# Trimmed Mean Rescaling
# =============================================================================

def trimmed_mean(tensor: torch.Tensor, 
                 trim_ratio: float = DEFAULT_TRIM_RATIO) -> torch.Tensor:
    """
    計算Trimmed Mean
    
    
    Args:
        tensor: 輸入 tensor（任意 shape）
        trim_ratio: 裁切比例，上下各去掉多少（預設 0.1 = 10%）
        
    Returns:
        result: 純量 tensor，表示 trimmed mean 值
        
    Raises:
        ValueError: 如果 trim_ratio 不在有效範圍內
        
    Example:
        >>> t = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])  # 100 是離群值
        >>> trimmed_mean(t, trim_ratio=0.2)
        tensor(3.)  # 去掉 1 和 100，計算 2, 3, 4 的平均
        
    Note:
        - 使用絕對值計算，因為我們關心的是參數的「能量大小」
        - 排序時移到 CPU 避免某些 GPU 的相容性問題
    """
    # 參數檢查
    if not 0 <= trim_ratio < 0.5:
        raise ValueError(f"trim_ratio 必須在 [0, 0.5) 之間，但收到 {trim_ratio}")
    
    # 取絕對值並攤平成一維
    flat = torch.abs(tensor).view(-1).float()
    
    # 處理空 tensor
    if flat.numel() == 0:
        return torch.tensor(0.0, device=tensor.device)
    
    # 排序（移到 CPU 避免某些 GPU 相容性問題）
    if flat.device.type == 'cuda':
        sorted_vals = torch.sort(flat.cpu())[0]
    else:
        sorted_vals = torch.sort(flat)[0]
    
    # 計算裁切的索引範圍
    n = flat.numel()
    lower_idx = int(n * trim_ratio)        # 下界：去掉最小的部分
    upper_idx = int(n * (1 - trim_ratio))  # 上界：去掉最大的部分
    
    # 計算裁切後的平均值
    if lower_idx < upper_idx:
        trimmed = sorted_vals[lower_idx:upper_idx]
        result = torch.mean(trimmed)
    else:
        # 如果裁切後沒有元素，就用全部的平均
        result = torch.mean(flat)
    
    return result.to(tensor.device)


def compute_rescaling_masks(task_vectors: list,
                            unified_vector: Dict[str, torch.Tensor],
                            device: Optional[str] = None) -> Dict[str, List[torch.Tensor]]:
    """
    計算每個任務在每層的Mask
    
    Mask標記原始Task Vector中與統一向量方向一致的位置。
    只有方向一致的位置才會被保留用於重建。
    
    Args:
        task_vectors:原始 TaskVector 列表
        unified_vector:合併後的統一向量 {layer_name: tensor}
        device:運算裝置
        
    Returns:
        masks:{layer_name: [mask_task0, mask_task1, ...]}
               每個 mask 是 bool tensor，True 表示方向一致
    """
    if device is None:
        device = DEFAULT_DEVICE
        
    masks = {}
    
    for layer_name, unified_tensor in unified_vector.items():
        masks[layer_name] = []
        
        # 統一向量的方向（+1 或 -1）
        unified_sign = torch.sign(unified_tensor)
        
        for tv in task_vectors:
            if layer_name not in tv.vector:
                raise ValueError(f"Task vector 中找不到層: {layer_name}")
                
            original = tv.vector[layer_name].to(device)
            
            # Mask:原始向量與統一向量方向一致的位置
            mask = (original * unified_sign) > 0
            masks[layer_name].append(mask)
    
    return masks


def compute_rescalers(task_vectors: list,
                      unified_vector: Dict[str, torch.Tensor],
                      masks: Dict[str, List[torch.Tensor]],
                      trim_ratio: float = DEFAULT_TRIM_RATIO,
                      device: Optional[str] = None) -> Dict[str, List[torch.Tensor]]:
    """
    計算每個任務在每層的縮放係數
    
    rescaler = trimmed_mean(original) / trimmed_mean(unified * mask)
    
    Args:
        task_vectors: 原始 TaskVector 列表
        unified_vector: 合併後的統一向量
        masks: 每個任務的 masks
        trim_ratio: Trimmed mean 的裁切比例
        
    Returns:
        rescalers: {layer_name: [rescaler_task0, rescaler_task1, ...]}
    """


    if device is None:
        device = DEFAULT_DEVICE
        
    rescalers = {}
    
    for layer_name, unified_tensor in unified_vector.items():
        rescalers[layer_name] = []
        
        for i, tv in enumerate(task_vectors):
            original = tv.vector[layer_name].to(device)
            mask = masks[layer_name][i]
            
            # 分子：原始Task Vector的trimmed mean
            numerator = trimmed_mean(original, trim_ratio)
            
            # 分母：統一向量mask的trimmed mean
            masked_unified = unified_tensor * mask
            denominator = trimmed_mean(masked_unified, trim_ratio)
            
            # 計算縮放係數（避免除以零）
            if denominator.item() != 0:
                rescaler = numerator / denominator
            else:
                rescaler = torch.tensor(0.0, device=device)
            
            rescalers[layer_name].append(rescaler)
    
    return rescalers


def compute_layer_rescaling(task_vectors: list,
                            unified_vector: Dict[str, torch.Tensor],
                            trim_ratio: float = DEFAULT_TRIM_RATIO,
                            device: Optional[str] = None,
                            verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    Layer-wise Trimmed Mean Rescaling 的主要入口
    
    整合 mask 計算和 rescaler 計算。
    
    Args:
        task_vectors: 原始 TaskVector 列表
        unified_vector: 合併後的統一向量 {layer_name: tensor}
        trim_ratio: 預設 0.1
        
    Returns:
        masks: {layer_name: [mask_task0, mask_task1, ...]}
        rescalers: {layer_name: [rescaler_task0, rescaler_task1, ...]}
        
    Example:
        >>> masks, rescalers = compute_layer_rescaling(
        ...     task_vectors, 
        ...     unified_vector, 
        ...     trim_ratio=0.1
        ... )
    """
    if device is None:
        device = DEFAULT_DEVICE
    
    if verbose:
        print(f"[Rescaling] 計算 Layer-wise Rescaling (trim_ratio={trim_ratio})...")
    
    # Step 1: 計算 masks
    masks = compute_rescaling_masks(task_vectors, unified_vector, device)
    
    # Step 2: 計算 rescalers
    rescalers = compute_rescalers(
        task_vectors, unified_vector, masks, trim_ratio, device
    )
    
    if verbose:
        num_layers = len(unified_vector)
        num_tasks = len(task_vectors)
        print(f"[Rescaling] 完成: {num_layers} 層 × {num_tasks} 任務")
    
    return masks, rescalers



# =============================================================================
# 輔助函數
# =============================================================================

def reconstruct_task_vector(unified_vector: Dict[str, torch.Tensor],
                            masks: Dict[str, List[torch.Tensor]],
                            rescalers: Dict[str, List[torch.Tensor]],
                            task_idx: int) -> Dict[str, torch.Tensor]:
    """
    重建特定任務的Task Vector
    
    reconstructed[layer] = unified[layer] * mask[layer] * rescaler[layer]
    
    Args:
        unified_vector:統一向量 {layer_name: tensor}
        masks:所有任務的 masks {layer_name: [mask0, mask1, ...]}
        rescalers:所有任務的 rescalers {layer_name: [rescaler0, ...]}
        task_idx:要重建的任務索引
        
    Returns:
        reconstructed:重建的 task vector {layer_name: tensor}
        
    Example:
        >>> reconstructed = reconstruct_task_vector(unified, masks, rescalers, task_idx=0)
        >>> # 用這個reconstructed來評估第0個任務
    """
    # 檢查task_idx是否有效
    sample_layer = list(masks.keys())[0]
    num_tasks = len(masks[sample_layer])
    
    if not 0 <= task_idx < num_tasks:
        raise ValueError(f"task_idx 必須在 [0, {num_tasks}) 之間，但收到 {task_idx}")
    
    reconstructed = {}
    
    for layer_name in unified_vector:
        unified = unified_vector[layer_name]
        mask = masks[layer_name][task_idx]
        rescaler = rescalers[layer_name][task_idx]
        
        reconstructed[layer_name] = unified * mask * rescaler
    
    return reconstructed


def reconstruct_all_task_vectors(unified_vector: Dict[str, torch.Tensor],
                                  masks: Dict[str, List[torch.Tensor]],
                                  rescalers: Dict[str, List[torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """
    重建所有任務的Task Vector
    
    Args:
        unified_vector:統一向量
        masks:所有任務的masks
        rescalers:所有任務的rescalers
        
    Returns:
        reconstructed_list: [reconstructed_task0, reconstructed_task1, ...]
        
    Example:
        >>> all_reconstructed = reconstruct_all_task_vectors(unified, masks, rescalers)
        >>> for i, recon in enumerate(all_reconstructed):
        ...     model = apply_task_vector(recon, pretrained_path)
        ...     evaluate(model, datasets[i])
    """
    sample_layer = list(masks.keys())[0]
    num_tasks = len(masks[sample_layer])
    
    reconstructed_list = []
    for task_idx in range(num_tasks):
        recon = reconstruct_task_vector(unified_vector, masks, rescalers, task_idx)
        reconstructed_list.append(recon)
    
    return reconstructed_list



# =============================================================================
# 測試函數（確保模組正常運作）
# =============================================================================

def _test_trimmed_mean():
    """測試 trimmed_mean 函數"""
    print("測試 trimmed_mean...")
    
    # 測試1:基本功能
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = trimmed_mean(t, trim_ratio=0.2)
    expected = 3.0  # 去掉 1 和 5，計算 2, 3, 4 的平均
    assert abs(result.item() - expected) < 0.01, f"預期 {expected}，但得到 {result.item()}"
    
    # 測試2:有離群值
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
    result = trimmed_mean(t, trim_ratio=0.2)
    expected = 3.0  # 去掉 1 和 100
    assert abs(result.item() - expected) < 0.01, f"預期 {expected}，但得到 {result.item()}"
    
    # 測試3:空tensor
    t = torch.tensor([])
    result = trimmed_mean(t)
    assert result.item() == 0.0, "空 tensor 應該返回 0"
    
    print("trimmed_mean 測試通過")


def _test_module():
    """測試整個模組（需要有 task_vectors）"""
    print("\n" + "=" * 50)
    print("EMR Components 模組測試")
    print("=" * 50)
    
    _test_trimmed_mean()
    
    print("\n所有測試通過")
    print("=" * 50)


if __name__ == "__main__":
    _test_module()
