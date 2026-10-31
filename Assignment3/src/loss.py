#TODO: Implement loss functions
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    實作 Dice Loss，專為影像分割（尤其是類別不平衡時）設計。
    我們的 Dice Loss 是 1 - Dice Coefficient。
    """
    def __init__(self, smooth: float = 1e-6):
        """
        :param smooth: 一個微小的 "平滑" 常數，用來避免分母為 0 的情況。
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: 模型的原始輸出 (logits)，形狀為 [N, 1, H, W]
        :param targets: 真實的遮罩 (ground truth)，形狀為 [N, 1, H, W]
        :return: 該批次的 Dice Loss
        """
        
        # 1. 將模型的 logits 透過 Sigmoid 轉換為 [0, 1] 之間的機率
        inputs_prob = torch.sigmoid(inputs)
        
        # 2. 將 4D 張量 (N, C, H, W) 扁平化 (flatten) 成 1D 向量
        # 這樣我們可以輕易地計算總和 (sum)
        inputs_prob = inputs_prob.view(-1)
        targets = targets.view(-1)
        
        # 3. 計算交集 (Intersection)
        intersection = (inputs_prob * targets).sum()
        
        # 4. 計算 Dice Coefficient (DICE)
        # 公式 = (2 * |A ∩ B| + smooth) / (|A| + |B| + smooth)
        total_sum = inputs_prob.sum() + targets.sum()
        dice = (2. * intersection + self.smooth) / (total_sum + self.smooth)
        
        # 5. 返回 Dice Loss
        return 1 - dice

class BCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss 的一個簡單包裝。
    我們使用 BCEWithLogitsLoss 來獲得更好的數值穩定性。
    """
    def __init__(self):
        super(BCELoss, self).__init__()
        # Lecture 2, Page 7 提到了 Logistic Loss 是 Cross Entropy + Sigmoid。
        # nn.BCEWithLogitsLoss 會自動幫我們完成這兩步，
        # 它比手動做 Sigmoid 再接 nn.BCELoss 更穩定。
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: 模型的原始輸出 (logits)，形狀為 [N, 1, H, W]
        :param targets: 真實的遮罩 (ground truth)，形狀為 [N, 1, H, W]
        :return: 該批次的 BCE Loss
        """
        return self.loss_fn(inputs, targets)
    

class MSELoss(nn.Module):
    """
    Mean Squared Error Loss (L2 Loss)，常用於回歸和圖像重建任務。
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        # PyTorch 內建的 MSELoss，預設 reduction='mean'，所以它本來就會算出平均值
        self.loss_fn = nn.MSELoss()

    # 加入 **kwargs 來接收 'mean' 或其他 Trainer 傳入的參數
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        :param inputs: 模型的輸出，形狀為 [N, C, H, W]
        :param targets: 真實的目標圖像，形狀為 [N, C, H, W]
        :param kwargs: 接收像是 'mean' 這樣的額外參數，防止報錯
        :return: 該批次的 MSE Loss
        """
        # 直接呼叫 PyTorch 的 MSELoss，它會忽略 kwargs
        return self.loss_fn(inputs, targets)