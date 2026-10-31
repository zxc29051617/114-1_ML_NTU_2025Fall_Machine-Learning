import torch


class DiceScore:
    def __init__(self, smooth: float = 1e-5):
        self.smooth = smooth

    def __call__(self, pred: torch.Tensor, label: torch.Tensor, mean: bool = True) -> torch.Tensor:
        #TODO: Implement Dice Score calculation
        """
        計算 Dice Score (DSC)。

        :param pred: 模型的原始輸出 (logits)，形狀為 [N, 1, H, W]
        :param label: 真實的遮罩 (ground truth)，形狀為 [N, 1, H, W]
        :param mean: 是否回傳這批 (batch) 的平均 DSC。
        :return: 這批 (batch) 的 Dice Score。
        """
        # 1. 將模型的 logits 透過 Sigmoid 轉換為 [0, 1] 之間的機率
        pred_prob = torch.sigmoid(pred)
        
        # 2. [關鍵] 將機率值轉換為二元預測 (0 或 1)
        # 這是 Metric 和 Loss 最大的不同點
        pred_binary = (pred_prob > 0.5).float()

        # 為了計算每張影像的 DSC，我們只壓平 H 和 W 維度
        # [N, 1, H, W] -> [N, H*W]
        pred_binary = pred_binary.view(pred_binary.size(0), -1)
        label = label.view(label.size(0), -1)
        
        # 3. 計算每張影像的交集 (Intersection)
        # (dim=1 是沿著 H*W 維度加總)
        intersection = (pred_binary * label).sum(dim=1)
        
        # 4. 計算每張影像的總和
        total_sum = pred_binary.sum(dim=1) + label.sum(dim=1)
        
        # 5. 計算每張影像的 Dice Score
        # 公式 = (2 * |A ∩ B| + smooth) / (|A| + |B| + smooth)
        dice = (2. * intersection + self.smooth) / (total_sum + self.smooth)
        
        # 6. 根據 'mean' 參數回傳平均值或每張影像的分數
        return dice.mean() if mean else dice
