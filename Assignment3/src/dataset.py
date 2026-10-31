import glob
import os
from typing import Optional, Union, List, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
import random
from torchvision.transforms.functional import to_tensor, resize, hflip, vflip, rotate


class MedicalImageDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        has_label: bool = True,
        transform: Optional[callable] = None
    ):
        self.data_dir = data_dir
        self.data_list = glob.glob(f"{data_dir}/image/*.tif")
        self.has_label = has_label
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        if self.has_label:
            data =  {
                "filename": os.path.basename(self.data_list[idx]),
                "image": self.data_list[idx],
                "label": self.data_list[idx].replace("image", "label"),
            }
        else:
            data = {
                "filename": os.path.basename(self.data_list[idx]),
                "image": self.data_list[idx],
            }

        if self.transform:
            return self.transform(data)

        return data


class BaseTransform:
    def __init__(self, keys: List[str], *args, **kwargs):
        self.keys = keys
        self._parse_var(*args, **kwargs)

    def _parse_var(self, *args, **kwargs):
        pass

    def _process(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        raise NotImplementedError

    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        for key in self.keys:
            if key in data:
                data[key] = self._process(data[key], *args, **kwargs)
            else:
                raise KeyError(f"{key} is not a key in data.")
        return data


class LoadImaged(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(LoadImaged, self).__init__(keys, **kwargs)

    def _process(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        return Image.open(data)


class ToTensord(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(ToTensord, self).__init__(keys, **kwargs)

    def _process(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        return to_tensor(data)


class ResizeImaged(BaseTransform):
    def __init__(self, keys, size, **kwargs):
        super(ResizeImaged, self).__init__(keys, **kwargs)
        self.size = size

    def _process(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        return resize(data, self.size)

class RandomHorizontalFlip(BaseTransform):
    """
    隨機水平翻轉
    """
    def __init__(self, keys, p=0.5, **kwargs):
        super(RandomHorizontalFlip, self).__init__(keys, **kwargs)
        self.p = p

    def _process(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # data 必須是 Tensor (C, H, W)
        if random.random() < self.p:
            return hflip(data)
        return data


class RandomVerticalFlip(BaseTransform):
    """
    隨機垂直翻轉
    """
    def __init__(self, keys, p=0.5, **kwargs):
        super(RandomVerticalFlip, self).__init__(keys, **kwargs)
        self.p = p

    def _process(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # data 必須是 Tensor (C, H, W)
        if random.random() < self.p:
            return vflip(data)
        return data


class RandomRotation(BaseTransform):
    """
    隨機旋轉
    """
    def __init__(self, keys, degrees=15, **kwargs):
        super(RandomRotation, self).__init__(keys, **kwargs)
        self.degrees = degrees

    def _process(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # data 必須是 Tensor (C, H, W)
        # 隨機選擇一個角度
        angle = random.uniform(-self.degrees, self.degrees)
        return rotate(data, angle)
    

class DuplicateImaged(BaseTransform):
    """
    複製指定 key 的張量到新的 key (e.g., 'image' -> 'label')，用於 SSL 預訓練。
    
    注意：此 Transform 必須在 ToTensord 之後執行。
    """
    def __init__(self, keys: List[str], new_key: str):
        # keys 應該是 ["image"]
        super(DuplicateImaged, self).__init__(keys)
        self.new_key = new_key

    def __call__(self, data: Dict[str, any]) -> Dict[str, any]:
        for key in self.keys:
            if key in data:
                # 複製張量，並儲存到新的 key (例如 'label')
                # clone().detach() 確保它是一個獨立的、未被修改的目標
                data[self.new_key] = data[key].clone().detach() 
            else:
                raise KeyError(f"{key} is not a key in data for duplication.")
        return data


class RandomPatchMasking(BaseTransform):
    """
    隨機遮蔽圖像張量的圖像塊，模擬 Masked Autoencoder 的輸入。
    此 Transform 必須在 ToTensord 之後執行。
    """
    def _parse_var(self, patch_size: int, mask_ratio: float):
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def __init__(self, keys: List[str], patch_size: int, mask_ratio: float, **kwargs):
        super(RandomPatchMasking, self).__init__(keys, patch_size=patch_size, mask_ratio=mask_ratio)

    def _process(self, img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 假設 img 的形狀是 (C, H, W)
        C, H, W = img.shape
        
        # 計算 Patch 數量
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        num_patches = h_patches * w_patches
        
        # 計算需要遮蔽的 Patch 數量
        num_mask = int(self.mask_ratio * num_patches)
        
        if num_mask == 0:
            return img

        # 隨機選擇要遮蔽的 Patch 索引
        mask_indices = torch.randperm(num_patches)[:num_mask]
        
        # 將遮蔽索引轉換回 (h_patch_index, w_patch_index)
        mask_h = mask_indices // w_patches
        mask_w = mask_indices % w_patches
        
        # 應用遮蔽：將這些 Patch 區域的像素值設為 0
        img_masked = img.clone() # 在複製的張量上操作，避免修改 target
        
        for h_idx, w_idx in zip(mask_h, mask_w):
            start_h = h_idx * self.patch_size
            end_h = start_h + self.patch_size
            start_w = w_idx * self.patch_size
            end_w = start_w + self.patch_size
            
            # 將該區域設為 0.0 (黑塊)
            img_masked[:, start_h:end_h, start_w:end_w] = 0.0 
            
        return img_masked

    # 處理字典，而不是單一張量
    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        for key in self.keys:
            if key in data:
                # 這裡使用 _process 來執行遮蔽，但必須確保 data[key] 是 Tensor
                data[key] = self._process(data[key], *args, **kwargs)
            else:
                raise KeyError(f"{key} is not a key in data.")
        return data