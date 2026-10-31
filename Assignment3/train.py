from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# ❗️ 移除 torchvision.Compose，它不支援 dict-based transform
# from torchvision.transforms import Compose

from src.dataset import (
    MedicalImageDataset, 
    LoadImaged, 
    ToTensord, 
    ResizeImaged,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    RandomPatchMasking, 
    DuplicateImaged
)
from src.metric import DiceScore
from src.model import SimpleCNNUNet
from src.model_p2 import TransUNet
from src.loss import DiceLoss, BCELoss, MSELoss 
from src.trainer import Trainer
from src.utils import set_random_seed

from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau


# =======================================================
# ⭐ NEW：自製 Compose，支援 dict transforms
# =======================================================
class ComposeDict:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


# =======================================================
# Registries
# =======================================================
MODEL_REGISTRY = {
    "SimpleCNNUNet": SimpleCNNUNet,
    "TransUNet": TransUNet,
}

LOSS_REGISTRY = {
    "DiceLoss": DiceLoss,
    "BCELoss": BCELoss,
    "MSELoss": MSELoss, 
}

OPTIMIZER_REGISTRY = {
    "Adam": Adam,
    "AdamW": AdamW,
    "SGD": SGD,
}

SCHEDULER_REGISTRY = {
    "CosineAnnealingLR": CosineAnnealingLR,
    "StepLR": StepLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
}

METRIC_REGISTRY = {
    "DiceScore": DiceScore,
    "MSELoss": MSELoss,
}


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='problem1_dice') 
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.config}.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ===================================================================
    # 判斷訓練模式：Segmentation 或 SSL Pre-train
    # ===================================================================
    is_ssl_pretrain = "RandomPatchMasking" in config.transform
    
    # 基礎載入 keys
    load_keys = ["image"] if is_ssl_pretrain else ["image", "label"]

    # ===================================================================
    # 1. 訓練資料轉換
    # ===================================================================
    train_transforms_list = []
    
    # --- 基礎載入/Resize ---
    train_transforms_list.extend([
        LoadImaged(keys=load_keys),
        ToTensord(keys=load_keys),
        ResizeImaged(keys=load_keys, **config.transform.resized_imaged),
    ])

    if is_ssl_pretrain:
        # SSL：複製 label（重建 target）
        train_transforms_list.append(
            DuplicateImaged(keys=["image"], new_key="label")
        )
        # SSL：遮蔽 image
        train_transforms_list.append(
            RandomPatchMasking(keys=["image"], **config.transform.RandomPatchMasking)
        )
    else:
        # Seg augmentation
        if "RandomHorizontalFlip" in config.transform:
            train_transforms_list.append(
                RandomHorizontalFlip(keys=["image", "label"], **config.transform.RandomHorizontalFlip)
            )
        if "RandomVerticalFlip" in config.transform:
            train_transforms_list.append(
                RandomVerticalFlip(keys=["image", "label"], **config.transform.RandomVerticalFlip)
            )
        if "RandomRotation" in config.transform:
            train_transforms_list.append(
                RandomRotation(keys=["image", "label"], **config.transform.RandomRotation)
            )

    # ⭐ 使用 ComposeDict，而非 torchvision.Compose
    train_transform = ComposeDict(train_transforms_list)
    
    # ===================================================================
    # 2. 驗證資料轉換
    # ===================================================================
    valid_transform_list = [
        LoadImaged(keys=load_keys),
        ToTensord(keys=load_keys),
        ResizeImaged(keys=load_keys, **config.transform.resized_imaged),
    ]

    if is_ssl_pretrain:
        # ⭐ SSL validation 不 masking，只複製 label
        valid_transform_list.append(
            DuplicateImaged(keys=["image"], new_key="label")
        )

    valid_transform = ComposeDict(valid_transform_list)

    # ===================================================================
    # 3. DataLoader Setup
    # ===================================================================
    train_dataset = MedicalImageDataset(**config.train_dataset, transform=train_transform)
    train_loader = DataLoader(train_dataset, **config.train_loader)

    valid_dataset = MedicalImageDataset(**config.valid_dataset, transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, **config.valid_loader)

    # ===================================================================
    # 4. Model Setup
    # ===================================================================
    model_name = config.model.name
    model_args = config.model.args
    model = MODEL_REGISTRY[model_name](**model_args)

    # ===================================================================
    # ⭐ Fine-tune: 載入 SSL weights + 正確凍結策略
    # ===================================================================
    if "pretrain_path" in config.trainer and config.trainer.pretrain_path:
        print(f"🔄 Loading SSL Pre-trained weights from: {config.trainer.pretrain_path}")
        
        checkpoint = torch.load(config.trainer.pretrain_path, map_location=device)
        model_state_dict = model.state_dict()

        # 過濾 matched layers
        pretrained_dict = {
            k: v for k, v in checkpoint.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }

        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict)
        print(f"✅ Loaded {len(pretrained_dict)} pretrained layers.")

        # ⭐ 正確凍結策略（只 freeze CNN encoder + patch_embed）
        print("❄️ Freezing CNN encoder + patch_embed for fine-tuning...")
        for name, param in model.named_parameters():

            # freeze CNN encoder
            if name.startswith(("inc", "down", "conv")):
                param.requires_grad = False

            # freeze patch_embed
            elif "patch_embed" in name:
                param.requires_grad = False

            # transformer blocks & decoder remain trainable
            else:
                param.requires_grad = True

    # ===================================================================
    # Loss, Optimizer, Scheduler
    # ===================================================================
    criterion_name = config.criterion.name
    loss_func = LOSS_REGISTRY[criterion_name](**config.criterion.args)

    optimizer_name = config.optimizer.name
    parameters_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = OPTIMIZER_REGISTRY[optimizer_name](parameters_to_optimize, **config.optimizer.args)

    scheduler_name = config.lr_scheduler.name
    scheduler_args = config.lr_scheduler.args

    if scheduler_name == "CosineAnnealingLR":
        scheduler_args['T_max'] = config.trainer.n_epoch * len(train_loader)

    lr_scheduler = SCHEDULER_REGISTRY[scheduler_name](optimizer, **scheduler_args)

    eval_func = METRIC_REGISTRY[config.eval_func.name](**config.eval_func.args)

    # ===================================================================
    # W&B Logging
    # ===================================================================
    wandb.init(
        project="ml_hw3", 
        entity="r13945041-national-taiwan-university",
        name=config.name,
        dir="wandb_logs",
        config=OmegaConf.to_container(config, resolve=True)
    )
    wandb.watch(model, log="all")

    # ===================================================================
    # Trainer
    # ===================================================================
    trainer = Trainer(
        model=model.to(device),
        device=device,
        criterion=loss_func,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        eval_func=eval_func,
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_path=config.checkpoint.path,
        logger=wandb,
    )

    trainer_args = OmegaConf.to_container(config.trainer, resolve=True)
    trainer_args.pop("pretrain_path", None)

    trainer.fit(**trainer_args)
