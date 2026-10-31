import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_p2 import TransUNet

# === 你自己的 Dataset ===
from dataset import MedicalImageDataset
from dataset import LoadImaged, ToTensord, ResizeImaged


# ============================================================
# 建立 SSL 用的 transform（不需要 label、不需要 augment）
# ============================================================
def build_ssl_transform(image_size):
    return ComposeSSL([
        LoadImaged(keys=["image"]),
        ToTensord(keys=["image"]),
        ResizeImaged(keys=["image"], size=image_size),
    ])


# ============================================================
# Compose（你沒有提供，我幫你補）
# ============================================================
class ComposeSSL:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


# ============================================================
# Patch Masking (SSL core)
# ============================================================
def make_random_mask(img, mask_ratio=0.6, patch_size=8):
    """
    img: (B,1,H,W)
    return:
        masked_img (B,1,H,W)
        pixel_mask (B,1,H,W) = 1 -> masked region
    """
    B, C, H, W = img.shape
    device = img.device

    ph, pw = patch_size, patch_size
    assert H % ph == 0 and W % pw == 0

    h_patches = H // ph
    w_patches = W // pw
    N = h_patches * w_patches

    num_mask = int(mask_ratio * N)

    mask = torch.zeros((B, 1, h_patches, w_patches), device=device)

    for b in range(B):
        idx = torch.randperm(N, device=device)[:num_mask]
        mh = idx // w_patches
        mw = idx % w_patches
        mask[b, 0, mh, mw] = 1.0

    # 放大回 pixel space
    mask = mask.repeat_interleave(ph, dim=2).repeat_interleave(pw, dim=3)

    masked = img * (1 - mask)
    return masked, mask


# ============================================================
# SSL Pretrain
# ============================================================
def ssl_pretrain(
    data_dir,
    save_path,
    device="cuda",
    epochs=20,
    batch_size=8,
    lr=1e-4,
    mask_ratio=0.6,
    patch_size=8,
    image_size=256
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. 建立 Dataset
    transform = build_ssl_transform(image_size)
    dataset = MedicalImageDataset(
        data_dir=data_dir,
        has_label=False,
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 2. 模型
    model = TransUNet(in_channels=1, out_channels=1)
    model = model.to(device)

    # Reconstruction loss
    l1 = nn.L1Loss(reduction="none")
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    print("=== Start SSL Pretrain (Masked Image Reconstruction) ===")

    # 3. training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_samples = 0

        for batch in loader:
            img = batch["image"].to(device)  # (B,1,H,W)

            masked_img, pixel_mask = make_random_mask(
                img, mask_ratio=mask_ratio, patch_size=patch_size
            )

            # forward
            recon = model(masked_img)

            # loss only on masked pixels
            loss_map = l1(recon, img)
            masked_loss = loss_map * pixel_mask
            loss = masked_loss.sum() / (pixel_mask.sum() + 1e-8)

            optim.zero_grad()
            loss.backward()
            optim.step()

            bs = img.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

        avg = total_loss / total_samples
        print(f"[Epoch {epoch}/{epochs}] SSL Loss = {avg:.6f}")

        # 每 epoch 存權重
        torch.save(model.state_dict(), save_path)

    print(f"=== Finished. Saved SSL weights to {save_path} ===")


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="checkpoints/transunet_ssl.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)

    args = parser.parse_args()

    ssl_pretrain(
        data_dir=args.data_dir,
        save_path=args.save_path,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
        image_size=args.image_size
    )


if __name__ == "__main__":
    main()
