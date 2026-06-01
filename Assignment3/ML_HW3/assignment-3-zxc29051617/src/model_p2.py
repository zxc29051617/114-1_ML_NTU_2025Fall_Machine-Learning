import torch
import torch.nn as nn
from src.model import SimpleCNNUNet


# ----------------------
# Transformer Block
# ----------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Attention
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + h

        # FFN
        h = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + h
        return x


# ----------------------
# TransUNet
# ----------------------
class TransUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        embed_dim=512,
        trans_blocks=4,
        heads=8,
        mlp_dim=1024
    ):
        super().__init__()

        # ------------------
        # Load P1 UNet
        # ------------------
        self.cnn = SimpleCNNUNet(in_channels, out_channels)

        # Encoder
        self.inc = self.cnn.inc
        self.down1 = self.cnn.down1
        self.conv1 = self.cnn.conv1

        self.down2 = self.cnn.down2
        self.conv2 = self.cnn.conv2

        self.down3 = self.cnn.down3
        self.conv3 = self.cnn.conv3

        self.down4 = self.cnn.down4
        # conv4 不用

        # Decoder
        self.up1 = self.cnn.up1
        self.conv5 = self.cnn.conv5

        self.up2 = self.cnn.up2
        self.conv6 = self.cnn.conv6

        self.up3 = self.cnn.up3
        self.conv7 = self.cnn.conv7

        self.up4 = self.cnn.up4
        self.conv8 = self.cnn.conv8

        self.outc = self.cnn.outc

        # ------------------
        # Transformer BottleNeck
        # ------------------
        self.embed_dim = embed_dim

        # 1x1 conv → convert 512 → embed_dim
        self.patch_embed = nn.Conv2d(512, embed_dim, kernel_size=1)

        # Position embedding（dynamic）
        self.pos_embed = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, heads, mlp_dim)
            for _ in range(trans_blocks)
        ])

        # Project back to 1024 channels (UNet bottleneck)
        self.proj_back = nn.Linear(embed_dim, 1024)


    def forward(self, x):
        # ----------------
        # Encoder
        # ----------------
        x1 = self.inc(x)
        x2 = self.conv1(self.down1(x1))
        x3 = self.conv2(self.down2(x2))
        x4 = self.conv3(self.down3(x3))

        x5 = self.down4(x4)  # (B, 512, H/16, W/16)
        B, C, H, W = x5.shape
        N = H * W    

        # ----------------
        # Transformer
        # ----------------
        t = self.patch_embed(x5)          # (B, embed_dim, H, W)
        t = t.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        # 動態建立 Position Embedding
        if self.pos_embed is None or self.pos_embed.shape[1] != N:
            # 新建的 pos_embed 放在 x 的 device 上
            self.pos_embed = nn.Parameter(
                torch.zeros(1, N, self.embed_dim, device=x5.device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # 若模型被搬到不同 device，要同步 pos_embed
            if self.pos_embed.device != x5.device:
                self.pos_embed.data = self.pos_embed.data.to(x5.device)

        # 加上 position embedding
        t = t + self.pos_embed[:, :N, :]

        # Transformer blocks
        for blk in self.blocks:
            t = blk(t)

        # 回到 UNet 格式
        t = self.proj_back(t)                       # (B, N, 1024)
        t = t.transpose(1, 2).contiguous().view(B, 1024, H, W)

        # ----------------
        # Decoder
        # ----------------
        u1 = self.up1(t)
        u1 = self.conv5(torch.cat([u1, x4], dim=1))

        u2 = self.up2(u1)
        u2 = self.conv6(torch.cat([u2, x3], dim=1))

        u3 = self.up3(u2)
        u3 = self.conv7(torch.cat([u3, x2], dim=1))

        u4 = self.up4(u3)
        u4 = self.conv8(torch.cat([u4, x1], dim=1))

        return self.outc(u4)
