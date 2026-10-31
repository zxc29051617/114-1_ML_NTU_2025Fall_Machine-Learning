#TODO: Implement your model
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SimpleCNNUNet(nn.Module):
    """
    Problem 1 的 Simple-CNN UNet 模型。
    架構完全基於 Lecture 5, Page 72 的 U-Net 圖 。
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(SimpleCNNUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.MaxPool2d(2)  
        self.conv1 = DoubleConv(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(512, 1024) # 這是 U-Net 最底部的 "Bottleneck"

        # --- Decoder (Expansive Path) ---
        # 這是 U-Net 的右側 [cite: 1289]
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # 這是 "up-conv 2x2" 
        self.conv5 = DoubleConv(1024, 512) # 1024 = 512 (from up1) + 512 (from conv3 skip)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(512, 256) # 512 = 256 (from up2) + 256 (from conv2 skip)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(256, 128) # 256 = 128 (from up3) + 128 (from conv1 skip)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(128, 64) # 128 = 64 (from up4) + 64 (from inc skip)

        # --- Output Layer ---
        # 這是 "conv 1x1"，將 channel 轉換為最終的 class 數量 (二元分割為 1) 
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)     # 儲存 x1 供 Skip Connection
        x2 = self.down1(x1)
        x2 = self.conv1(x2)   # 儲存 x2 供 Skip Connection
        x3 = self.down2(x2)
        x3 = self.conv2(x3)   # 儲存 x3 供 Skip Connection
        x4 = self.down3(x3)
        x4 = self.conv3(x4)   # 儲存 x4 供 Skip Connection
        x5 = self.down4(x4)
        x5 = self.conv4(x5)   # Bottleneck

        # Decoder + Skip Connections 
        up1 = self.up1(x5)
        # 拼接 (Concatenate) 來自 Encoder 的特徵圖 (Skip Connection)
        x = torch.cat([x4, up1], dim=1) # dim=1 是 channel 維度
        x = self.conv5(x)

        up2 = self.up2(x)
        x = torch.cat([x3, up2], dim=1)
        x = self.conv6(x)

        up3 = self.up3(x)
        x = torch.cat([x2, up3], dim=1)
        x = self.conv7(x)

        up4 = self.up4(x)
        x = torch.cat([x1, up4], dim=1)
        x = self.conv8(x)

        # 輸出
        logits = self.outc(x)
        return logits