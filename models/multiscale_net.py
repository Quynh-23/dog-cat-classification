import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MultiScaleNet(nn.Module):
    def __init__(self, mode='net0', num_classes=2):
        super().__init__()
        self.mode = mode

        # Up branch
        self.up_b1 = ConvBlock(3, 64)
        self.up_b2 = ConvBlock(64, 128)
        self.up_b3 = ConvBlock(128, 256)

        # Down branch
        self.down_b1 = ConvBlock(3, 64)

        if mode == 'net1':
            self.down_b2 = ConvBlock(64 + 64, 128)
            self.down_b3 = ConvBlock(128 + 128, 256)
        else:
            self.down_b2 = ConvBlock(64, 128)
            self.down_b3 = ConvBlock(128, 256)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x_up = x
        x_down = F.interpolate(x, scale_factor=0.5)

        # Stage 1
        f_up1 = self.up_b1(x_up)
        f_down1 = self.down_b1(x_down)

        # Stage 2
        f_up2 = self.up_b2(f_up1)
        f_up1_ds = F.interpolate(f_up1, size=f_down1.shape[2:])

        if self.mode == 'net0':
            f_down2 = self.down_b2(f_down1)

        elif self.mode == 'net1':
            f_down2 = self.down_b2(torch.cat([f_down1, f_up1_ds], dim=1))

        elif self.mode == 'net2':
            f_down2 = self.down_b2((f_down1 + f_up1_ds) / 2)

        elif self.mode == 'net3':
            attn = torch.sigmoid(f_up1_ds)
            f_down2 = self.down_b2(f_down1 * attn)

        # Stage 3
        f_up3 = self.up_b3(f_up2)
        f_up2_ds = F.interpolate(f_up2, size=f_down2.shape[2:])

        if self.mode == 'net1':
            f_down3 = self.down_b3(torch.cat([f_down2, f_up2_ds], dim=1))
        elif self.mode == 'net2':
            f_down3 = self.down_b3((f_down2 + f_up2_ds) / 2)
        elif self.mode == 'net3':
            attn = torch.sigmoid(f_up2_ds)
            f_down3 = self.down_b3(f_down2 * attn)
        else:
            f_down3 = self.down_b3(f_down2)

        # Pool
        out_up = self.pool(f_up3)
        out_down = self.pool(f_down3)

        out = torch.cat([out_up, out_down], dim=1)
        out = torch.flatten(out, 1)

        return self.fc(out)