import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg, max_], dim=1)
        attn = torch.sigmoid(self.conv(x_cat))
        return x * attn

class ScaleAttention(nn.Module):
    def __init__(self, channels, num_scales):
        super().__init__()
        self.fc = nn.Linear(channels, num_scales)

    def forward(self, features):
        stacked = torch.stack(features, dim=1)  # [B, S, C, H, W]
        b, s, c, h, w = stacked.shape

        pooled = stacked.mean(dim=[3,4])  # [B, S, C]
        weights = self.fc(pooled.mean(dim=1))  # [B, S]
        weights = torch.softmax(weights, dim=1)

        weights = weights.view(b, s, 1, 1, 1)
        out = (stacked * weights).sum(dim=1)

        return out

class MultiBranchResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_branches=3):
        super().__init__()
        self.branches = nn.ModuleList()

        for i in range(num_branches):
            if i == 0:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            elif i == 1:
                mid = out_channels // 4
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, mid, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(mid),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid, out_channels, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride, 2, dilation=2, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 2, dilation=2, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            self.branches.append(branch)

        self.scale_attn = ScaleAttention(out_channels, num_branches)
        self.se = SEBlock(out_channels)
        self.sa = SpatialAttention()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        branch_outs = [b(x) for b in self.branches]

        out = self.scale_attn(branch_outs)

        out = self.se(out)
        out = self.sa(out)

        out += residual
        return F.relu(out)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)

class MultiBranchResNetImproved(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = MultiBranchResidualSEBlock(64, 64)
        self.layer2 = MultiBranchResidualSEBlock(64, 128, stride=2)
        self.layer3 = MultiBranchResidualSEBlock(128, 256, stride=2)
        self.layer4 = MultiBranchResidualSEBlock(256, 512, stride=2)

        self.head1 = nn.Linear(64, num_classes)
        self.head2 = nn.Linear(128, num_classes)
        self.head3 = nn.Linear(256, num_classes)
        self.head4 = nn.Linear(512, num_classes)

        self.projection = ProjectionHead(512)

    def forward(self, x, return_features=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        def pool(feat):
            return torch.flatten(F.adaptive_avg_pool2d(feat, 1), 1)

        p1 = pool(f1)
        p2 = pool(f2)
        p3 = pool(f3)
        p4 = pool(f4)

        out1 = self.head1(p1)
        out2 = self.head2(p2)
        out3 = self.head3(p3)
        out4 = self.head4(p4)

        out = (out1 + out2 + out3 + out4) / 4

        if return_features:
            proj = self.projection(p4)
            return out, proj, [p1, p2, p3, p4]

        return out