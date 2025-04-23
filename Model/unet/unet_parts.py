""" Parts of the U-Net model: convolutional blocks, downsampling, upsampling, and output layers """
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Two consecutive 3x3 convolutions (with padding) each followed by BatchNorm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        # 3x3 convolution layers with padding to preserve spatial dimensions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class Down(nn.Module):
    """Downscaling: MaxPool 2x2 then double convolution."""
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x
class Up(nn.Module):
    """Upscaling: Transposed Conv 2x2 (to double spatial size) then double convolution after concatenating with skip connection."""
    def __init__(self, in_channels: int, out_channels: int):
        super(Up, self).__init__()
        # Transposed convolution (2x2) to upsample feature map by 2x.
        # Reduce channel count by half (in_channels -> in_channels//2) because of skip connection concatenation.
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Double convolution will take the concatenated channels (skip + upsampled) as input.
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for upsampling.
        Args:
            x1: Tensor from the lower (deeper) layer of decoder (to be upsampled).
            x2: Tensor from the corresponding skip connection (from encoder).
        """
        x1 = self.up(x1)  # upsample the lower layer
        # Compute padding needed to match x2 size (in case of odd dimensions)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        # Pad x1 (upsampled) on all sides as necessary to align with x2
        # (left, right, top, bottom) padding
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # Concatenate along channels dimension
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    """Final 1x1 convolution for output layer (maps to desired number of classes)."""
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)