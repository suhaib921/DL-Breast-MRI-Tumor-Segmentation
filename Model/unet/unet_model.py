import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import DoubleConv, Down, Up, OutConv

class UNet(nn.Module):
    """
    U-Net model for binary segmentation.

    This model implements the U-Net architecture&#8203;:contentReference[oaicite:0]{index=0} with an encoder-decoder structure and skip connections.
    It takes an input tensor of shape (N, C, H, W) and outputs a tensor of shape (N, 1, H, W) representing a segmentation mask 
    for the positive class. All convolutions use padding to preserve spatial dimensions, so the output height and width 
    match the input. No pre-trained weights are used; all weights are initialized from scratch.

    Args:
        in_channels (int): Number of channels in the input images (e.g., 3 for RGB).
        out_channels (int): Number of output channels (for binary segmentation, use 1).
        init_features (int): Number of feature channels in the first convolution layer (default=64). 
                              This number doubles at each downsampling step.

    Example:
        >>> model = UNet(in_channels=3, out_channels=1, init_features=64)
        >>> x = torch.randn(1, 3, 300, 300)
        >>> pred_mask = model(x)  # pred_mask will have shape (1, 1, 300, 300)
    """
    def __init__(
                self, 
                in_channels: int = 3, 
                out_channels: int = 1, 
                init_features: int = 64,
                ):
        super(UNet, self).__init__()
        features = init_features
        # Encoder (downsampling path)
        self.inc    = DoubleConv(in_channels, features)        # Initial double conv
        self.down1  = Down(features, features * 2)
        self.down2  = Down(features * 2, features * 4)
        self.down3  = Down(features * 4, features * 8)
        self.down4  = Down(features * 8, features * 16)
        
       # Decoder (upsampling path)
        self.up4    = Up(features * 16, features * 8)
        self.up3    = Up(features * 8, features * 4)
        self.up2    = Up(features * 4, features * 2)
        self.up1    = Up(features * 2, features)

        # Final 1x1 convolution (output)
        self.outc   = OutConv(features, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: capture feature maps
        x1 = self.inc(x)            # Feature map size: (features, H, W)
        x2 = self.down1(x1)         # (features*2, H/2, W/2)
        x3 = self.down2(x2)         # (features*4, H/4, W/4)
        x4 = self.down3(x3)         # (features*8, H/8, W/8)
        x5 = self.down4(x4)         # (features*16, H/16, W/16) - bottleneck features
        
        # Decoder: upsample and fuse with skip connections
        x = self.up4(x5, x4)        # (features*8, H/8, W/8)
        x = self.up3(x, x3)         # (features*4, H/4, W/4)
        x = self.up2(x, x2)         # (features*2, H/2, W/2)
        x = self.up1(x, x1)         # (features, H, W)
        output = self.outc(x)       # (1, H, W) raw logits for the mask
        # Apply sigmoid to get probabilities in [0,1]
        output = torch.sigmoid(output)
        return output