"""
SRResNet: Super-Resolution Residual Network for 4x satellite image upscaling.

Architecture features:
  - Partial Convolution Padding (PCP) for better edge handling.
  - Conditional Batch Normalization (CBN) for flexible feature normalization.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    """Convolutional layer with Partial Convolution Padding (PCP).

    Instead of zero-padding, the layer conditions its output only on valid
    (non-padded) input pixels, preventing border artifacts that are
    particularly noticeable in satellite imagery.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        # Fixed mask convolution — counts the number of valid pixels per window.
        self.mask_conv = nn.Conv2d(
            1,
            1,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        with torch.no_grad():
            mask_out = self.mask_conv(mask)
        # Normalise by the ratio of total-to-valid pixels.
        kernel_area = self.mask_conv.weight.numel()
        ratio = kernel_area / (mask_out + 1e-8)
        out = self.conv(x) * ratio
        return out


class ConditionalBatchNorm2d(nn.Module):
    """Conditional Batch Normalization (CBN).

    Applies a learnable per-channel affine transform conditioned on an
    optional embedding vector.  When no condition is provided the layer
    behaves as standard Batch Normalization.
    """

    def __init__(self, num_features: int, condition_dim: int = 0) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.condition_dim = condition_dim
        if condition_dim > 0:
            self.gamma_fc = nn.Linear(condition_dim, num_features)
            self.beta_fc = nn.Linear(condition_dim, num_features)
            nn.init.ones_(self.gamma_fc.weight)
            nn.init.zeros_(self.gamma_fc.bias)
            nn.init.zeros_(self.beta_fc.weight)
            nn.init.zeros_(self.beta_fc.bias)
        else:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.bn(x)
        if self.condition_dim > 0 and condition is not None:
            gamma = self.gamma_fc(condition).unsqueeze(-1).unsqueeze(-1)
            beta = self.beta_fc(condition).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = self.gamma.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
        return gamma * out + beta


class ResidualBlock(nn.Module):
    """Residual block using PCP convolutions and CBN normalisation."""

    def __init__(self, channels: int = 64, condition_dim: int = 0) -> None:
        super().__init__()
        self.conv1 = PartialConv2d(channels, channels, 3, padding=1, bias=False)
        self.cbn1 = ConditionalBatchNorm2d(channels, condition_dim)
        self.conv2 = PartialConv2d(channels, channels, 3, padding=1, bias=False)
        self.cbn2 = ConditionalBatchNorm2d(channels, condition_dim)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x
        out = F.relu(self.cbn1(self.conv1(x), condition), inplace=True)
        out = self.cbn2(self.conv2(out), condition)
        return out + residual


class UpsampleBlock(nn.Module):
    """Sub-pixel convolution block for 2x upsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.pixel_shuffle(self.conv(x)))


class SRResNet(nn.Module):
    """SRResNet for 4x satellite image super-resolution.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        num_features: Number of feature channels in the residual blocks.
        num_residual_blocks: Depth of the residual tower.
        scale_factor: Upscaling factor; must be a power of 2.
        condition_dim: Dimension of an optional conditioning vector fed into
            every CBN layer.  Set to 0 (default) for unconditional mode.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_residual_blocks: int = 16,
        scale_factor: int = 4,
        condition_dim: int = 0,
    ) -> None:
        super().__init__()
        if scale_factor not in (2, 4, 8):
            raise ValueError("scale_factor must be one of 2, 4, 8.")
        self.num_upsample = int(math.log2(scale_factor))

        # Initial feature extraction.
        self.head = PartialConv2d(in_channels, num_features, 9, padding=4)

        # Residual tower.
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_features, condition_dim) for _ in range(num_residual_blocks)]
        )

        # Post-residual convolution + normalisation.
        self.mid_conv = PartialConv2d(num_features, num_features, 3, padding=1, bias=False)
        self.mid_cbn = ConditionalBatchNorm2d(num_features, condition_dim)

        # Upsampling blocks (each performs 2x).
        self.upsample_blocks = nn.Sequential(
            *[UpsampleBlock(num_features) for _ in range(self.num_upsample)]
        )

        # Output projection.
        self.tail = nn.Conv2d(num_features, in_channels, 9, padding=4)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        head = F.relu(self.head(x), inplace=True)

        res = head
        for block in self.residual_blocks:
            res = block(res, condition)

        res = self.mid_cbn(self.mid_conv(res), condition) + head
        res = self.upsample_blocks(res)
        return torch.sigmoid(self.tail(res))
