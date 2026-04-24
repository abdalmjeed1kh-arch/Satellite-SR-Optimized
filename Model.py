import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def icnr_init(conv, upscale_factor=2):
    """ICNR initialization for sub-pixel convolution to prevent checkerboard artifacts."""
    new_shape = [conv.weight.shape[0] // (upscale_factor**2)] + list(conv.weight.shape[1:])
    sub_weights = torch.randn(new_shape)
    sub_weights = init.kaiming_normal_(sub_weights, nonlinearity='leaky_relu')
    conv.weight.data.copy_(sub_weights.repeat(upscale_factor**2, 1, 1, 1))

class PartialConv(nn.Module):
    """Partial Convolution based Padding (PCP) to handle boundary artifacts in patches."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, use_weight_norm=True):
        super(PartialConv, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if use_weight_norm:
            self.conv = nn.utils.weight_norm(conv)
        else:
            self.conv = conv
            
        self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
        self.slide_winsize = kernel_size * kernel_size

    def forward(self, x):
        with torch.no_grad():
            mask = torch.ones(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
            mask_count = F.conv2d(mask, self.weight_maskUpdater.to(x.device), 
                                 stride=self.conv.stride, padding=self.conv.padding)

        raw_out = self.conv(x)
        mask_ratio = self.slide_winsize / (mask_count + 1e-8)
        
        if self.conv.bias is not None:
            bias_view = self.conv.bias.view(1, self.conv.out_channels, 1, 1)
            return (raw_out - bias_view) * mask_ratio + bias_view
        else:
            return raw_out * mask_ratio

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        # Using PartialConv with Weight Norm inside
        self.conv1 = PartialConv(channels, channels, kernel_size=3, padding=1,use_weight_norm=True)
        self.prelu = nn.PReLU()
        self.conv2 = PartialConv(channels, channels, kernel_size=3, padding=1,use_weight_norm=True)
        self.res_scale = 0.1 

    def forward(self, x):        
        out = self.conv1(x)
        out = self.prelu(out)
        out = self.conv2(out)
        return x + out * self.res_scale

class UpsampleBlock(nn.Module):
    def __init__(self, channels=64, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        # Standard Conv2d is fine for upsampling, but ICNR is a must
        conv = nn.Conv2d(channels, channels * (scale_factor ** 2), kernel_size=3, padding=1)
        icnr_init(conv, upscale_factor=scale_factor)
        
        self.block = nn.Sequential(
            conv,
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)

class SRresnet(nn.Module):
    def __init__(self, num_residual_blocks=16, scale_factor=4):
        super(SRresnet, self).__init__()
        
        # Initial feature extraction (Kernel 9x9)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # Residual Body
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        # Middle Conv using PCP and Weight Norm
        self.conv2 = PartialConv(64, 64, kernel_size=3, padding=1, use_weight_norm=True)

        # Upsampling Stage
        if scale_factor == 4:
            self.upsample_blocks = nn.Sequential(
                UpsampleBlock(64, 2),
                UpsampleBlock(64, 2)
            )
        else:
            self.upsample_blocks = UpsampleBlock(64, scale_factor)

        # Final reconstruction (Kernel 9x9)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        # 1. Mean Shift Down
        x = x - 0.5
        
        initial = self.conv1(x)
        out = self.residual_blocks(initial)
        out = self.conv2(out)
        
        # Global Residual Connection
        out = out + initial
        
        out = self.upsample_blocks(out)
        out = self.conv3(out)
        
        # 2. Mean Shift Back Up
        return out + 0.5

# --- Test code ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRresnet(num_residual_blocks=16, scale_factor=4).to(device)
    dummy_input = torch.randn(1, 3, 12, 12).to(device) 
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") 