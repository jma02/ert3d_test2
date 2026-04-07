import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


def make_skip_connection_3d(dim_in: int, dim_out: int) -> nn.Module:
    if dim_in == dim_out:
        return nn.Identity()
    return nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=True)


def make_block_3d(dim_in: int, dim_out: int, num_groups: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=dim_in),
        nn.SiLU(),
        nn.Dropout3d(dropout) if dropout != 0 else nn.Identity(),
        nn.Conv3d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
    )


class ResBlock3D(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_groups: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.skip_connection = make_skip_connection_3d(dim_in, dim_out)
        self.block1 = make_block_3d(dim_in, dim_out, num_groups, dropout=0)
        self.block2 = make_block_3d(dim_out, dim_out, num_groups, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return (self.skip_connection(x) + h) / np.sqrt(2.0)
