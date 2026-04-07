# CNN VAE
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(num_channels: int, groups: int) -> int:
    groups = min(groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return max(groups, 1)


class ResBlock3D(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, groups: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(dim_in, groups), dim_in)
        self.norm2 = nn.GroupNorm(_group_count(dim_out, groups), dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.conv1 = nn.Conv3d(dim_in, dim_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(dim_out, dim_out, kernel_size=3, padding=1)
        self.skip = nn.Conv3d(dim_in, dim_out, kernel_size=1) if dim_in != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return (self.skip(x) + h) / math.sqrt(2.0)


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int, int]) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: tuple[int, int, int]) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.conv(x)


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
        groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.res1 = ResBlock3D(in_channels, in_channels, groups, dropout)
        self.res2 = ResBlock3D(in_channels, in_channels, groups, dropout)
        self.down = Downsample3D(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.res1(x)
        h = self.res2(h)
        return self.down(h)


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: tuple[int, int, int],
        groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.up = Upsample3D(in_channels, out_channels, scale_factor)
        self.res1 = ResBlock3D(out_channels, out_channels, groups, dropout)
        self.res2 = ResBlock3D(out_channels, out_channels, groups, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = self.res1(h)
        h = self.res2(h)
        return h


class UNetVAE(nn.Module):
    """3D UNet-style VAE with 2D latents (B, C, 8, 8) from (B, 1, 64, 32, 64)."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 16,
        base_channels: int = 32,
        groups: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels

        self.input_proj = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_strides = [
            (2, 2, 2),
            (2, 1, 2),
            (2, 2, 2),
            (2, 1, 1),
            (2, 1, 1),
            (2, 1, 1),
        ]
        self.up_strides = list(reversed(self.down_strides))

        channels = [base_channels, base_channels * 2, base_channels * 4]
        channels += [base_channels * 4] * (len(self.down_strides) - 2)
        self.down_blocks = nn.ModuleList(
            [
                DownBlock3D(channels[i], channels[i + 1], self.down_strides[i], groups, dropout)
                for i in range(len(self.down_strides))
            ]
        )

        self.mid1 = ResBlock3D(channels[-1], channels[-1], groups, dropout)
        self.mid2 = ResBlock3D(channels[-1], channels[-1], groups, dropout)

        self.mu = nn.Conv3d(channels[-1], latent_channels, kernel_size=1)
        self.logvar = nn.Conv3d(channels[-1], latent_channels, kernel_size=1)
        self.latent_proj = nn.Conv3d(latent_channels, channels[-1], kernel_size=1)

        self.up_blocks = nn.ModuleList(
            [
                UpBlock3D(
                    channels[-1 - i],
                    channels[-2 - i],
                    self.up_strides[i],
                    groups,
                    dropout,
                )
                for i in range(len(self.up_strides))
            ]
        )

        self.final = nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        for block in self.down_blocks:
            h = block(h)

        h = self.mid1(h)
        h = self.mid2(h)

        mu = self.mu(h).squeeze(2)
        logvar = self.logvar(h).squeeze(2)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.latent_proj(z.unsqueeze(2))
        for block in self.up_blocks:
            h = block(h)
        return self.final(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
