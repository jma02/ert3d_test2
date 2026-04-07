from typing import List

import torch
import torch.nn as nn
from einops import rearrange

from .modules3d import Downsample3D, Upsample3D, ResBlock3D


class Unet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        ch: int = 8,
        ch_mul: List[int] = [1, 2, 2, 2],
        groups: int = 32,
        dropout: float = 0.1,
        output_shape: tuple[int, int, int] = (64, 32, 64),
    ) -> None:
        super().__init__()

        self.ch = ch
        self.ch_mul = ch_mul
        self.dropout = dropout
        self.groups = groups
        self.output_shape = output_shape
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_proj = nn.Conv3d(self.in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList([])
        self.mid = None
        self.up = nn.ModuleList([])

        self.make_paths()

        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=2 * self.ch),
            nn.SiLU(),
            nn.Conv3d(2 * self.ch, self.out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = rearrange(x, "n c h w -> n c 1 h w")

        x = nn.AdaptiveAvgPool3d(self.output_shape)(x)
        initial_proj = self.input_proj(x)
        h = initial_proj

        down_path: list[torch.Tensor] = []
        for block_group in self.down:
            h = block_group[0](h)
            h = block_group[1](h)
            down_path.append(h)

            if len(block_group) > 2:
                h = block_group[2](h)

        h = self.mid[0](h)
        h = self.mid[1](h)

        for block_group in self.up:
            h = torch.cat((h, down_path.pop()), dim=1)
            h = block_group[0](h)
            h = block_group[1](h)

            if len(block_group) > 2:
                h = block_group[2](h)

        x = torch.cat((h, initial_proj), dim=1)
        return self.final(x)

    def make_transition(self, res: int, down: bool) -> nn.Module:
        dim = self.ch * self.ch_mul[res]

        if down:
            is_last_res = res == (len(self.ch_mul) - 1)
            if is_last_res:
                return Downsample3D(dim, dim)

            dim_out = self.ch * self.ch_mul[res + 1]
            return Downsample3D(dim, dim_out)

        is_first_res = res == 0
        if is_first_res:
            return Upsample3D(dim, dim)

        dim_out = self.ch * self.ch_mul[res - 1]
        return Upsample3D(dim, dim_out)

    def make_res(self, res: int, down: bool) -> nn.ModuleList:
        dim = self.ch * self.ch_mul[res]
        transition = self.make_transition(res, down)

        if down:
            block1 = ResBlock3D(dim, dim, self.groups, self.dropout)
            block2 = ResBlock3D(dim, dim, self.groups, self.dropout)
        else:
            block1 = ResBlock3D(2 * dim, dim, self.groups, self.dropout)
            block2 = ResBlock3D(dim, dim, self.groups, self.dropout)

        return nn.ModuleList([block1, block2, transition])

    def make_paths(self) -> None:
        num_res = len(self.ch_mul)

        for res in range(num_res):
            is_last_res = res == (num_res - 1)

            down_blocks = self.make_res(res, down=True)
            up_blocks = self.make_res(res, down=False)

            if is_last_res:
                down_blocks = down_blocks[:-1]
            if res == 0:
                up_blocks = up_blocks[:-1]

            self.down.append(down_blocks)
            self.up.insert(0, up_blocks)

        nch = self.ch * self.ch_mul[-1]
        self.mid = nn.ModuleList(
            [
                ResBlock3D(nch, nch, self.groups, self.dropout),
                ResBlock3D(nch, nch, self.groups, self.dropout),
            ]
        )
