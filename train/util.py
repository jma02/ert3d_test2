import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

PLOT_FONT_FAMILY = "DejaVu Serif"
PLOT_TITLE_FONT = {"family": PLOT_FONT_FAMILY}
PLOT_TITLE_FONTSIZE = 12
PLOT_TITLE_FONTWEIGHT = "bold"


def tv2d_aniso(u: torch.Tensor) -> torch.Tensor:
    dx = u[..., :, 1:] - u[..., :, :-1]
    dy = u[..., 1:, :] - u[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def tv2d_iso(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dx = u[..., :, 1:] - u[..., :, :-1]
    dy = u[..., 1:, :] - u[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + eps).mean()


def tv3d_aniso(u: torch.Tensor) -> torch.Tensor:
    dz = u[..., 1:, :, :] - u[..., :-1, :, :]
    dy = u[..., :, 1:, :] - u[..., :, :-1, :]
    dx = u[..., :, :, 1:] - u[..., :, :, :-1]
    return dz.abs().mean() + dy.abs().mean() + dx.abs().mean()


def tv3d_iso(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dz = u[..., 1:, :, :] - u[..., :-1, :, :]
    dy = u[..., :, 1:, :] - u[..., :, :-1, :]
    dx = u[..., :, :, 1:] - u[..., :, :, :-1]
    dz = F.pad(dz, (0, 0, 0, 0, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dx = F.pad(dx, (0, 1, 0, 0, 0, 0))
    return torch.sqrt(dx * dx + dy * dy + dz * dz + eps).mean()


class TVHuberLoss2D(nn.Module):
    def __init__(
        self,
        lam_tv: float = 1e-3,
        beta: float = 1.0,
        tv: str = "iso",
        tv_on: str = "pred",
        w_in: float = 1.0,
        thresh: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lam_tv = lam_tv
        self.beta = beta
        self.tv = tv
        self.tv_on = tv_on
        self.w_in = w_in
        self.thresh = thresh
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = pred.float()
        target_f = target.float()
        r = pred_f - target_f

        if self.w_in > 1.0:
            bg = target_f.flatten(2).median(dim=-1).values[..., None, None]
            mask = (target_f - bg).abs() > self.thresh
            w = 1.0 + (self.w_in - 1.0) * mask.float()
            data = (w * F.smooth_l1_loss(pred_f, target_f, beta=self.beta, reduction="none")).mean()
        else:
            data = F.smooth_l1_loss(pred_f, target_f, beta=self.beta)

        tv_arg = pred_f if self.tv_on == "pred" else r
        reg = tv2d_aniso(tv_arg) if self.tv == "aniso" else tv2d_iso(tv_arg, eps=self.eps)
        return data + self.lam_tv * reg


class TVHuberLoss3D(nn.Module):
    def __init__(
        self,
        lam_tv: float = 1e-3,
        beta: float = 1.0,
        tv: str = "iso",
        tv_on: str = "pred",
        w_in: float = 1.0,
        thresh: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lam_tv = lam_tv
        self.beta = beta
        self.tv = tv
        self.tv_on = tv_on
        self.w_in = w_in
        self.thresh = thresh
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = pred.float()
        target_f = target.float()
        r = pred_f - target_f

        if self.w_in > 1.0:
            bg = target_f.flatten(2).median(dim=-1).values[..., None, None]
            mask = (target_f - bg).abs() > self.thresh
            w = 1.0 + (self.w_in - 1.0) * mask.float()
            data = (w * F.smooth_l1_loss(pred_f, target_f, beta=self.beta, reduction="none")).mean()
        else:
            data = F.smooth_l1_loss(pred_f, target_f, beta=self.beta)

        tv_arg = pred_f if self.tv_on == "pred" else r
        reg = tv3d_aniso(tv_arg) if self.tv == "aniso" else tv3d_iso(tv_arg, eps=self.eps)
        return data + self.lam_tv * reg


def _nanify(values: list[float | None]) -> list[float]:
    return [np.nan if v is None else v for v in values]
