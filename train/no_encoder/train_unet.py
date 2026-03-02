import argparse
import os

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.unet import Unet


torch.manual_seed(159753)
np.random.seed(159753)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


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


class TVHuberLoss(nn.Module):
    """
    loss = SmoothL1(pred, target) + lam_tv * TV(pred)
    Optional: TV on residual instead (tv_on="residual").
    Optional: simple reweighting of the inclusion region (w_in>1).
    """

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
        if self.tv == "aniso":
            reg = tv2d_aniso(tv_arg)
        else:
            reg = tv2d_iso(tv_arg, eps=self.eps)

        return data + self.lam_tv * reg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/3dert_test2_easy.npy")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="unet_easy_no_encoder")
    args = parser.parse_args()

    epochs = 250
    batch_size = 64
    lr = 5e-3
    log_clamp_min = 1e-12

    data = np.load(args.data, allow_pickle=True).item()
    x = torch.from_numpy(np.asarray(data["X"], dtype=np.float32))
    y = torch.from_numpy(np.asarray(data["Y_easy"], dtype=np.float32))

    n_samples, in_c, in_h, in_w = x.shape
    _, out_c, out_h, out_w = y.shape

    perm = torch.randperm(n_samples)
    split = int(0.9 * n_samples)
    train_idx = perm[:split]
    val_idx = perm[split:]

    x_train = x[train_idx]
    x_val = x[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    y_train = torch.log(torch.clamp(y_train, min=log_clamp_min))
    y_val = torch.log(torch.clamp(y_val, min=log_clamp_min))

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device)

    model = Unet(
        in_channels=in_c,
        out_channels=out_c,
        output_shape=(out_h, out_w),
        ch=64,
    ).to(device)
    model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=1e-3)

    loss_fn = TVHuberLoss(
        lam_tv=3e-4,
        beta=0.5,
        tv="iso",
        tv_on="pred",
        w_in=10.0,
        thresh=0.5,
    )

    os.makedirs(f"saved_runs/{args.save_dir}", exist_ok=True)
    os.chdir(f"saved_runs/{args.save_dir}")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    ckpt = args.ckpt
    best_val_loss = float("inf")

    if ckpt is not None:
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        log_step = int(checkpoint["log_step"])
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
    else:
        start_epoch = 0
        log_step = 0

    pbar = tqdm(range(start_epoch, epochs + 1), desc="Epochs")
    for epoch in pbar:
        model.train()
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)

            if torch.isnan(loss):
                continue

            loss.backward()
            optim.step()
            log_step += 1

        model.eval()
        with torch.no_grad():
            x_batch, y_batch = next(iter(val_loader))
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            val_loss = loss_fn(pred, y_batch).item()
            pbar.set_postfix({"val_loss": f"{val_loss:.5f}"})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
                checkpoint = {
                    "epoch": int(epoch),
                    "log_step": int(log_step),
                    "model_state_dict": model_to_save.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    "best_val_loss": float(best_val_loss),
                }
                torch.save(checkpoint, "checkpoints/best_val_loss.tar")

            xq = data["xq"]
            zq = data["zq"]
            extent = (float(xq[0]), float(xq[-1]), float(zq[0]), float(zq[-1]))

            plt.rcParams["font.family"] = "DejaVu Serif"
            title_font = {"family": "DejaVu Serif", "weight": "bold", "size": 12}

            y_log = y_batch[0].detach().cpu().numpy()
            pred_log = pred[0].detach().cpu().numpy()
            abs_err = np.abs(pred_log - y_log)

            fig, axs = plt.subplots(3, 3, figsize=(16, 10), gridspec_kw={"width_ratios": [1, 1, 1]})
            for idx in range(min(3, y_log.shape[0])):
                plots = [
                    (y_log[idx], "Ground Truth (log)"),
                    (pred_log[idx], "Prediction (log)"),
                    (abs_err[idx], "Abs Error"),
                ]
                for ax, (values, title) in zip(axs[idx, :], plots):
                    if title in {"Ground Truth (log)", "Prediction (log)"}:
                        im = ax.imshow(
                            values,
                            origin="lower",
                            aspect="auto",
                            extent=extent,
                            cmap="Blues",
                            vmin=y_log[idx].min(),
                            vmax=y_log[idx].max(),
                        )
                    else:
                        im = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap="Blues")
                    ax.set_title(f"Slice {idx + 1}: {title}", fontdict=title_font)
                    ax.set_xlabel("X")
                    ax.set_ylabel("Z")
                    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, shrink=0.75)
                    cb.ax.tick_params(labelsize=9)

            plt.tight_layout()
            fig.savefig(f"samples/epoch_{epoch}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

if __name__ == "__main__":
    main()
