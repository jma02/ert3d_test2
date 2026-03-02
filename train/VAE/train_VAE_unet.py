import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.VAE.default import UNetVAE


def load_sigma_dataset(
    path: Path,
    log_clamp_min: float,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    sigma = data["Y"].astype(np.float32, copy=False)
    sigma = sigma[:, None, ...]
    sigma = np.log(np.clip(sigma, log_clamp_min, None))
    xq = data["xq"].astype(np.float32, copy=False)
    yq = data["yq"].astype(np.float32, copy=False)
    zq = data["zq"].astype(np.float32, copy=False)
    return torch.from_numpy(sigma), xq, yq, zq


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def tv3d_iso(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dz = u[..., 1:, :, :] - u[..., :-1, :, :]
    dy = u[..., :, 1:, :] - u[..., :, :-1, :]
    dx = u[..., :, :, 1:] - u[..., :, :, :-1]
    dz = F.pad(dz, (0, 0, 0, 0, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dx = F.pad(dx, (0, 1, 0, 0, 0, 0))
    return torch.sqrt(dx * dx + dy * dy + dz * dz + eps).mean()


class TVHuberLoss3D(nn.Module):
    """
    loss = SmoothL1(pred, target) + lam_tv * TV(pred)
    Optional: TV on residual instead (tv_on="residual").
    Optional: simple reweighting of the inclusion region (w_in>1).
    """

    def __init__(
        self,
        lam_tv: float = 1e-3,
        beta: float = 1.0,
        tv_on: str = "pred",
        w_in: float = 1.0,
        thresh: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lam_tv = lam_tv
        self.beta = beta
        self.tv_on = tv_on
        self.w_in = w_in
        self.thresh = thresh
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = pred.float()
        target_f = target.float()

        r = pred_f - target_f

        if self.w_in > 1.0:
            bg = target_f.flatten(2).median(dim=-1).values[..., None, None, None]
            mask = (target_f - bg).abs() > self.thresh
            w = 1.0 + (self.w_in - 1.0) * mask.float()
            data = (w * F.smooth_l1_loss(pred_f, target_f, beta=self.beta, reduction="none")).mean()
        else:
            data = F.smooth_l1_loss(pred_f, target_f, beta=self.beta)

        tv_arg = pred_f if self.tv_on == "pred" else r
        reg = tv3d_iso(tv_arg, eps=self.eps)
        return data + self.lam_tv * reg


def save_slices(
    target: torch.Tensor,
    recon: torch.Tensor,
    out_path: Path,
    xq: np.ndarray,
    yq: np.ndarray,
    zq: np.ndarray,
    sigma_min: float,
    max_points: int,
    generated: torch.Tensor,
) -> None:

    target = target.detach().cpu().squeeze(0).squeeze(0).numpy()
    recon = recon.detach().cpu().squeeze(0).squeeze(0).numpy()
    generated_np = generated.detach().cpu().squeeze(0).squeeze(0).numpy()

    zz, yy, xx = np.meshgrid(zq, yq, xq, indexing="ij")
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    zz = zz.reshape(-1)

    flat_target = target.reshape(-1)
    sigma_cutoff = max(sigma_min, float(np.quantile(flat_target, 0.05)))
    shared_mask = flat_target > sigma_cutoff
    if not np.any(shared_mask):
        shared_mask = flat_target > sigma_min

    def sample_points(
        vol: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sigma = vol.reshape(-1)
        mask = (mask if mask is not None else sigma > sigma_min)
        x = xx[mask]
        y = yy[mask]
        z = zz[mask]
        sigma = sigma[mask]
        if sigma.size == 0:
            return x, y, z, sigma
        if sigma.size > max_points:
            idx = np.linspace(0, sigma.size - 1, max_points).astype(int)
            x = x[idx]
            y = y[idx]
            z = z[idx]
            sigma = sigma[idx]
        return x, y, z, sigma

    x_t, y_t, z_t, s_t = sample_points(target, shared_mask)
    x_r, y_r, z_r, s_r = sample_points(recon, shared_mask)
    x_g, y_g, z_g, s_g = sample_points(generated_np, shared_mask)

    def scatter_with_alpha(ax, x, y, z, sigma, title, norm, cmap):
        vmin = float(norm.vmin)
        vmax = float(norm.vmax)
        denom = vmax - vmin if vmax > vmin else 1.0
        alpha = (vmax - sigma) / denom
        alpha = np.clip(alpha, 0.05, 1.0)

        colors = cmap(norm(sigma))
        colors[:, 3] = alpha

        sc = ax.scatter(x, y, z, c=colors, s=4)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return sc

    target_min = float(s_t.min())
    target_max = float(s_t.max())
    norm = plt.Normalize(vmin=target_min, vmax=target_max)
    cmap = plt.get_cmap("plasma")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    ncols = 3
    fig = plt.figure(figsize=(5 * ncols, 4))
    ax1 = fig.add_subplot(1, ncols, 1, projection="3d")
    ax2 = fig.add_subplot(1, ncols, 2, projection="3d")
    ax3 = fig.add_subplot(1, ncols, 3, projection="3d")

    scatter_with_alpha(ax1, x_t, y_t, z_t, s_t, "target", norm, cmap)
    fig.colorbar(mappable, ax=ax1, fraction=0.03, pad=0.08, label="log(sigma)")

    scatter_with_alpha(ax2, x_r, y_r, z_r, s_r, "recon", norm, cmap)
    fig.colorbar(mappable, ax=ax2, fraction=0.03, pad=0.08, label="log(sigma)")

    scatter_with_alpha(ax3, x_g, y_g, z_g, s_g, "prior sample", norm, cmap)
    fig.colorbar(mappable, ax=ax3, fraction=0.03, pad=0.08, label="log(sigma)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(159753)
np.random.seed(159753)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a UNet VAE on sigma voxels.")
    parser.add_argument("--data_path", type=str, default="data/3dert_test2.npy")
    parser.add_argument("--save_dir", type=str, default="vae_sigma_unet")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    device = args.device

    log_clamp_min = 1e-12
    val_split = 0.05
    batch_size = 256
    epochs = 2000
    lr = 2e-4
    beta = 1e-2
    kl_warmup_epochs = 600
    recon_beta = 0.5
    tv_lambda = 3e-4
    num_workers = 16
    sigma_min = float(np.log(log_clamp_min))
    max_plot_points = 8000

    sigma, xq, yq, zq = load_sigma_dataset(
        Path(args.data_path),
        log_clamp_min=log_clamp_min,
    )

    sigma = sigma.to(device, non_blocking=True)
    num_workers = 0

    pin_memory = False

    dataset = TensorDataset(sigma)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(159753)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = UNetVAE(base_channels=8, latent_channels=8).to(device)
    model = torch.compile(model)

    loss_fn = TVHuberLoss3D(
        lam_tv=tv_lambda,
        beta=recon_beta,
        w_in=10.0,
        thresh=0.5,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=1e-3)
    amp_device_type = "cuda" if device.startswith("cuda") else "cpu"

    beta_tag = f"beta{beta:.0e}"
    kl_tag = f"kl{kl_warmup_epochs}"
    save_dir = f"{args.save_dir}_{beta_tag}_{kl_tag}"
    save_root = Path("saved_runs") / save_dir
    (save_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_root / "samples").mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_loss = float("inf")
    ema_state_dict = None
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        ema_state_dict = checkpoint.get("ema_state_dict")

    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ema_model = AveragedModel(
        base_model,
        multi_avg_fn=get_ema_multi_avg_fn(0.999),
    ).to(device)
    if ema_state_dict is not None:
        ema_model.load_state_dict(ema_state_dict)

    epoch_bar = tqdm(range(start_epoch, epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        kl_weight = beta * min(1.0, epoch / max(1, kl_warmup_epochs))
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0

        for (x,) in tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False):
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=amp_device_type, enabled=device.startswith("cuda"), dtype=torch.bfloat16):
                recon, mu, logvar = model(x)
                recon_loss = loss_fn(recon, x)
                kl_loss = kl_divergence(mu, logvar)
                loss = recon_loss + kl_weight * kl_loss

            if not torch.isfinite(loss):
                continue

            loss.backward()
            optimizer.step()
            ema_model.update_parameters(model._orig_mod if hasattr(model, "_orig_mod") else model)

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()

        model.eval()
        ema_model.module.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon, mu, logvar = ema_model.module(x)
                recon_loss = loss_fn(recon, x)
                kl_loss = kl_divergence(mu, logvar)
                val_loss += (recon_loss + kl_weight * kl_loss).item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        epoch_bar.set_postfix(
            train=f"{train_loss:.6f}",
            recon=f"{train_recon / max(1, len(train_loader)):.6f}",
            kl=f"{train_kl / max(1, len(train_loader)):.6f}",
            val=f"{val_loss:.6f}",
            kl_weight=f"{kl_weight:.6f}",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model._orig_mod.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "log_clamp_min": log_clamp_min,
                "best_val_loss": float(best_val_loss),
                "ema_state_dict": ema_model.state_dict(),
            }
            torch.save(checkpoint, save_root / "checkpoints" / "best_val_loss.pt")

        if epoch % 5 == 0 or epoch == epochs:
            sample = next(iter(val_loader))[0][:1].to(device)
            recon, _, _ = ema_model.module(sample)
            prior_z = torch.randn((1, 8, 8, 8), device=device)
            prior_sample = ema_model.module.decode(prior_z)
            save_slices(
                sample,
                recon,
                save_root / "samples" / f"recon_epoch_{epoch}.png",
                xq,
                yq,
                zq,
                sigma_min,
                max_plot_points,
                generated=prior_sample,
            )

    final_checkpoint = {
        "epoch": epochs,
        "model_state_dict": model._orig_mod.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "log_clamp_min": log_clamp_min,
        "best_val_loss": float(best_val_loss),
        "ema_state_dict": ema_model.state_dict(),
    }
    torch.save(final_checkpoint, save_root / "checkpoints" / "final_model.pt")


if __name__ == "__main__":
    main()
