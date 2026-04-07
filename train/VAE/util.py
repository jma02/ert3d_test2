import numpy as np
import matplotlib.pyplot as plt
import torch

from train.util import (
    PLOT_FONT_FAMILY,
    PLOT_TITLE_FONT,
    PLOT_TITLE_FONTSIZE,
    PLOT_TITLE_FONTWEIGHT,
    _nanify,
)


def plot_vae_loss_curves(
    loss_history: dict[str, list[float | None]],
    loss_title: str,
    out_path: str = "loss_curve.png",
) -> None:
    plt.rcParams["font.family"] = PLOT_FONT_FAMILY

    fig, axes = plt.subplots(3, 1, figsize=(7, 8.6), sharex=True)

    axes[0].plot(loss_history["train_total"], label="train total")
    axes[0].plot(loss_history["val_total"], label="val total")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(
        "Total",
        fontdict=PLOT_TITLE_FONT,
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight=PLOT_TITLE_FONTWEIGHT,
    )
    axes[0].legend()

    axes[1].plot(_nanify(loss_history["train_recon"]), label="train recon")
    axes[1].plot(_nanify(loss_history["val_recon"]), label="val recon")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(
        "Reconstruction",
        fontdict=PLOT_TITLE_FONT,
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight=PLOT_TITLE_FONTWEIGHT,
    )
    axes[1].legend()

    axes[2].plot(_nanify(loss_history["train_kl"]), label="train kl")
    axes[2].plot(_nanify(loss_history["val_kl"]), label="val kl")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title(
        "KL",
        fontdict=PLOT_TITLE_FONT,
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight=PLOT_TITLE_FONTWEIGHT,
    )
    axes[2].legend()

    fig.suptitle(
        loss_title,
        fontdict=PLOT_TITLE_FONT,
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight=PLOT_TITLE_FONTWEIGHT,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_vae_slices(
    target: torch.Tensor,
    recon: torch.Tensor,
    out_path: str,
    xq: np.ndarray,
    yq: np.ndarray,
    zq: np.ndarray,
    sigma_min: float,
    max_points: int,
    generated: torch.Tensor,
) -> None:
    target_np = target.detach().cpu().squeeze(0).squeeze(0).numpy()
    recon_np = recon.detach().cpu().squeeze(0).squeeze(0).numpy()
    generated_np = generated.detach().cpu().squeeze(0).squeeze(0).numpy()

    zz, yy, xx = np.meshgrid(zq, yq, xq, indexing="ij")
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    zz = zz.reshape(-1)

    flat_target = target_np.reshape(-1)
    sigma_cutoff = max(sigma_min, float(np.quantile(flat_target, 0.05)))
    shared_mask = flat_target > sigma_cutoff
    if not np.any(shared_mask):
        shared_mask = flat_target > sigma_min

    def sample_points(
        vol: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sigma = vol.reshape(-1)
        mask = mask if mask is not None else sigma > sigma_min
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

    x_t, y_t, z_t, s_t = sample_points(target_np, shared_mask)
    x_r, y_r, z_r, s_r = sample_points(recon_np, shared_mask)
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
    cmap = plt.get_cmap("PuBu")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    fig = plt.figure(figsize=(15, 4))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    scatter_with_alpha(ax1, x_t, y_t, z_t, s_t, "target", norm, cmap)
    fig.colorbar(mappable, ax=ax1, fraction=0.03, pad=0.08, label="log(sigma)")

    scatter_with_alpha(ax2, x_r, y_r, z_r, s_r, "recon", norm, cmap)
    fig.colorbar(mappable, ax=ax2, fraction=0.03, pad=0.08, label="log(sigma)")

    scatter_with_alpha(ax3, x_g, y_g, z_g, s_g, "prior sample", norm, cmap)
    fig.colorbar(mappable, ax=ax3, fraction=0.03, pad=0.08, label="log(sigma)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
