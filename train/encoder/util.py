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


def plot_encoder_loss_curves(
    loss_history: dict[str, list[float | None]],
    loss_title: str,
    out_path: str = "loss_curve.png",
) -> None:
    plt.rcParams["font.family"] = PLOT_FONT_FAMILY
    sections = ["total"]
    has_pixel = any(v is not None for v in loss_history["train_pixel"] + loss_history["val_pixel"])
    has_latent = any(v is not None for v in loss_history["train_latent"] + loss_history["val_latent"])
    if has_pixel:
        sections.append("pixel")
    if has_latent:
        sections.append("latent")

    fig_height = 2.6 * len(sections) + 0.8
    fig, axes = plt.subplots(len(sections), 1, figsize=(7, fig_height), sharex=True)
    if len(sections) == 1:
        axes = [axes]

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

    axis_idx = 1
    if "pixel" in sections:
        axes[axis_idx].plot(_nanify(loss_history["train_pixel"]), label="train pixel")
        axes[axis_idx].plot(_nanify(loss_history["val_pixel"]), label="val pixel")
        axes[axis_idx].set_ylabel("Loss")
        axes[axis_idx].set_title(
            "Pixel",
            fontdict=PLOT_TITLE_FONT,
            fontsize=PLOT_TITLE_FONTSIZE,
            fontweight=PLOT_TITLE_FONTWEIGHT,
        )
        axes[axis_idx].legend()
        axis_idx += 1

    if "latent" in sections:
        axes[axis_idx].plot(_nanify(loss_history["train_latent"]), label="train latent")
        axes[axis_idx].plot(_nanify(loss_history["val_latent"]), label="val latent")
        axes[axis_idx].set_xlabel("Epoch")
        axes[axis_idx].set_ylabel("Loss")
        axes[axis_idx].set_title(
            "Latent",
            fontdict=PLOT_TITLE_FONT,
            fontsize=PLOT_TITLE_FONTSIZE,
            fontweight=PLOT_TITLE_FONTWEIGHT,
        )
        axes[axis_idx].legend()

    fig.suptitle(
        loss_title,
        fontdict=PLOT_TITLE_FONT,
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight=PLOT_TITLE_FONTWEIGHT,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_scatter_slices(
    target: torch.Tensor,
    pred: torch.Tensor,
    out_path: str,
    xq: np.ndarray,
    yq: np.ndarray,
    zq: np.ndarray,
    sigma_min: float,
    max_points: int = 100000,
    subtitle: str | None = None,
    colorbar_label: str = "log(sigma)",
) -> None:
    target_np = target.detach().cpu().squeeze(0).squeeze(0).numpy()
    pred_np = pred.detach().cpu().squeeze(0).squeeze(0).numpy()
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
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vals = vol.reshape(-1)
        mask = mask & (np.isfinite(vals))
        x = xx[mask]
        y = yy[mask]
        z = zz[mask]
        vals = vals[mask]
        if vals.size == 0:
            return x, y, z, vals
        if vals.size > max_points:
            idx = np.linspace(0, vals.size - 1, max_points).astype(int)
            x = x[idx]
            y = y[idx]
            z = z[idx]
            vals = vals[idx]
        return x, y, z, vals

    abs_err_signed = pred_np - target_np

    x_t, y_t, z_t, s_t = sample_points(target_np, shared_mask)
    x_p, y_p, z_p, s_p = sample_points(pred_np, shared_mask)
    x_e, y_e, z_e, s_e = sample_points(abs_err_signed, shared_mask)

    target_min = float(s_t.min()) if s_t.size else 0.0
    target_max = float(s_t.max()) if s_t.size else 1.0
    target_norm = plt.Normalize(vmin=target_min, vmax=target_max)
    target_cmap = plt.get_cmap("PuBu")

    err_abs = float(np.abs(s_e).max()) if s_e.size else 1.0
    err_abs = max(err_abs, 1e-8)
    err_norm = plt.Normalize(vmin=-err_abs, vmax=err_abs)
    err_cmap = plt.get_cmap("RdBu_r")

    def scatter_with_alpha(ax, x, y, z, sigma, norm, cmap, alpha_mode: str) -> None:
        colors = cmap(norm(sigma))
        if alpha_mode == "sigma":
            vmin = float(norm.vmin)
            vmax = float(norm.vmax)
            denom = vmax - vmin if vmax > vmin else 1.0
            alpha = (vmax - sigma) / denom
            alpha = np.clip(alpha, 0.05, 1.0)
        else:
            abs_max = float(norm.vmax) if norm.vmax else 1.0
            alpha = np.abs(sigma) / abs_max
            alpha = np.log1p(np.clip(alpha, 0.0, 1.0)) / np.log(2.0)
            alpha = np.clip(alpha, 0.0, 1.0)
        colors[:, 3] = alpha
        ax.scatter(x, y, z, c=colors, s=4)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.rcParams["font.family"] = PLOT_FONT_FAMILY

    fig = plt.figure(figsize=(22, 6))
    fig.subplots_adjust(left=0.05, right=0.98, wspace=0.12)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    scatter_with_alpha(ax1, x_t, y_t, z_t, s_t, target_norm, target_cmap, "sigma")
    ax1.set_title(
        "Ground Truth",
        fontdict=PLOT_TITLE_FONT,
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight=PLOT_TITLE_FONTWEIGHT,
    )
    scatter_with_alpha(ax2, x_p, y_p, z_p, s_p, target_norm, target_cmap, "sigma")
    ax2.set_title(
        "Prediction",
        fontdict=PLOT_TITLE_FONT,
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight=PLOT_TITLE_FONTWEIGHT,
    )
    scatter_with_alpha(ax3, x_e, y_e, z_e, s_e, err_norm, err_cmap, "error")
    ax3.set_title(
        "Error (pred - target)",
        fontdict=PLOT_TITLE_FONT,
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight=PLOT_TITLE_FONTWEIGHT,
    )

    cb_kw = dict(shrink=0.5, aspect=18, pad=0.08)
    cb1 = fig.colorbar(
        plt.cm.ScalarMappable(norm=target_norm, cmap=target_cmap),
        ax=ax1, **cb_kw,
    )
    cb1.ax.set_title(colorbar_label, fontsize=9, pad=4)
    cb2 = fig.colorbar(
        plt.cm.ScalarMappable(norm=target_norm, cmap=target_cmap),
        ax=ax2, **cb_kw,
    )
    cb2.ax.set_title(colorbar_label, fontsize=9, pad=4)
    cb3 = fig.colorbar(
        plt.cm.ScalarMappable(norm=err_norm, cmap=err_cmap),
        ax=ax3, **cb_kw,
    )
    cb3.ax.set_title("error", fontsize=9, pad=4)

    if subtitle is not None:
        fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=10,
                 fontfamily=PLOT_FONT_FAMILY)
        fig.subplots_adjust(bottom=0.08)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
