import matplotlib.pyplot as plt
import numpy as np

from train.util import (
    PLOT_FONT_FAMILY,
    PLOT_TITLE_FONT,
    PLOT_TITLE_FONTSIZE,
    PLOT_TITLE_FONTWEIGHT,
)


def plot_no_encoder_loss_curves(
    train_values: list[float],
    val_values: list[float],
    loss_title: str,
    out_path: str = "loss_curve.png",
) -> None:
    plt.rcParams["font.family"] = PLOT_FONT_FAMILY

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.4))
    ax.plot(train_values, label="train loss")
    ax.plot(val_values, label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        "Loss",
        fontdict=PLOT_TITLE_FONT,
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight=PLOT_TITLE_FONTWEIGHT,
    )
    ax.legend()

    fig.suptitle(
        loss_title,
        fontdict=PLOT_TITLE_FONT,
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight=PLOT_TITLE_FONTWEIGHT,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_no_encoder_slices(
    target: np.ndarray,
    pred: np.ndarray,
    out_path: str,
    extent: tuple[float, float, float, float],
) -> None:
    plt.rcParams["font.family"] = "DejaVu Serif"
    title_font = {"family": "DejaVu Serif", "weight": "bold", "size": 12}

    abs_err = np.abs(pred - target)

    fig, axs = plt.subplots(3, 3, figsize=(16, 10), gridspec_kw={"width_ratios": [1, 1, 1]})
    for idx in range(min(3, target.shape[0])):
        plots = [
            (target[idx], "Ground Truth (log)"),
            (pred[idx], "Prediction (log)"),
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
                    vmin=target[idx].min(),
                    vmax=target[idx].max(),
                )
            else:
                im = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap="PuBu")
            ax.set_title(f"Slice {idx + 1}: {title}", fontdict=title_font)
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, shrink=0.75)
            cb.ax.tick_params(labelsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
