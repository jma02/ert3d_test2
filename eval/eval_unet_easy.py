import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.unet import Unet


torch.manual_seed(159753)
np.random.seed(159753)


def make_dataloader(data_path: str, batch_size: int) -> tuple[DataLoader, dict]:
    data = np.load(data_path, allow_pickle=True).item()
    x = torch.from_numpy(np.asarray(data["X"], dtype=np.float32))
    y = torch.from_numpy(np.asarray(data["Y_easy"], dtype=np.float32))

    n_samples = y.shape[0]
    perm = torch.randperm(n_samples)
    split = int(0.9 * n_samples)
    val_idx = perm[split:]

    x_val = x[val_idx]
    y_val = y[val_idx]

    y_val = torch.log(torch.clamp(y_val, min=1e-12))

    val_ds = TensorDataset(x_val, y_val)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    meta = {
        "data": data,
        "val_dataset": val_ds,
        "val_indices": val_idx.cpu().numpy(),
    }
    return val_loader, meta


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    numer = torch.linalg.vector_norm(diff.flatten(1), dim=1)
    denom = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-12)
    return numer / denom


def rel_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    numer = diff.flatten(1).abs().sum(dim=1)
    denom = target.flatten(1).abs().sum(dim=1).clamp_min(1e-12)
    return numer / denom


def eval_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    rel_l2_list = []
    rel_l1_list = []
    gt_list = []
    pred_list = []
    input_list = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            rel_l2_list.append(rel_l2(pred, y).cpu())
            rel_l1_list.append(rel_l1(pred, y).cpu())
            gt_list.append(y.detach().cpu().numpy())
            pred_list.append(pred.detach().cpu().numpy())
            input_list.append(x.detach().cpu().numpy())

    rel_l2_all = torch.cat(rel_l2_list).numpy()
    rel_l1_all = torch.cat(rel_l1_list).numpy()
    gt_arr = np.concatenate(gt_list, axis=0)
    pred_arr = np.concatenate(pred_list, axis=0)
    input_arr = np.concatenate(input_list, axis=0)
    return rel_l2_all, rel_l1_all, gt_arr, pred_arr, input_arr


@dataclass
class SampleViz:
    label: str
    rel_error: float
    y_log: np.ndarray
    pred_log: np.ndarray
    abs_err: np.ndarray
    input_signal: np.ndarray


def build_samples(
    rel_errors: np.ndarray,
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    input_arr: np.ndarray,
) -> List[SampleViz]:
    sorted_idx = np.argsort(rel_errors)
    best = sorted_idx[0]
    worst = sorted_idx[-1]
    median = sorted_idx[len(sorted_idx) // 2]

    samples = []
    for label, idx in ("Best", best), ("Median", median), ("Worst", worst):
        y_log = gt_arr[idx]
        pred_log = pred_arr[idx]
        abs_err = np.abs(pred_log - y_log)
        input_signal = input_arr[idx]
        samples.append(
            SampleViz(
                label=f"{label} (rel L2={rel_errors[idx]:.4f})",
                rel_error=float(rel_errors[idx]),
                y_log=y_log,
                pred_log=pred_log,
                abs_err=abs_err,
                input_signal=input_signal,
            )
        )
    return samples


def plot_samples(
    samples: List[SampleViz],
    extent: tuple[float, float, float, float],
    output_path: str,
) -> None:
    plt.rcParams["font.family"] = "DejaVu Serif"
    title_font = {"family": "DejaVu Serif", "weight": "bold", "size": 12}

    fig, axs = plt.subplots(
        len(samples) * 3,
        4,
        figsize=(20, 3.2 * len(samples) * 3),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.8]},
    )
    if len(samples) == 1:
        axs = np.expand_dims(axs, axis=0)

    for s_idx, sample in enumerate(samples):
        for slice_idx in range(min(3, sample.y_log.shape[0])):
            gt_slice = sample.y_log[slice_idx]
            pred_slice = sample.pred_log[slice_idx]
            shared_vmin = min(gt_slice.min(), pred_slice.min())
            shared_vmax = max(gt_slice.max(), pred_slice.max())
            row = s_idx * 3 + slice_idx
            plots = [
                (gt_slice, "Ground Truth (log)", True),
                (pred_slice, "Prediction (log)", True),
                (sample.abs_err[slice_idx], "Abs Error", True),
                (sample.input_signal, "Input signals", False),
            ]
            for col, (values, title, is_spatial) in enumerate(plots):
                ax = axs[row, col]
                if is_spatial:
                    is_error = title == "Abs Error"
                    cmap = "magma" if is_error else "Blues"
                    im = ax.imshow(
                        values,
                        origin="lower",
                        aspect="auto",
                        extent=extent,
                        cmap=cmap,
                        vmin=shared_vmin if title in {"Ground Truth (log)", "Prediction (log)"} else None,
                        vmax=shared_vmax if title in {"Ground Truth (log)", "Prediction (log)"} else None,
                    )
                else:
                    signals = values.reshape(values.shape[0], -1)
                    if slice_idx == 0:
                        for ch_idx, label in enumerate(["line1", "line2", "line3"]):
                            ax.plot(signals[ch_idx], linewidth=1.5, label=label)
                        ax.set_title("Input signals", fontdict=title_font)
                        ax.set_xlabel("Index")
                        ax.set_ylabel("Signal")
                        ax.legend(frameon=False, fontsize=8)
                    else:
                        ax.axis("off")
                    continue
                if is_error:
                    ax.set_title(
                        f"Slice {slice_idx + 1}: {title}",
                        fontdict={**title_font, "weight": "bold"},
                    )
                else:
                    ax.set_title(f"Slice {slice_idx + 1}: {title}", fontdict=title_font)
                ax.set_xlabel("X")
                ax.set_ylabel("Z")
                cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, shrink=0.75)
                cb.ax.tick_params(labelsize=9)

            axs[row, 0].text(
                -0.25,
                0.5,
                sample.label,
                transform=axs[row, 0].transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=13,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#ffe8c2" if s_idx % 2 == 0 else "#d7f0ff",
                    edgecolor="#ff6b00" if s_idx % 2 == 0 else "#1f78b4",
                    linewidth=1.6,
                ),
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_exp_samples(samples: List[SampleViz]) -> List[SampleViz]:
    exp_samples = []
    for sample in samples:
        y_exp = np.exp(sample.y_log)
        pred_exp = np.exp(sample.pred_log)
        abs_err = np.abs(pred_exp - y_exp)
        diff = pred_exp - y_exp
        numer = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1).sum()
        denom = np.linalg.norm(y_exp.reshape(y_exp.shape[0], -1), axis=1).sum()
        rel_error = float(numer / max(denom, 1e-12))
        exp_samples.append(
            SampleViz(
                label=sample.label.split("(")[0].strip() + f" (rel L2={rel_error:.4f})",
                rel_error=rel_error,
                y_log=y_exp,
                pred_log=pred_exp,
                abs_err=abs_err,
                input_signal=sample.input_signal,
            )
        )
    return exp_samples


def build_random_samples(
    rel_errors: np.ndarray,
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    input_arr: np.ndarray,
    n_samples: int = 3,
) -> List[SampleViz]:
    n_total = rel_errors.shape[0]
    n_samples = min(n_samples, n_total)
    random_idx = np.random.choice(n_total, size=n_samples, replace=False)

    samples = []
    for i, idx in enumerate(random_idx, start=1):
        y_log = gt_arr[idx]
        pred_log = pred_arr[idx]
        abs_err = np.abs(pred_log - y_log)
        input_signal = input_arr[idx]
        samples.append(
            SampleViz(
                label=f"Random {i} (rel L2={rel_errors[idx]:.4f})",
                rel_error=float(rel_errors[idx]),
                y_log=y_log,
                pred_log=pred_log,
                abs_err=abs_err,
                input_signal=input_signal,
            )
        )
    return samples


def plot_log_histogram(gt_arr: np.ndarray, output_path: str) -> None:
    plt.rcParams["font.family"] = "DejaVu Serif"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = gt_arr.ravel()
    ax.hist(values, bins=80, color="#3b82f6", alpha=0.85, edgecolor="white")
    ax.set_title("Ground Truth Log-Values Histogram", fontsize=12, fontweight="bold")
    ax.set_xlabel("log(value)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/3dert_test2_easy.npy")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    loader, meta = make_dataloader(args.data, args.batch_size)

    in_c, in_h, in_w = meta["val_dataset"][0][0].shape
    out_c, out_h, out_w = meta["val_dataset"][0][1].shape

    device = torch.device(args.device)
    model = Unet(
        in_channels=in_c,
        out_channels=out_c,
        output_shape=(out_h, out_w),
        ch=64,
    ).to(device)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    rel_l2_all, rel_l1_all, gt_arr, pred_arr, input_arr = eval_model(model, loader, device)

    gt_exp = np.exp(gt_arr)
    pred_exp = np.exp(pred_arr)
    rel_l2_exp = rel_l2(
        torch.from_numpy(pred_exp.astype(np.float32)),
        torch.from_numpy(gt_exp.astype(np.float32)),
    ).numpy()
    rel_l1_exp = rel_l1(
        torch.from_numpy(pred_exp.astype(np.float32)),
        torch.from_numpy(gt_exp.astype(np.float32)),
    ).numpy()

    print("Relative L2: mean={:.6f} median={:.6f}".format(rel_l2_all.mean(), np.median(rel_l2_all)))
    print("Relative L1: mean={:.6f} median={:.6f}".format(rel_l1_all.mean(), np.median(rel_l1_all)))
    print("Exp Relative L2: mean={:.6f} median={:.6f}".format(rel_l2_exp.mean(), np.median(rel_l2_exp)))
    print("Exp Relative L1: mean={:.6f} median={:.6f}".format(rel_l1_exp.mean(), np.median(rel_l1_exp)))

    samples = build_samples(rel_l2_all, gt_arr, pred_arr, input_arr)
    exp_samples = build_exp_samples(samples)
    xq = meta["data"]["xq"]
    zq = meta["data"]["zq"]
    extent = (float(xq[0]), float(xq[-1]), float(zq[0]), float(zq[-1]))
    os.makedirs("eval_outputs", exist_ok=True)
    plot_path = os.path.join("eval_outputs", "eval_unet_easy.png")
    plot_samples(samples, extent, plot_path)
    print(f"Saved qualitative plot to {plot_path}")

    exp_plot_path = os.path.join("eval_outputs", "eval_unet_easy_exp.png")
    plot_samples(exp_samples, extent, exp_plot_path)
    print(f"Saved exp-space qualitative plot to {exp_plot_path}")

    hist_path = os.path.join("eval_outputs", "eval_unet_easy_log_hist.png")
    plot_log_histogram(gt_arr, hist_path)
    print(f"Saved log histogram to {hist_path}")

    random_samples = build_random_samples(rel_l2_all, gt_arr, pred_arr, input_arr, n_samples=3)
    random_plot_path = os.path.join("eval_outputs", "eval_unet_easy_random.png")
    plot_samples(random_samples, extent, random_plot_path)
    print(f"Saved random qualitative plot to {random_plot_path}")

    exp_random_samples = build_exp_samples(random_samples)
    exp_random_plot_path = os.path.join("eval_outputs", "eval_unet_easy_random_exp.png")
    plot_samples(exp_random_samples, extent, exp_random_plot_path)
    print(f"Saved exp-space random qualitative plot to {exp_random_plot_path}")


if __name__ == "__main__":
    main()
