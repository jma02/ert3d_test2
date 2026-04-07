import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from eval.noise.noise_model import NOISE_SEED, add_noise_to_loader
from models.unet import Unet
from train.util import PLOT_FONT_FAMILY, PLOT_TITLE_FONT, PLOT_TITLE_FONTSIZE, PLOT_TITLE_FONTWEIGHT


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "saved_runs" / "grid_search_no_encoder"
OUTPUT_ROOT = REPO_ROOT / "eval_outputs" / "grid_search_no_encoder"
DATA_PATH = REPO_ROOT / "data" / "3dert_test2_easy.npy"
LOG_EPS = 1e-6
SPLIT_SEED = 159753
RANDOM_SAMPLE_SEED = 42
N_RANDOM_SAMPLES = 5


def fixed_random_indices(test_size: int, n: int, seed: int) -> list[int]:
    return np.random.RandomState(seed).choice(test_size, size=n, replace=False).tolist()


def find_checkpoint(run_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = run_dir / "checkpoints" / name
        if path.exists():
            return path
    return None


def build_test_loader(
    batch_size: int,
    device: torch.device,
) -> tuple[DataLoader, tuple[int, int, int], tuple[int, int, int], np.ndarray, np.ndarray]:
    data = np.load(DATA_PATH, allow_pickle=True).item()
    xq = data["xq"]
    zq = data["zq"]
    x = torch.from_numpy(np.asarray(data["X"], dtype=np.float32)).to(device)
    y = torch.from_numpy(np.asarray(data["Y_easy"], dtype=np.float32)).to(device)
    y = torch.log(torch.clamp(y, min=LOG_EPS))
    n_samples, in_c, in_h, in_w = x.shape
    _, out_c, out_h, out_w = y.shape
    dataset = TensorDataset(x, y)
    train_size = int(0.8 * n_samples)
    val_size = int(0.1 * n_samples)
    test_size = n_samples - train_size - val_size
    _, _, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    return test_loader, (in_c, in_h, in_w), (out_c, out_h, out_w), xq, zq


def relative_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_flat = pred.float().reshape(pred.shape[0], -1)
    target_flat = target.float().reshape(target.shape[0], -1)
    diff = pred_flat - target_flat
    rel_l2 = torch.linalg.vector_norm(diff, ord=2, dim=1) / torch.clamp(
        torch.linalg.vector_norm(target_flat, ord=2, dim=1), min=1e-12,
    )
    rel_l1 = torch.linalg.vector_norm(diff, ord=1, dim=1) / torch.clamp(
        torch.linalg.vector_norm(target_flat, ord=1, dim=1), min=1e-12,
    )
    return rel_l1, rel_l2


def mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample MAE (mean over pixels)."""
    return (pred.float() - target.float()).abs().reshape(pred.shape[0], -1).mean(dim=1)


def summarize(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {
            f"mean_{prefix}": None,
            f"median_{prefix}": None,
            f"min_{prefix}": None,
            f"max_{prefix}": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"mean_{prefix}": float(arr.mean()),
        f"median_{prefix}": float(np.median(arr)),
        f"min_{prefix}": float(arr.min()),
        f"max_{prefix}": float(arr.max()),
    }


def plot_sample_slices(
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    sample_index: int,
    rel_l2: float,
    rel_l1: float,
    xq: np.ndarray,
    zq: np.ndarray,
    value_label: str = "σ (S/m)",
    noise_level: float | None = None,
) -> None:
    plt.rcParams["font.family"] = PLOT_FONT_FAMILY
    extent = (float(xq[0]), float(xq[-1]), float(zq[0]), float(zq[-1]))
    n_slices = gt.shape[0]
    err = pred - gt  # signed error

    fig, axes = plt.subplots(n_slices, 3, figsize=(18, 4.5 * n_slices), constrained_layout=True)
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    for s in range(n_slices):
        shared_vmin = float(min(gt[s].min(), pred[s].min()))
        shared_vmax = float(max(gt[s].max(), pred[s].max()))
        err_abs = float(max(np.abs(err[s]).max(), 1e-8))

        panels = [
            (gt[s],   f"GT ({value_label})",          "Blues",  shared_vmin, shared_vmax),
            (pred[s], f"Pred ({value_label})",         "Blues",  shared_vmin, shared_vmax),
            (err[s],  "Error (pred − target)", "RdBu_r", -err_abs,    err_abs),
        ]
        for ax, (values, title, cmap, vmin, vmax) in zip(axes[s], panels):
            im = ax.imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(
                f"Slice {s + 1}: {title}",
                fontdict=PLOT_TITLE_FONT,
                fontsize=PLOT_TITLE_FONTSIZE,
                fontweight=PLOT_TITLE_FONTWEIGHT,
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, shrink=0.8)
            cb.ax.tick_params(labelsize=9)

    noise_str = f"noise={noise_level} | " if noise_level is not None else ""
    fig.suptitle(
        f"{noise_str}Sample {sample_index} | rel_l2={rel_l2:.6f} | rel_l1={rel_l1:.6f}",
        fontdict=PLOT_TITLE_FONT,
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight=PLOT_TITLE_FONTWEIGHT,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_run(
    run_dir: Path,
    device: torch.device,
    output_root: Path,
    noise_level: float | None = None,
    noise_seed: int = NOISE_SEED,
) -> dict[str, object]:
    hparams_path = run_dir / "hparams.json"
    with hparams_path.open("r", encoding="utf-8") as f:
        hparams = json.load(f)

    best_ckpt = find_checkpoint(run_dir, ("best_val.pt", "best_val_loss.pt"))
    final_ckpt = find_checkpoint(run_dir, ("final_model.pt", "final.pt"))
    run_name = hparams.get("save_dir", run_dir.name)
    if isinstance(run_name, str):
        run_name = Path(run_name).name

    row: dict[str, object] = {
        "run_dir": str(run_dir),
        "save_dir": run_name,
        "noise_level": noise_level,
        "lr": float(hparams["lr"]),
        "batch_size": int(hparams["batch_size"]),
        "epochs": int(hparams["epochs"]),
        "unet_ch": int(hparams["unet_ch"]),
        "weight_decay": float(hparams.get("weight_decay", 1e-3)),
        "best_checkpoint": str(best_ckpt) if best_ckpt is not None else None,
        "final_checkpoint": str(final_ckpt) if final_ckpt is not None else None,
        "training_done": best_ckpt is not None and final_ckpt is not None,
        "status": "ok" if best_ckpt is not None and final_ckpt is not None else "training_not_done",
    }

    row.update(summarize([], "rel_l1"))
    row.update(summarize([], "rel_l2"))
    row.update(summarize([], "mae"))

    if best_ckpt is None or final_ckpt is None:
        return row

    test_loader, (in_c, _, _), (out_c, out_h, out_w), xq, zq = build_test_loader(
        batch_size=int(hparams["batch_size"]),
        device=device,
    )
    if noise_level is not None and noise_level > 0:
        test_loader = add_noise_to_loader(test_loader, noise_level, seed=noise_seed)
    model = Unet(
        in_channels=in_c,
        out_channels=out_c,
        output_shape=(out_h, out_w),
        ch=int(hparams["unet_ch"]),
    ).to(device)
    checkpoint = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rel_l1_values: list[float] = []
    rel_l2_values: list[float] = []
    mae_values: list[float] = []
    gt_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred = model(x_batch)
            # invert log(sigma + eps) → sigma
            pred_sigma = torch.exp(pred) - LOG_EPS
            gt_sigma = torch.exp(y_batch) - LOG_EPS
            rel_l1, rel_l2 = relative_metrics(pred_sigma, gt_sigma)
            mae = mean_absolute_error(pred_sigma, gt_sigma)
            rel_l1_values.extend(rel_l1.cpu().tolist())
            rel_l2_values.extend(rel_l2.cpu().tolist())
            mae_values.extend(mae.cpu().tolist())
            gt_list.append(gt_sigma.cpu().numpy())
            pred_list.append(pred_sigma.cpu().numpy())

    row.update(summarize(rel_l1_values, "rel_l1"))
    row.update(summarize(rel_l2_values, "rel_l2"))
    row.update(summarize(mae_values, "mae"))

    gt_arr = np.concatenate(gt_list, axis=0)
    pred_arr = np.concatenate(pred_list, axis=0)
    rel_l2_arr = np.array(rel_l2_values)
    rel_l1_arr = np.array(rel_l1_values)

    out_dir = output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    test_set_summary = {
        "run": run_name,
        "noise_level": noise_level,
        "checkpoint": str(best_ckpt),
        "n_test_samples": len(rel_l2_values),
        **summarize(rel_l2_values, "rel_l2"),
        **summarize(rel_l1_values, "rel_l1"),
        **summarize(mae_values, "mae"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(test_set_summary, f, indent=2)

    sorted_idx = np.argsort(rel_l2_arr)
    rand_idx = fixed_random_indices(len(rel_l2_arr), N_RANDOM_SAMPLES, RANDOM_SAMPLE_SEED)

    selections: dict[str, list[int]] = {
        "best":   [int(sorted_idx[0])],
        "median": [int(sorted_idx[len(sorted_idx) // 2])],
        "worst":  [int(sorted_idx[-1])],
        "random": rand_idx,
    }
    subdir_map = {
        "best":   (out_dir / "best",        "best"),
        "median": (out_dir / "median",      "median"),
        "worst":  (out_dir / "worst",       "worst"),
        "random": (out_dir / "fixed_evals", "fixed"),
    }
    for label, indices in selections.items():
        base_dir, label_prefix = subdir_map[label]
        for enum_i, idx in enumerate(indices):
            sample_dir = base_dir / f"fixed_{enum_i}" if label == "random" else base_dir
            sample_dir.mkdir(parents=True, exist_ok=True)
            plot_sample_slices(
                gt=gt_arr[idx],
                pred=pred_arr[idx],
                out_path=sample_dir / f"{label_prefix}_idx{idx:04d}_rel_l2{rel_l2_arr[idx]:.4f}.png",
                sample_index=idx,
                rel_l2=float(rel_l2_arr[idx]),
                rel_l1=float(rel_l1_arr[idx]),
                xq=xq,
                zq=zq,
                noise_level=noise_level,
            )

    return row


def main() -> None:
    from eval.noise.noise_model import DEFAULT_NOISE_LEVELS

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default=str(RUNS_ROOT))
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--output", type=str, default=None,
                        help="Single output file path (csv/parquet) for non-noise mode")
    parser.add_argument("--noise_levels", type=str, default="",
                        help="Comma-separated noise levels (fraction, 1.0=100%%). Empty = no noise.")
    parser.add_argument("--noise_seed", type=int, default=NOISE_SEED)
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    output_root = Path(args.output_root).resolve()
    device = torch.device(args.device)
    noise_levels = (
        [float(x) for x in args.noise_levels.split(",") if x.strip()]
        if args.noise_levels.strip() else [None]
    )

    run_dirs = sorted(path.parent for path in runs_root.rglob("hparams.json"))

    for noise_level in noise_levels:
        if noise_level is not None:
            level_dir = output_root / f"noise-{noise_level}"
            print(f"\n=== Noise level: {noise_level} ===")
        else:
            level_dir = output_root

        rows = [evaluate_run(run_dir, device, level_dir, noise_level, args.noise_seed) for run_dir in run_dirs]
        df = pl.DataFrame(rows).sort(
            by=["status", "mean_rel_l2", "save_dir"],
            descending=[False, False, False],
            nulls_last=True,
        )
        print(df)

        completed_df = df.filter(pl.col("status") == "ok")
        if completed_df.height > 0:
            best_row = completed_df.sort("mean_rel_l2").row(0, named=True)
            worst_row = completed_df.sort("mean_rel_l2", descending=True).row(0, named=True)
            print(f"\nBest:  {best_row['save_dir']}  (mean_rel_l2={best_row['mean_rel_l2']:.6f})")
            print(f"Worst: {worst_row['save_dir']}  (mean_rel_l2={worst_row['mean_rel_l2']:.6f})")

        if noise_level is not None:
            level_dir.mkdir(parents=True, exist_ok=True)
            df.write_parquet(str(level_dir / "results.parquet"))
            df.write_csv(str(level_dir / "results.csv"))
        elif args.output is not None:
            output_path = Path(args.output)
            if output_path.suffix == ".csv":
                df.write_csv(output_path)
            else:
                df.write_parquet(output_path)


if __name__ == "__main__":
    main()
