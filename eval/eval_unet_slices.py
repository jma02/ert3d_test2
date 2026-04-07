import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval.eval_unet import (
    LOG_EPS,
    REPO_ROOT,
    RUNS_ROOT,
    build_model,
    build_test_loader,
    find_checkpoint,
    load_encoder,
    normalize_bool,
    relative_metrics,
    relative_metrics_masked,
    resolve_data_path,
)


DEFAULT_RUN_DIR = RUNS_ROOT / "_dense_latent_only_lr5e-02_bs64_ep500_unet_ch32_latent"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "eval_outputs"


def load_hparams_for_checkpoint(checkpoint_path: Path) -> tuple[Path, dict]:
    run_dir = checkpoint_path.resolve().parents[1]
    hparams_path = run_dir / "hparams.json"
    with hparams_path.open("r", encoding="utf-8") as f:
        hparams = json.load(f)
    return run_dir, hparams


def resolve_checkpoint_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_dir():
        checkpoint_path = find_checkpoint(resolved, ("best_val.pt", "best_val_loss.pt"))
        if checkpoint_path is None:
            raise FileNotFoundError(f"No best checkpoint found under {resolved / 'checkpoints'}")
        return checkpoint_path
    return resolved


def collect_predictions(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    run_dir, hparams = load_hparams_for_checkpoint(checkpoint_path)
    sparse = normalize_bool(hparams["sparse"])
    pixel_output = normalize_bool(hparams["pixel_output"])
    data_path = resolve_data_path(hparams["data"])
    test_loader, _, _, solid_mask = build_test_loader(
        data_path=data_path,
        sparse=sparse,
        batch_size=int(hparams["batch_size"]),
        device=device,
    )

    encoder = load_encoder(device=device, sparse=sparse)
    model = build_model(hparams=hparams, device=device, encoder=encoder, test_loader=test_loader)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pred_list = []
    gt_list = []
    rel_l2_list = []
    below_surface_rel_l2_list = []
    x_list = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            if not pixel_output:
                pred = encoder.decode(pred)
            _, rel_l2 = relative_metrics(pred, y_batch)
            _, below_surface_rel_l2 = relative_metrics_masked(pred, y_batch, solid_mask)
            pred_list.append(pred.detach().cpu().numpy())
            gt_list.append(y_batch.detach().cpu().numpy())
            rel_l2_list.append(rel_l2.detach().cpu().numpy())
            below_surface_rel_l2_list.append(below_surface_rel_l2.detach().cpu().numpy())
            x_list.append(x_batch.detach().cpu().numpy())

    pred_arr = np.concatenate(pred_list, axis=0)
    gt_arr = np.concatenate(gt_list, axis=0)
    rel_l2_arr = np.concatenate(rel_l2_list, axis=0)
    below_surface_rel_l2_arr = np.concatenate(below_surface_rel_l2_list, axis=0)
    x_arr = np.concatenate(x_list, axis=0)
    return hparams, x_arr, gt_arr, pred_arr, rel_l2_arr, below_surface_rel_l2_arr


def plot_sample_slices(
    gt_volume: np.ndarray,
    pred_volume: np.ndarray,
    output_dir: Path,
    sample_index: int,
    rel_l2: float,
    below_surface_rel_l2: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_volume = np.squeeze(gt_volume)
    pred_volume = np.squeeze(pred_volume)
    abs_err = np.abs(pred_volume - gt_volume)

    shared_vmin = float(min(gt_volume.min(), pred_volume.min()))
    shared_vmax = float(max(gt_volume.max(), pred_volume.max()))
    err_vmax = float(abs_err.max())

    n_slices = gt_volume.shape[1]
    for slice_idx in range(n_slices):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        panels = [
            (gt_volume[:, slice_idx, :], "GT", "viridis", shared_vmin, shared_vmax),
            (pred_volume[:, slice_idx, :], "Pred", "viridis", shared_vmin, shared_vmax),
            (abs_err[:, slice_idx, :], "Error", "magma", 0.0, err_vmax),
        ]
        for ax, (values, title, cmap, vmin, vmax) in zip(axes, panels, strict=True):
            im = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"Slice {slice_idx:03d} {title}")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"Sample {sample_index} | rel_l2={rel_l2:.6f} | below_surface_rel_l2={below_surface_rel_l2:.6f}",
            fontsize=12,
        )
        fig.savefig(output_dir / f"slice_{slice_idx:03d}.png", dpi=150)
        plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_index", type=int, default=9)
    parser.add_argument("--selection_metric", choices=("below_surface_rel_l2", "rel_l2"), default="below_surface_rel_l2")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(Path(args.checkpoint))
    device = torch.device(args.device)
    hparams, x_arr, gt_arr, pred_arr, rel_l2_arr, below_surface_rel_l2_arr = collect_predictions(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    if args.sample_index is None:
        metric = below_surface_rel_l2_arr if args.selection_metric == "below_surface_rel_l2" else rel_l2_arr
        sample_index = int(np.argmin(metric))
    else:
        sample_index = args.sample_index

    run_name = Path(hparams.get("save_dir", checkpoint_path.parents[1].name)).name
    output_dir = Path(args.output_root).resolve() / f"{run_name}_best_case_slices"
    plot_sample_slices(
        gt_volume=gt_arr[sample_index],
        pred_volume=pred_arr[sample_index],
        output_dir=output_dir,
        sample_index=sample_index,
        rel_l2=float(rel_l2_arr[sample_index]),
        below_surface_rel_l2=float(below_surface_rel_l2_arr[sample_index]),
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "data": hparams["data"],
        "sample_index": sample_index,
        "rel_l2": float(rel_l2_arr[sample_index]),
        "below_surface_rel_l2": float(below_surface_rel_l2_arr[sample_index]),
        "selection_metric": args.selection_metric,
        "input_shape": list(x_arr[sample_index].shape),
        "volume_shape": list(gt_arr[sample_index].shape),
        "output_dir": str(output_dir),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved slice plots to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
