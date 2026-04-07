"""Evaluate specific sample indices across noise levels for two_inclusions OOD.

Only produces fixed_evals plots (no best/median/worst, no full sweep).
Output structure is compatible with the viewer manifest builder:
  {output_root}/{model_name}/noise-{level}/fixed_evals/fixed_{i}/{plot}.png

Usage (easy):
    PYTHONPATH=. uv run python eval/ood/eval_fixed_samples.py \
        --mode easy --device cuda:1 --indices 100,200,400,600

Usage (voxel):
    PYTHONPATH=. uv run python eval/ood/eval_fixed_samples.py \
        --mode voxel --device cuda:0 --indices 100,200,400,600
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from eval.noise.noise_model import NOISE_SEED, add_noise_to_loader

REPO_ROOT = Path(__file__).resolve().parents[2]
NOISE_LEVELS = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

# ── easy helpers ─────────────────────────────────────────────────────────────

EASY_RUNS_ROOT = REPO_ROOT / "saved_runs" / "grid_search_no_encoder"
EASY_DATA_PATH = REPO_ROOT / "data" / "3dert_test2_easy_test_set.npy"
LOG_EPS = 1e-6
EASY_SUBSET = {"two_inclusions": (1000, 2000)}


def build_easy_test_data(subset: str, device: torch.device):
    start, end = EASY_SUBSET[subset]
    data = np.load(EASY_DATA_PATH, allow_pickle=True).item()
    xq, zq = data["xq"], data["zq"]
    x = torch.from_numpy(np.asarray(data["X"][start:end], dtype=np.float32)).to(device)
    y = torch.from_numpy(np.asarray(data["Y_easy"][start:end], dtype=np.float32)).to(device)
    y = torch.log(torch.clamp(y, min=LOG_EPS))
    return x, y, xq, zq


def run_easy(args):
    from eval.ood.eval_ood_easy import plot_sample_slices, relative_metrics
    from models.unet import Unet
    from train.util import PLOT_FONT_FAMILY

    device = torch.device(args.device)
    indices = [int(x) for x in args.indices.split(",")]
    output_root = Path(args.output_root)

    x_all, y_all, xq, zq = build_easy_test_data("two_inclusions", device)
    n_samples, in_c, in_h, in_w = x_all.shape
    _, out_c, out_h, out_w = y_all.shape

    run_dirs = sorted(p.parent for p in EASY_RUNS_ROOT.rglob("hparams.json"))
    print(f"Found {len(run_dirs)} runs, evaluating indices {indices}")

    for run_dir in run_dirs:
        hparams = json.loads((run_dir / "hparams.json").read_text())
        run_name = Path(hparams.get("save_dir", run_dir.name)).name

        best_ckpt = run_dir / "checkpoints" / "best_val.pt"
        if not best_ckpt.exists():
            best_ckpt = run_dir / "checkpoints" / "best_val_loss.pt"
        if not best_ckpt.exists():
            continue

        model = Unet(
            in_channels=in_c, out_channels=out_c,
            output_shape=(out_h, out_w), ch=int(hparams["unet_ch"]),
        ).to(device)
        ckpt = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        for noise_level in NOISE_LEVELS:
            # Apply noise to inputs
            if noise_level > 0:
                loader = DataLoader(TensorDataset(x_all, y_all), batch_size=len(x_all), shuffle=False)
                loader = add_noise_to_loader(loader, noise_level, seed=NOISE_SEED)
                x_noised, y_batch = next(iter(loader))
            else:
                x_noised, y_batch = x_all, y_all

            with torch.no_grad():
                pred = model(x_noised)

            pred_sigma = torch.exp(pred) - LOG_EPS
            gt_sigma = torch.exp(y_batch) - LOG_EPS
            rel_l1, rel_l2 = relative_metrics(pred_sigma, gt_sigma)

            gt_np = gt_sigma.cpu().numpy()
            pred_np = pred_sigma.cpu().numpy()
            rel_l2_np = rel_l2.cpu().numpy()
            rel_l1_np = rel_l1.cpu().numpy()

            for enum_i, idx in enumerate(indices):
                sample_dir = output_root / run_name / f"noise-{noise_level}" / "fixed_evals" / f"fixed_{enum_i}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                plot_sample_slices(
                    gt=gt_np[idx], pred=pred_np[idx],
                    out_path=sample_dir / f"fixed_idx{idx:04d}_rel_l2{rel_l2_np[idx]:.4f}.png",
                    sample_index=idx,
                    rel_l2=float(rel_l2_np[idx]), rel_l1=float(rel_l1_np[idx]),
                    xq=xq, zq=zq, noise_level=noise_level, subset_label="two_inclusions",
                )

        print(f"  {run_name}: done")


# ── voxel helpers ────────────────────────────────────────────────────────────

VOXEL_RUNS_ROOT = REPO_ROOT / "saved_runs" / "grid_search_full_2"
VOXEL_DATA_PATH = REPO_ROOT / "data" / "3dert_test2_test_set.npy"


def run_voxel(args):
    from eval.ood.eval_ood_voxel import (
        build_ood_test_loader, collect_predictions, load_encoder,
        build_model, save_sample, load_overlay, find_checkpoint,
        normalize_bool, fixed_random_indices, select_samples,
        RANDOM_SAMPLE_SEED, N_RANDOM_SAMPLES,
    )

    device = torch.device(args.device)
    indices = [int(x) for x in args.indices.split(",")]
    output_root = Path(args.output_root)
    overlay = load_overlay(str(REPO_ROOT / "data" / "sample1.mat"))

    run_dirs = sorted(p.parent for p in VOXEL_RUNS_ROOT.rglob("hparams.json"))
    print(f"Found {len(run_dirs)} runs, evaluating indices {indices}")

    for run_dir in run_dirs:
        hparams = json.loads((run_dir / "hparams.json").read_text())
        sparse = normalize_bool(hparams.get("sparse", False))
        pixel_output = normalize_bool(hparams.get("pixel_output", False))
        normalization = hparams.get("normalization", "log")
        norm_stats = hparams.get("norm_stats", {})

        best_ckpt = find_checkpoint(run_dir, ("best_val.pt", "best_val_loss.pt"))
        if best_ckpt is None:
            continue

        run_name = Path(hparams.get("save_dir", run_dir.name)).name

        for noise_level in NOISE_LEVELS:
            test_loader, xq, yq, zq, solid_mask, _ = build_ood_test_loader(
                subset="two_inclusions", sparse=sparse,
                batch_size=int(hparams["batch_size"]),
                device=device, normalization=normalization,
            )
            if noise_level > 0:
                test_loader = add_noise_to_loader(test_loader, noise_level, seed=NOISE_SEED)

            encoder = load_encoder(device=device, sparse=sparse)
            model = build_model(hparams=hparams, device=device, encoder=encoder, test_loader=test_loader)
            ckpt = torch.load(best_ckpt, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            pred_all, gt_all, bs_rel_l1, bs_rel_l2, rel_l2, rel_l1, mae, bs_mae = collect_predictions(
                model=model, encoder=encoder, test_loader=test_loader,
                solid_mask=solid_mask, pixel_output=pixel_output,
                device=device, normalization=normalization, norm_stats=norm_stats,
            )

            for enum_i, idx in enumerate(indices):
                sample_dir = output_root / run_name / f"noise-{noise_level}" / "fixed_evals" / f"fixed_{enum_i}"
                save_sample(
                    idx=idx, label="fixed",
                    pred_all=pred_all, gt_all=gt_all,
                    bs_rel_l1=bs_rel_l1, bs_rel_l2=bs_rel_l2,
                    rel_l2=rel_l2, rel_l1=rel_l1, mae=mae, bs_mae=bs_mae,
                    xq=xq, yq=yq, zq=zq, out_dir=sample_dir,
                    overlay=overlay, value_label="σ (S/m)",
                    noise_level=noise_level, subset_label="two_inclusions",
                )

        print(f"  {run_name}: done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["easy", "voxel"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--indices", type=str, default="100,200,400,600")
    parser.add_argument("--output_root", type=str, default=None)
    args = parser.parse_args()

    if args.output_root is None:
        if args.mode == "easy":
            args.output_root = str(REPO_ROOT / "eval_outputs" / "ood_two_inclusions_easy-custom-fixed_evals")
        else:
            args.output_root = str(REPO_ROOT / "eval_outputs" / "ood_two_inclusions-custom-fixed_evals")

    print(f"Mode: {args.mode}, device: {args.device}")
    print(f"Output: {args.output_root}")

    if args.mode == "easy":
        run_easy(args)
    else:
        run_voxel(args)

    print(f"\nDone. Results in {args.output_root}")


if __name__ == "__main__":
    main()
