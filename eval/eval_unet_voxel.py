import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from scipy.io import loadmat
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, TensorDataset, random_split

from build_3dert_test2 import surface_z
from models.VAE.default import UNetVAE
from models.unet import Unet, Unet3D
from train.encoder.train_unet import invert_normalization, load_sigma_dataset
from eval.noise.noise_model import NOISE_SEED, add_noise_to_loader
from train.encoder.util import save_scatter_slices


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "saved_runs" / "grid_search_full_2"
OUTPUT_ROOT = REPO_ROOT / "eval_outputs" / "grid_search_2"
DATA_PATH = REPO_ROOT / "data" / "3dert_test2.npy"
LOG_EPS = 1e-6
SPLIT_SEED = 159753
RANDOM_SAMPLE_SEED = 42
N_RANDOM_SAMPLES = 5
SURFACE_PARAMS = {"dx_main": 30.0, "dy_main": 15.0, "k": 0.0025}

LINE_COLORS = {"line1": "red", "line2": "green", "line3": "blue"}

SPARSE_ENCODER_CKPT = (
    REPO_ROOT
    / "saved_runs/grid_search_vae_2/_lr2e-04_beta5e-02_kl400_bs128_ep1200_bc8_lc4_sparse/checkpoints/best_val_loss.pt"
)
DENSE_ENCODER_CKPT = (
    REPO_ROOT
    / "saved_runs/grid_search_vae_2/_lr2e-04_beta5e-02_kl400_bs128_ep1200_bc8_lc4/checkpoints/best_val_loss.pt"
)


def normalize_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def find_checkpoint(run_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = run_dir / "checkpoints" / name
        if path.exists():
            return path
    return None


def load_encoder(device: torch.device, sparse: bool) -> UNetVAE:
    encoder_ckpt = SPARSE_ENCODER_CKPT if sparse else DENSE_ENCODER_CKPT
    with (encoder_ckpt.parents[1] / "hparams.json").open("r", encoding="utf-8") as f:
        enc_hparams = json.load(f)
    encoder = UNetVAE(
        base_channels=int(enc_hparams["bc"]),
        latent_channels=int(enc_hparams["lc"]),
    ).to(device)
    checkpoint = torch.load(encoder_ckpt, map_location="cpu")
    ema_state = checkpoint.get("ema_state_dict")
    if ema_state is not None:
        ema_model = AveragedModel(encoder, multi_avg_fn=get_ema_multi_avg_fn(0.999))
        ema_model.load_state_dict(ema_state)
        encoder.load_state_dict(ema_model.module.state_dict())
    else:
        encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def build_test_loader(
    sparse: bool,
    batch_size: int,
    device: torch.device,
    normalization: str = "log",
) -> tuple[DataLoader, np.ndarray, np.ndarray, np.ndarray, torch.Tensor, dict]:
    data = np.load(DATA_PATH, allow_pickle=True).item()
    x = torch.from_numpy(np.asarray(data["X"], dtype=np.float32)).to(device)
    y, xq, yq, zq, norm_stats = load_sigma_dataset(str(DATA_PATH), log_eps=LOG_EPS, sparse=sparse, normalization=normalization)
    y = y.to(device)
    n_samples = x.shape[0]
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
    xx, yy, zz = np.meshgrid(xq, yq, zq, indexing="xy")
    surf = surface_z(xx, yy, SURFACE_PARAMS)
    solid_mask = torch.from_numpy(np.transpose(zz <= surf, (2, 0, 1))).to(
        device=device, dtype=torch.bool
    )
    return test_loader, xq, yq, zq, solid_mask, norm_stats


def build_model(
    hparams: dict,
    device: torch.device,
    encoder: UNetVAE,
    test_loader: DataLoader,
) -> torch.nn.Module:
    pixel_output = normalize_bool(hparams.get("pixel_output", False))
    x_batch, y_batch = next(iter(test_loader))
    in_c = x_batch.shape[1]
    _, out_c, out_d, out_h, out_w = y_batch.shape
    unet_ch = int(hparams["unet_ch"])
    if pixel_output:
        return Unet3D(
            in_channels=in_c,
            out_channels=out_c,
            output_shape=(out_d, out_h, out_w),
            ch=unet_ch,
        ).to(device)
    with torch.no_grad():
        mu, _ = encoder.encode(y_batch[:1].to(device))
    latent_out_c, latent_out_h, latent_out_w = mu.shape[1:]
    return Unet(
        in_channels=in_c,
        out_channels=latent_out_c,
        output_shape=(latent_out_h, latent_out_w),
        ch=unet_ch,
        ch_mul=[1, 2, 2],
    ).to(device)


def relative_metrics_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_flat = mask.reshape(1, -1).expand(pred.shape[0], -1)
    pred_flat = pred.float().reshape(pred.shape[0], -1)
    target_flat = target.float().reshape(target.shape[0], -1)
    pred_masked = pred_flat[mask_flat].reshape(pred.shape[0], -1)
    target_masked = target_flat[mask_flat].reshape(target.shape[0], -1)
    diff = pred_masked - target_masked
    rel_l2 = torch.linalg.vector_norm(diff, ord=2, dim=1) / torch.clamp(
        torch.linalg.vector_norm(target_masked, ord=2, dim=1), min=1e-12,
    )
    rel_l1 = torch.linalg.vector_norm(diff, ord=1, dim=1) / torch.clamp(
        torch.linalg.vector_norm(target_masked, ord=1, dim=1), min=1e-12,
    )
    return rel_l1, rel_l2


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
    """Per-sample MAE (mean over voxels)."""
    return (pred.float() - target.float()).abs().reshape(pred.shape[0], -1).mean(dim=1)


def mean_absolute_error_masked(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample MAE masked to below-surface region."""
    mask_flat = mask.reshape(1, -1).expand(pred.shape[0], -1)
    pred_flat = pred.float().reshape(pred.shape[0], -1)
    target_flat = target.float().reshape(target.shape[0], -1)
    diff = (pred_flat[mask_flat] - target_flat[mask_flat]).abs()
    n_masked = int(mask_flat[0].sum())
    return diff.reshape(pred.shape[0], n_masked).mean(dim=1)


def collect_predictions(
    model: torch.nn.Module,
    encoder: UNetVAE,
    test_loader: DataLoader,
    solid_mask: torch.Tensor,
    pixel_output: bool,
    device: torch.device,
    normalization: str = "log",
    norm_stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if norm_stats is None:
        norm_stats = {}
    pred_list, gt_list = [], []
    bs_rel_l1_list, bs_rel_l2_list, rel_l2_list, rel_l1_list = [], [], [], []
    mae_list, bs_mae_list = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            if not pixel_output:
                pred = encoder.decode(pred)
            # invert normalization before computing metrics so errors are in original sigma space
            pred_sigma = invert_normalization(pred, normalization, norm_stats, LOG_EPS)
            gt_sigma = invert_normalization(y_batch, normalization, norm_stats, LOG_EPS)
            bs_rel_l1, bs_rel_l2 = relative_metrics_masked(pred_sigma, gt_sigma, solid_mask)
            rel_l1, rel_l2 = relative_metrics(pred_sigma, gt_sigma)
            mae = mean_absolute_error(pred_sigma, gt_sigma)
            bs_mae = mean_absolute_error_masked(pred_sigma, gt_sigma, solid_mask)
            pred_list.append(pred_sigma.cpu().numpy())
            gt_list.append(gt_sigma.cpu().numpy())
            bs_rel_l1_list.append(bs_rel_l1.cpu().numpy())
            bs_rel_l2_list.append(bs_rel_l2.cpu().numpy())
            rel_l2_list.append(rel_l2.cpu().numpy())
            rel_l1_list.append(rel_l1.cpu().numpy())
            mae_list.append(mae.cpu().numpy())
            bs_mae_list.append(bs_mae.cpu().numpy())
    return (
        np.concatenate(pred_list, axis=0),
        np.concatenate(gt_list, axis=0),
        np.concatenate(bs_rel_l1_list, axis=0),
        np.concatenate(bs_rel_l2_list, axis=0),
        np.concatenate(rel_l2_list, axis=0),
        np.concatenate(rel_l1_list, axis=0),
        np.concatenate(mae_list, axis=0),
        np.concatenate(bs_mae_list, axis=0),
    )


def fixed_random_indices(test_size: int, n: int, seed: int) -> list[int]:
    return np.random.RandomState(seed).choice(test_size, size=n, replace=False).tolist()


def select_samples(
    metrics: np.ndarray,
    random_indices: list[int],
) -> dict[str, list[int]]:
    n = len(metrics)
    sorted_idx = np.argsort(metrics)
    return {
        "best":   [int(sorted_idx[0])],
        "median": [int(sorted_idx[n // 2])],
        "worst":  [int(sorted_idx[-1])],
        "random": random_indices,
    }


def _sample_points(
    vol: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    mask: np.ndarray,
    max_points: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vals = vol.reshape(-1)[mask]
    x, y, z = xx.reshape(-1)[mask], yy.reshape(-1)[mask], zz.reshape(-1)[mask]
    if vals.size > max_points:
        idx = np.linspace(0, vals.size - 1, max_points).astype(int)
        vals, x, y, z = vals[idx], x[idx], y[idx], z[idx]
    return x, y, z, vals


def _scatter3d_trace(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    s: np.ndarray,
    name: str,
    colorscale: str,
    vlo: float,
    vhi: float,
    colorbar_title: str = "",
) -> go.Scatter3d:
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=2,
            color=s,
            colorscale=colorscale,
            cmin=vlo,
            cmax=vhi,
            showscale=True,
            colorbar=dict(title=colorbar_title, thickness=15, len=0.7),
        ),
        name=name,
    )


def _electrode_traces(overlay: dict) -> list[go.Scatter3d]:
    traces = []
    for line_name, line_data in overlay.items():
        c = LINE_COLORS[line_name]
        e = line_data["electrodes"]
        traces.append(go.Scatter3d(
            x=e[:, 0], y=e[:, 1], z=e[:, 2],
            mode="lines+markers",
            line=dict(color=c, width=4),
            marker=dict(size=4, color=c, symbol="circle"),
            name=f"{line_name} electrodes",
        ))
        for j, (pin, pout) in enumerate(line_data["segments"]):
            traces.append(go.Scatter3d(
                x=[pin[0], pout[0]], y=[pin[1], pout[1]], z=[pin[2], pout[2]],
                mode="lines+markers",
                line=dict(color=c, width=3, dash="dash"),
                marker=dict(size=7, color=c, symbol="diamond", opacity=0.9),
                name=f"{line_name} injection",
                showlegend=(j == 0),
            ))
    return traces


def load_overlay(mat_path: str | None) -> dict | None:
    if not mat_path or not Path(mat_path).exists():
        return None
    u = loadmat(mat_path, squeeze_me=True, struct_as_record=False)["u"]
    overlay = {}
    for line_name in ("line1", "line2", "line3"):
        line = getattr(u, line_name)
        electrodes = np.asarray(line.electrodes_xyz, dtype=np.float32)
        inj = line.inj
        if not isinstance(inj, np.ndarray):
            inj = np.array([inj], dtype=object)
        segs = [
            (np.asarray(it.Iin_xyz, dtype=np.float32).reshape(-1),
             np.asarray(it.Iout_xyz, dtype=np.float32).reshape(-1))
            for it in inj
        ]
        overlay[line_name] = {"electrodes": electrodes, "segments": segs}
    return overlay


def save_scatter_plotly(
    pred_np: np.ndarray,
    gt_np: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
    zq: np.ndarray,
    out_path: str,
    subtitle: str = "",
    max_points: int = 100_000,
    overlay: dict | None = None,
    value_label: str = "σ",
) -> None:
    gt_np = gt_np.squeeze()
    pred_np = pred_np.squeeze()
    err_np = pred_np - gt_np

    zz, yy, xx = np.meshgrid(zq, yq, xq, indexing="ij")
    flat_gt = gt_np.reshape(-1)
    cutoff = max(LOG_EPS, float(np.quantile(flat_gt, 0.05)))
    shared_mask = flat_gt > cutoff

    xg, yg, zg, sg = _sample_points(gt_np,   xx, yy, zz, shared_mask, max_points)
    xp, yp, zp, sp = _sample_points(pred_np, xx, yy, zz, shared_mask, max_points)
    xe, ye, ze, se = _sample_points(err_np,  xx, yy, zz, shared_mask, max_points)

    vmin = float(min(sg.min(), sp.min()))
    vmax = float(max(sg.max(), sp.max()))
    err_abs = float(max(np.abs(se).max(), 1e-8))

    scene = dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
    stem, ext = out_path.rsplit(".", 1)

    for tag, data_traces, title in (
        ("gt",   [_scatter3d_trace(xg, yg, zg, sg, "Ground Truth", "PuBu",   vmin,     vmax,    value_label)],   f"Ground Truth – {subtitle}"),
        ("pred", [_scatter3d_trace(xp, yp, zp, sp, "Prediction",   "PuBu",   vmin,     vmax,    value_label)],   f"Prediction – {subtitle}"),
        ("err",  [_scatter3d_trace(xe, ye, ze, se, "Error",        "RdBu_r", -err_abs, err_abs, "error")],    f"Error (pred − target) – {subtitle}"),
    ):
        elec_traces = _electrode_traces(overlay) if overlay is not None else []
        fig = go.Figure(data_traces + elec_traces)
        fig.update_layout(title=title, scene=scene)
        fig.write_html(f"{stem}_{tag}.{ext}", include_plotlyjs="cdn")


def save_sample(
    idx: int,
    label: str,
    pred_all: np.ndarray,
    gt_all: np.ndarray,
    bs_rel_l1: np.ndarray,
    bs_rel_l2: np.ndarray,
    rel_l2: np.ndarray,
    rel_l1: np.ndarray,
    mae: np.ndarray,
    bs_mae: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
    zq: np.ndarray,
    out_dir: Path,
    overlay: dict | None = None,
    value_label: str = "σ",
    noise_level: float | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_idx{idx:04d}_bsrel{bs_rel_l2[idx]:.4f}"
    subtitle = (
        f"rel_l2={rel_l2[idx]:.4f}  rel_l1={rel_l1[idx]:.4f}  MAE={mae[idx]:.6f}"
        f"  bs_rel_l2={bs_rel_l2[idx]:.4f}  bs_rel_l1={bs_rel_l1[idx]:.4f}  bs_MAE={bs_mae[idx]:.6f}"
    )
    if noise_level is not None:
        subtitle = f"noise={noise_level}  " + subtitle
    # pred_all / gt_all are in physical sigma space (unnormalized)
    pred_t = torch.from_numpy(pred_all[idx]).unsqueeze(0)
    gt_t = torch.from_numpy(gt_all[idx]).unsqueeze(0)
    save_scatter_slices(
        target=gt_t,
        pred=pred_t,
        out_path=str(out_dir / f"{stem}.png"),
        xq=xq,
        yq=yq,
        zq=zq,
        sigma_min=LOG_EPS,
        subtitle=subtitle,
        colorbar_label=value_label,
    )
    save_scatter_plotly(
        pred_np=pred_all[idx],
        gt_np=gt_all[idx],
        xq=xq,
        yq=yq,
        zq=zq,
        out_path=str(out_dir / f"{stem}.html"),
        subtitle=subtitle,
        overlay=overlay,
        value_label=value_label,
    )


def eval_run(
    run_dir: Path,
    output_root: Path,
    device: torch.device,
    overlay: dict | None = None,
    noise_level: float | None = None,
    noise_seed: int = NOISE_SEED,
) -> dict | None:
    hparams_path = run_dir / "hparams.json"
    with hparams_path.open("r", encoding="utf-8") as f:
        hparams = json.load(f)

    sparse = normalize_bool(hparams.get("sparse", False))
    pixel_output = normalize_bool(hparams.get("pixel_output", False))
    normalization = hparams.get("normalization", "log")
    norm_stats = hparams.get("norm_stats", {})
    best_ckpt = find_checkpoint(run_dir, ("best_val.pt", "best_val_loss.pt"))
    if best_ckpt is None:
        print(f"  [skip] no best checkpoint in {run_dir.name}")
        return None

    run_name = Path(hparams.get("save_dir", run_dir.name)).name
    out_dir = output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  evaluating {run_name} ...")
    test_loader, xq, yq, zq, solid_mask, _ = build_test_loader(
        sparse=sparse,
        batch_size=int(hparams["batch_size"]),
        device=device,
        normalization=normalization,
    )
    if noise_level is not None and noise_level > 0:
        test_loader = add_noise_to_loader(test_loader, noise_level, seed=noise_seed)
    encoder = load_encoder(device=device, sparse=sparse)
    model = build_model(hparams=hparams, device=device, encoder=encoder, test_loader=test_loader)
    checkpoint = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pred_all, gt_all, bs_rel_l1, bs_rel_l2, rel_l2, rel_l1, mae, bs_mae = collect_predictions(
        model=model,
        encoder=encoder,
        test_loader=test_loader,
        solid_mask=solid_mask,
        pixel_output=pixel_output,
        device=device,
        normalization=normalization,
        norm_stats=norm_stats,
    )

    # Random indices fixed by test-set size — same samples across all runs
    rand_idx = fixed_random_indices(len(bs_rel_l2), N_RANDOM_SAMPLES, RANDOM_SAMPLE_SEED)
    samples = select_samples(bs_rel_l2, random_indices=rand_idx)

    # Label colorbars according to the space being plotted
    value_label = "σ (S/m)"

    # Map each label to its own subdirectory and filename prefix
    subdir_map = {
        "best":   (out_dir / "best",        "best"),
        "median": (out_dir / "median",      "median"),
        "worst":  (out_dir / "worst",       "worst"),
        "random": (out_dir / "fixed_evals", "fixed"),
    }

    for label, indices in samples.items():
        base_dir, label_prefix = subdir_map[label]
        for enum_i, idx in enumerate(indices):
            # Each sample gets its own subdirectory
            if label == "random":
                sample_dir = base_dir / f"fixed_{enum_i}"
            else:
                sample_dir = base_dir
            save_sample(
                idx=idx,
                label=label_prefix,
                pred_all=pred_all,
                gt_all=gt_all,
                bs_rel_l1=bs_rel_l1,
                bs_rel_l2=bs_rel_l2,
                rel_l2=rel_l2,
                rel_l1=rel_l1,
                mae=mae,
                bs_mae=bs_mae,
                xq=xq,
                yq=yq,
                zq=zq,
                out_dir=sample_dir,
                overlay=overlay,
                value_label=value_label,
                noise_level=noise_level,
            )

    summary = {
        "run": run_name,
        "noise_level": noise_level,
        "best_idx": samples["best"][0],
        "median_idx": samples["median"][0],
        "worst_idx": samples["worst"][0],
        "random_indices": samples["random"],
        "best_bs_rel_l2": float(bs_rel_l2[samples["best"][0]]),
        "median_bs_rel_l2": float(bs_rel_l2[samples["median"][0]]),
        "worst_bs_rel_l2": float(bs_rel_l2[samples["worst"][0]]),
        "mean_bs_rel_l2": float(bs_rel_l2.mean()),
        "best_bs_rel_l1": float(bs_rel_l1[samples["best"][0]]),
        "median_bs_rel_l1": float(bs_rel_l1[samples["median"][0]]),
        "worst_bs_rel_l1": float(bs_rel_l1[samples["worst"][0]]),
        "mean_bs_rel_l1": float(bs_rel_l1.mean()),
        "mean_mae": float(mae.mean()),
        "mean_bs_mae": float(bs_mae.mean()),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> saved to {out_dir}  (mean_bs_rel_l2={summary['mean_bs_rel_l2']:.4f})")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default=str(RUNS_ROOT))
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mat", type=str, default=str(REPO_ROOT / "data" / "sample1.mat"), help="Path to sample*.mat for electrode overlay (pass empty string to disable)")
    parser.add_argument("--noise_levels", type=str, default="",
                        help="Comma-separated noise levels (fraction, 1.0=100%%). Empty = no noise.")
    parser.add_argument("--noise_seed", type=int, default=NOISE_SEED)
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    output_root = Path(args.output_root).resolve()
    device = torch.device(args.device)
    overlay = load_overlay(args.mat)
    noise_levels = (
        [float(x) for x in args.noise_levels.split(",") if x.strip()]
        if args.noise_levels.strip() else [None]
    )

    run_dirs = sorted(p.parent for p in runs_root.rglob("hparams.json"))
    print(f"Found {len(run_dirs)} run(s) under {runs_root}")

    # collect per-run, per-noise-level metrics for the sweep summary
    # keyed by run name -> list of {noise_level, mean_bs_rel_l2, mean_bs_rel_l1}
    sweep: dict[str, list[dict]] = {}

    for noise_level in noise_levels:
        if noise_level is not None:
            level_dir = output_root / f"noise-{noise_level}"
            print(f"\n=== Noise level: {noise_level} ===")
        else:
            level_dir = output_root

        run_summaries = []
        for run_dir in run_dirs:
            s = eval_run(
                run_dir=run_dir, output_root=level_dir, device=device,
                overlay=overlay, noise_level=noise_level, noise_seed=args.noise_seed,
            )
            if s is not None:
                run_summaries.append(s)
                sweep.setdefault(s["run"], []).append({
                    "noise_level": noise_level,
                    "mean_bs_rel_l2": s["mean_bs_rel_l2"],
                    "mean_bs_rel_l1": s["mean_bs_rel_l1"],
                    "mean_mae": s["mean_mae"],
                    "mean_bs_mae": s["mean_bs_mae"],
                })

        if noise_level is not None and run_summaries:
            sorted_l2 = sorted(run_summaries, key=lambda r: r["mean_bs_rel_l2"])
            sorted_l1 = sorted(run_summaries, key=lambda r: r["mean_bs_rel_l1"])
            print(f"  Best (L2) at noise {noise_level}: {sorted_l2[0]['run']}  (mean_bs_rel_l2={sorted_l2[0]['mean_bs_rel_l2']:.4f})")
            print(f"  Best (L1) at noise {noise_level}: {sorted_l1[0]['run']}  (mean_bs_rel_l1={sorted_l1[0]['mean_bs_rel_l1']:.4f})")
            summary_l2 = {
                "noise_level": noise_level,
                "metric": "mean_bs_rel_l2",
                "n_runs": len(run_summaries),
                "ranking": [
                    {"run": r["run"], "mean_bs_rel_l2": r["mean_bs_rel_l2"]}
                    for r in sorted_l2
                ],
            }
            summary_l1 = {
                "noise_level": noise_level,
                "metric": "mean_bs_rel_l1",
                "n_runs": len(run_summaries),
                "ranking": [
                    {"run": r["run"], "mean_bs_rel_l1": r["mean_bs_rel_l1"]}
                    for r in sorted_l1
                ],
            }
            with (level_dir / "noise_summary_l2.json").open("w", encoding="utf-8") as f:
                json.dump(summary_l2, f, indent=2)
            with (level_dir / "noise_summary_l1.json").open("w", encoding="utf-8") as f:
                json.dump(summary_l1, f, indent=2)

    # write cross-noise sweep summary
    if sweep:
        noise_sweep = {
            run_name: sorted(entries, key=lambda e: e["noise_level"] or -1)
            for run_name, entries in sweep.items()
        }
        with (output_root / "noise_sweep.json").open("w", encoding="utf-8") as f:
            json.dump(noise_sweep, f, indent=2)

    print(f"\nDone. Results in {output_root}")


if __name__ == "__main__":
    main()
