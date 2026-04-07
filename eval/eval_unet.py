import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, TensorDataset, random_split

from build_3dert_test2 import surface_z
from models.VAE.default import UNetVAE
from models.unet import Unet, Unet3D
from eval.noise.noise_model import NOISE_SEED, add_noise_to_loader
from train.encoder.train_unet import invert_normalization, load_sigma_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "saved_runs" / "grid_search_full"
LOG_EPS = 1e-6
SPLIT_SEED = 159753
SPARSE_ENCODER_CKPT = (
    REPO_ROOT
    / "saved_runs/grid_search_vae_2/_lr2e-04_beta5e-02_kl400_bs128_ep1200_bc8_lc4_sparse/checkpoints/best_val_loss.pt"
)
DENSE_ENCODER_CKPT = (
    REPO_ROOT
    / "saved_runs/grid_search_vae_2/_lr2e-04_beta5e-02_kl400_bs128_ep1200_bc8_lc4/checkpoints/best_val_loss.pt"
)
SURFACE_PARAMS = {"dx_main": 30.0, "dy_main": 15.0, "k": 0.0025}


def normalize_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def resolve_data_path(data_path: str) -> Path:
    path = Path(data_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


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
    data_path: Path,
    sparse: bool,
    batch_size: int,
    device: torch.device,
    normalization: str = "log",
) -> tuple[DataLoader, tuple[int, int, int], tuple[int, int, int, int], torch.Tensor, dict]:
    data = np.load(data_path, allow_pickle=True).item()
    x = torch.from_numpy(np.asarray(data["X"], dtype=np.float32)).to(device)
    y, xq, yq, zq, norm_stats = load_sigma_dataset(str(data_path), log_eps=LOG_EPS, sparse=sparse, normalization=normalization)
    y = y.to(device)
    n_samples, in_c, in_h, in_w = x.shape
    _, out_c, out_d, out_h, out_w = y.shape
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
    solid_mask = torch.from_numpy(np.transpose(zz <= surf, (2, 0, 1))).to(device=device, dtype=torch.bool)
    return test_loader, (in_c, in_h, in_w), (out_c, out_d, out_h, out_w), solid_mask, norm_stats


def build_model(
    hparams: dict,
    device: torch.device,
    encoder: UNetVAE,
    test_loader: DataLoader,
):
    sparse = normalize_bool(hparams["sparse"])
    pixel_output = normalize_bool(hparams["pixel_output"])
    batch = next(iter(test_loader))[1][:1].to(device)
    with torch.no_grad():
        mu, _ = encoder.encode(batch)
    latent_out_c, latent_out_h, latent_out_w = mu.shape[1:]
    data_path = resolve_data_path(hparams["data"])
    _, input_shape, output_shape, _, _ = build_test_loader(
        data_path=data_path,
        sparse=sparse,
        batch_size=int(hparams["batch_size"]),
        device=device,
    )
    in_c, _, _ = input_shape
    out_c, out_d, out_h, out_w = output_shape
    unet_ch = int(hparams["unet_ch"])
    if pixel_output:
        return Unet3D(
            in_channels=in_c,
            out_channels=out_c,
            output_shape=(out_d, out_h, out_w),
            ch=unet_ch,
        ).to(device)
    return Unet(
        in_channels=in_c,
        out_channels=latent_out_c,
        output_shape=(latent_out_h, latent_out_w),
        ch=unet_ch,
        ch_mul=[1, 2, 2],
    ).to(device)


def relative_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_flat = pred.float().reshape(pred.shape[0], -1)
    target_flat = target.float().reshape(target.shape[0], -1)
    diff = pred_flat - target_flat
    rel_l2 = torch.linalg.vector_norm(diff, ord=2, dim=1) / torch.clamp(
        torch.linalg.vector_norm(target_flat, ord=2, dim=1),
        min=1e-12,
    )
    rel_l1 = torch.linalg.vector_norm(diff, ord=1, dim=1) / torch.clamp(
        torch.linalg.vector_norm(target_flat, ord=1, dim=1),
        min=1e-12,
    )
    return rel_l1, rel_l2


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
        torch.linalg.vector_norm(target_masked, ord=2, dim=1),
        min=1e-12,
    )
    rel_l1 = torch.linalg.vector_norm(diff, ord=1, dim=1) / torch.clamp(
        torch.linalg.vector_norm(target_masked, ord=1, dim=1),
        min=1e-12,
    )
    return rel_l1, rel_l2


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


def find_checkpoint(run_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = run_dir / "checkpoints" / name
        if path.exists():
            return path
    return None


def evaluate_run(
    run_dir: Path,
    device: torch.device,
    noise_level: float | None = None,
    noise_seed: int = NOISE_SEED,
) -> dict[str, object]:
    hparams_path = run_dir / "hparams.json"
    with hparams_path.open("r", encoding="utf-8") as f:
        hparams = json.load(f)

    sparse = normalize_bool(hparams["sparse"])
    antipodal_loss = normalize_bool(hparams["antipodal_loss"])
    pixel_output = normalize_bool(hparams["pixel_output"])
    best_ckpt = find_checkpoint(run_dir, ("best_val.pt", "best_val_loss.pt"))
    final_ckpt = find_checkpoint(run_dir, ("final_model.pt", "final.pt"))

    row: dict[str, object] = {
        "run_dir": str(run_dir),
        "save_dir": hparams.get("save_dir", run_dir.name),
        "noise_level": noise_level,
        "sparse": sparse,
        "antipodal_loss": antipodal_loss,
        "pixel_output": pixel_output,
        "lr": float(hparams["lr"]),
        "batch_size": int(hparams["batch_size"]),
        "epochs": int(hparams["epochs"]),
        "unet_ch": int(hparams["unet_ch"]),
        "best_checkpoint": str(best_ckpt) if best_ckpt is not None else None,
        "final_checkpoint": str(final_ckpt) if final_ckpt is not None else None,
        "training_done": best_ckpt is not None and final_ckpt is not None,
        "status": "ok" if best_ckpt is not None and final_ckpt is not None else "training_not_done",
    }

    row.update(summarize([], "rel_l1"))
    row.update(summarize([], "rel_l2"))
    row.update(summarize([], "below_surface_rel_l1"))
    row.update(summarize([], "below_surface_rel_l2"))

    if best_ckpt is None or final_ckpt is None:
        return row

    normalization = hparams.get("normalization", "log")
    norm_stats = hparams.get("norm_stats", {})
    data_path = resolve_data_path(hparams["data"])
    test_loader, _, _, solid_mask, _ = build_test_loader(
        data_path=data_path,
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

    rel_l1_values: list[float] = []
    rel_l2_values: list[float] = []
    below_surface_rel_l1_values: list[float] = []
    below_surface_rel_l2_values: list[float] = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            if not pixel_output:
                pred = encoder.decode(pred)
            # invert normalization so metrics are in original sigma space
            pred_sigma = invert_normalization(pred, normalization, norm_stats, LOG_EPS)
            gt_sigma = invert_normalization(y_batch, normalization, norm_stats, LOG_EPS)
            rel_l1, rel_l2 = relative_metrics(pred_sigma, gt_sigma)
            below_surface_rel_l1, below_surface_rel_l2 = relative_metrics_masked(pred_sigma, gt_sigma, solid_mask)
            rel_l1_values.extend(rel_l1.detach().cpu().tolist())
            rel_l2_values.extend(rel_l2.detach().cpu().tolist())
            below_surface_rel_l1_values.extend(below_surface_rel_l1.detach().cpu().tolist())
            below_surface_rel_l2_values.extend(below_surface_rel_l2.detach().cpu().tolist())

    row.update(summarize(rel_l1_values, "rel_l1"))
    row.update(summarize(rel_l2_values, "rel_l2"))
    row.update(summarize(below_surface_rel_l1_values, "below_surface_rel_l1"))
    row.update(summarize(below_surface_rel_l2_values, "below_surface_rel_l2"))
    return row


def main() -> None:
    from eval.noise.noise_model import DEFAULT_NOISE_LEVELS

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default=str(RUNS_ROOT))
    parser.add_argument("--output_root", type=str, default=None,
                        help="Output directory for results (used with --noise_levels)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None,
                        help="Single output file path (csv/parquet) for non-noise mode")
    parser.add_argument("--noise_levels", type=str, default="",
                        help="Comma-separated noise levels (fraction, 1.0=100%%). Empty = no noise.")
    parser.add_argument("--noise_seed", type=int, default=NOISE_SEED)
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    device = torch.device(args.device)
    noise_levels = (
        [float(x) for x in args.noise_levels.split(",") if x.strip()]
        if args.noise_levels.strip() else [None]
    )
    run_dirs = sorted(path.parent for path in runs_root.rglob("hparams.json"))

    for noise_level in noise_levels:
        if noise_level is not None:
            print(f"\n=== Noise level: {noise_level} ===")

        rows = [evaluate_run(run_dir, device, noise_level, args.noise_seed) for run_dir in run_dirs]
        df = pl.DataFrame(rows).sort(
            by=["status", "mean_below_surface_rel_l2", "save_dir"],
            descending=[False, False, False],
            nulls_last=True,
        )
        print(df)
        completed_df = df.filter(pl.col("status") == "ok")
        for sparse_value, label in ((True, "Sparse"), (False, "Dense")):
            dataset_df = completed_df.filter(pl.col("sparse") == sparse_value)
            if dataset_df.height == 0:
                continue
            best_row = dataset_df.sort("mean_below_surface_rel_l2").row(0, named=True)
            worst_row = dataset_df.sort("mean_below_surface_rel_l2", descending=True).row(0, named=True)
            print(
                f"{label} best config: {best_row['save_dir']} "
                f"(mean_below_surface_rel_l2={best_row['mean_below_surface_rel_l2']:.6f})"
            )
            print(
                f"{label} worst config: {worst_row['save_dir']} "
                f"(mean_below_surface_rel_l2={worst_row['mean_below_surface_rel_l2']:.6f})"
            )

        # Save results
        if noise_level is not None and args.output_root:
            out_dir = Path(args.output_root) / f"noise-{noise_level}"
            out_dir.mkdir(parents=True, exist_ok=True)
            df.write_parquet(str(out_dir / "results.parquet"))
            df.write_csv(str(out_dir / "results.csv"))
        elif args.output is not None:
            output_path = Path(args.output)
            if output_path.suffix == ".csv":
                df.write_csv(output_path)
            else:
                df.write_parquet(output_path)


if __name__ == "__main__":
    main()
