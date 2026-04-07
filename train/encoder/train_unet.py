import argparse
import os
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch import nn

from models.unet import Unet, Unet3D
from models.VAE.default import UNetVAE
from train.util import TVHuberLoss3D
from train.encoder.util import plot_encoder_loss_curves, save_scatter_slices


torch.manual_seed(159753)
np.random.seed(159753)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

def load_sigma_dataset(
    path: str,
    log_eps: float,
    sparse: bool = False,
    normalization: str = "log",
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=True).item()
    sigma = data["Y"].astype(np.float32, copy=False)
    sigma = sigma[:, None, ...]
    xq = data["xq"].astype(np.float32, copy=False)
    yq = data["yq"].astype(np.float32, copy=False)
    zq = data["zq"].astype(np.float32, copy=False)
    norm_stats: dict = {}
    if normalization == "log":
        sigma = np.log(sigma + log_eps)
    elif normalization == "minmax":
        sigma_min = float(sigma.min())
        sigma_max = float(sigma.max())
        sigma = (sigma - sigma_min) / max(sigma_max - sigma_min, 1e-8)
        norm_stats = {"sigma_min": sigma_min, "sigma_max": sigma_max}
    if sparse:
        sigma = sigma[:, :, :, ::2, :]
        yq = yq[::2]
    return torch.from_numpy(sigma), xq, yq, zq, norm_stats


def invert_normalization(
    tensor: torch.Tensor,
    normalization: str,
    norm_stats: dict,
    log_eps: float = 1e-6,
) -> torch.Tensor:
    if normalization == "log":
        return torch.exp(tensor) - log_eps
    elif normalization == "minmax":
        return tensor * (norm_stats["sigma_max"] - norm_stats["sigma_min"]) + norm_stats["sigma_min"]
    return tensor


def compute_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    encoder: UNetVAE,
    pixel_output: bool,
    antipodal_loss: bool,
    pixel_loss_fn: nn.Module,
    latent_loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    total = torch.zeros((), device=pred.device)
    pixel_loss = None
    latent_loss = None

    if pixel_output:
        pixel_loss = pixel_loss_fn(pred, target)
        total += pixel_loss
        if antipodal_loss:
            with torch.no_grad():
                mu_target, _ = encoder.encode(target)
            mu_pred, _ = encoder.encode(pred)
            latent_loss = latent_loss_fn(mu_pred, mu_target)
            total += latent_loss
    else:
        with torch.no_grad():
            mu_target, _ = encoder.encode(target)
        latent_loss = latent_loss_fn(pred, mu_target)
        total += latent_loss
        if antipodal_loss:
            recon = encoder.decode(pred)
            pixel_loss = pixel_loss_fn(recon, target)
            total += pixel_loss

    return total, pixel_loss, latent_loss

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/3dert_test2.npy")
    parser.add_argument("--sparse", choices=("true", "false"), default="true")
    # seems like a good name. basically if im outputting pixel do i also want to take latent loss, and vice versa
    parser.add_argument("--antipodal_loss", choices=("true", "false"), default="true")
    # whether our unet outputs in latent or pixel space
    parser.add_argument("--pixel_output", choices=("true", "false"), default="false")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="unet_latents_TEST2")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--unet_ch", type=int, default=32)
    parser.add_argument("--normalization", choices=("log", "minmax"), default="log")
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    log_eps = 1e-6
    sparse = args.sparse == "true"
    antipodal_loss = args.antipodal_loss == "true"
    pixel_output = args.pixel_output == "true"
    unet_ch = args.unet_ch
    normalization = args.normalization

    device = torch.device(args.device)

    data = np.load(args.data, allow_pickle=True).item()
    x = torch.from_numpy(np.asarray(data["X"], dtype=np.float32)).to(device)
    y, xq, yq, zq, norm_stats = load_sigma_dataset(
        args.data,
        log_eps=log_eps,
        sparse=sparse,
        normalization=normalization,
    )
    y = y.to(device)

    n_samples, in_c, in_h, in_w = x.shape
    _, out_c, out_d, out_h, out_w = y.shape
    dataset = TensorDataset(x, y)
    train_size = int(0.8 * n_samples)
    val_size = int(0.1 * n_samples)
    test_size = n_samples - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(159753),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    
    best_vae_run = (
        "grid_search_vae_2/_lr2e-04_beta5e-02_kl400_bs128_ep1200_bc8_lc4_sparse"
        if sparse
        else "grid_search_vae_2/_lr2e-04_beta5e-02_kl400_bs128_ep1200_bc8_lc4"
    )
    run_dir = Path("saved_runs") / best_vae_run
    with (run_dir / "hparams.json").open("r", encoding="utf-8") as f:
        encoder_hparams = json.load(f)
    encoder_bc = int(encoder_hparams["bc"])
    encoder_lc = int(encoder_hparams["lc"])
    encoder_ckpt = run_dir / "checkpoints" / "best_val_loss.pt"

    encoder = UNetVAE(base_channels=encoder_bc, latent_channels=encoder_lc).to(device)
    encoder_checkpoint = torch.load(encoder_ckpt, map_location="cpu")
    ema_state = encoder_checkpoint.get("ema_state_dict")
    if ema_state is not None:
        ema_model = AveragedModel(encoder, multi_avg_fn=get_ema_multi_avg_fn(0.999))
        ema_model.load_state_dict(ema_state)
        encoder.load_state_dict(ema_model.module.state_dict())
    else:
        encoder.load_state_dict(encoder_checkpoint["model_state_dict"])
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        _, y_sample = next(iter(val_loader))
        y_sample = y_sample[:1].to(device)
        mu, _ = encoder.encode(y_sample)
    latent_out_c, latent_out_h, latent_out_w = mu.shape[1:]

    if pixel_output:
        model = Unet3D(
            in_channels=in_c,
            out_channels=out_c,
            output_shape=(out_d, out_h, out_w),
            ch=unet_ch,
        ).to(device)
    else:
        model = Unet(
            in_channels=in_c,
            out_channels=latent_out_c,
            output_shape=(latent_out_h, latent_out_w),
            ch=unet_ch,
            ch_mul=[1, 2, 2],
        ).to(device)
    model = torch.compile(model, mode="max-autotune")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=1e-3, fused=True)

    pixel_loss_fn = TVHuberLoss3D(
        lam_tv=3e-4,
        beta=0.5,
        tv="iso",
        tv_on="pred",
        w_in=10.0,
        thresh=0.5,
    )
    latent_loss_fn = nn.MSELoss()

    save_dir = args.save_dir
    save_dir += "_sparse" if sparse else "_dense"
    if antipodal_loss:
        save_dir += "_antipodal"
    else:
        save_dir += "_pixel_only" if pixel_output else "_latent_only"
    save_dir += f"_lr{lr:.0e}_bs{batch_size}_ep{epochs}"
    save_dir += f"_unet_ch{unet_ch}"
    save_dir += "_pixel" if pixel_output else "_latent"
    save_dir += f"_norm{normalization}"

    os.makedirs(f"saved_runs/{save_dir}", exist_ok=True)
    os.chdir(f"saved_runs/{save_dir}")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    with open("hparams.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sparse": args.sparse,
                "antipodal_loss": antipodal_loss,
                "pixel_output": pixel_output,
                "ckpt": args.ckpt,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "save_dir": save_dir,
                "unet_ch": unet_ch,
                "normalization": normalization,
                "norm_stats": norm_stats,
            },
            f,
            indent=2,
        )
    
    ckpt = args.ckpt
    best_val_loss = float("inf")
    early_stopping_patience = 100
    epochs_without_improvement = 0
    last_epoch = start_epoch if 'start_epoch' in locals() else 0

    if ckpt is not None:
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
    else:
        start_epoch = 0

    loss_history = {
        "train_total": [],
        "train_pixel": [],
        "train_latent": [],
        "val_total": [],
        "val_pixel": [],
        "val_latent": [],
    }

    pbar = tqdm(range(start_epoch, epochs + 1), desc="Epochs")
    for epoch in pbar:
        last_epoch = epoch
        model.train()
        running_loss = torch.tensor(0.0, device=device)
        running_pixel_loss = torch.tensor(0.0, device=device)
        running_latent_loss = torch.tensor(0.0, device=device)
        train_batches = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model(x_batch)
                loss, pixel_loss, latent_loss = compute_losses(
                    pred,
                    y_batch,
                    encoder,
                    pixel_output,
                    antipodal_loss,
                    pixel_loss_fn,
                    latent_loss_fn,
                )

            if torch.isnan(loss):
                continue

            loss.backward()
            optim.step()
            running_loss += loss.item()
            if pixel_loss is not None:
                running_pixel_loss += pixel_loss.item()
            if latent_loss is not None:
                running_latent_loss += latent_loss.item()
            train_batches += 1

        model.eval()
        with torch.no_grad():
            val_running_loss = torch.tensor(0.0, device=device)
            val_running_pixel_loss = torch.tensor(0.0, device=device)
            val_running_latent_loss = torch.tensor(0.0, device=device)
            val_batches = 0
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                val_loss, val_pixel_loss, val_latent_loss = compute_losses(
                    pred,
                    y_batch,
                    encoder,
                    pixel_output,
                    antipodal_loss,
                    pixel_loss_fn,
                    latent_loss_fn,
                )
                val_running_loss += val_loss.item()
                if val_pixel_loss is not None:
                    val_running_pixel_loss += val_pixel_loss.item()
                if val_latent_loss is not None:
                    val_running_latent_loss += val_latent_loss.item()
                val_batches += 1

            val_loss = float((val_running_loss / max(1, val_batches)).item())
            val_pixel_loss = (
                val_running_pixel_loss / max(1, val_batches)
                if val_batches
                else None
            )
            val_latent_loss = (
                val_running_latent_loss / max(1, val_batches)
                if val_batches
                else None
            )

            avg_train_loss = float((running_loss / max(1, train_batches)).item())
            avg_train_pixel_loss = float((running_pixel_loss / max(1, train_batches)).item())
            avg_train_latent_loss = float((running_latent_loss / max(1, train_batches)).item())
            pbar.set_postfix(
                {
                    "train": f"{avg_train_loss:.5f}",
                    "val": f"{val_loss:.5f}",
                }
            )
            loss_history["train_total"].append(avg_train_loss)
            loss_history["train_pixel"].append(avg_train_pixel_loss)
            loss_history["train_latent"].append(avg_train_latent_loss)

            loss_history["val_total"].append(val_loss)
            loss_history["val_pixel"].append(float(val_pixel_loss.item()) if val_pixel_loss is not None else None)
            loss_history["val_latent"].append(float(val_latent_loss.item()) if val_latent_loss is not None else None)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
                checkpoint = {
                    "epoch": int(epoch),
                    "model_state_dict": model_to_save.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    "best_val_loss": float(best_val_loss),
                }
                torch.save(checkpoint, "checkpoints/best_val_loss.pt")
            else:
                epochs_without_improvement += 1

            if epoch % 50 == 0 or epoch == epochs:
                x_plot, y_plot = next(iter(val_loader))
                x_plot = x_plot[:1].to(device)
                y_plot = y_plot[:1].to(device)
                plot_pred = model(x_plot)
                plot_pred = plot_pred if pixel_output else encoder.decode(plot_pred)
                # save_scatter_slices expects log-sigma; invert normalization first
                if normalization == "minmax":
                    y_plot_log = torch.log(invert_normalization(y_plot, normalization, norm_stats, log_eps) + log_eps)
                    plot_pred_log = torch.log(invert_normalization(plot_pred, normalization, norm_stats, log_eps) + log_eps)
                else:
                    y_plot_log = y_plot
                    plot_pred_log = plot_pred
                save_scatter_slices(
                    target=y_plot_log,
                    pred=plot_pred_log,
                    out_path=f"samples/epoch_{epoch}.png",
                    xq=xq,
                    yq=yq,
                    zq=zq,
                    sigma_min=float(np.log(log_eps)),
                )

                loss_title = "pixel loss" if pixel_output else "latent loss"
                if antipodal_loss:
                    loss_title += " + antipodal"

                plot_encoder_loss_curves(
                    loss_history,
                    loss_title,
                    out_path="loss_curve.png",
                )

            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch} after {early_stopping_patience} epochs without validation improvement."
                )
                break

    final_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    final_checkpoint = {
        "epoch": int(last_epoch),
        "model_state_dict": final_model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "best_val_loss": float(best_val_loss),
    }
    torch.save(final_checkpoint, "checkpoints/final.pt")

if __name__ == "__main__":
    main()
