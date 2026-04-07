import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm

from models.VAE.default import UNetVAE
from train.VAE.util import plot_vae_loss_curves, save_vae_slices
from train.util import TVHuberLoss3D


def load_sigma_dataset(
    path: Path,
    log_eps: float,
    sparse: bool = False,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    sigma = data["Y"].astype(np.float32, copy=False)
    sigma = sigma[:, None, ...]
    sigma = np.log(sigma + log_eps)
    xq = data["xq"].astype(np.float32, copy=False)
    yq = data["yq"].astype(np.float32, copy=False)
    zq = data["zq"].astype(np.float32, copy=False)
    if sparse:
        sigma = sigma[:, :, :, ::2, :]
        yq = yq[::2]
    return torch.from_numpy(sigma), xq, yq, zq


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(159753)
np.random.seed(159753)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a UNet VAE on sigma voxels.")
    parser.add_argument("--data_path", type=str, default="data/3dert_test2.npy")
    parser.add_argument("--sparse", choices=("true", "false"), default="false")
    parser.add_argument("--save_dir", type=str, default="vae_sigma_unet")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--bc", type=int, default=16)
    parser.add_argument("--lc", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=128)  
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=5e-2)
    parser.add_argument("--kl_warmup_epochs", type=int, default=200)

    args = parser.parse_args()

    device = args.device
    sparse = args.sparse == "true"

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    beta = args.beta
    kl_warmup_epochs = args.kl_warmup_epochs
    log_eps = 1e-6
    recon_beta = 0.5
    tv_lambda = 3e-4
    sigma_min = float(np.log(log_eps))

    max_plot_points = 100000

    sigma, xq, yq, zq = load_sigma_dataset(
        Path(args.data_path),
        log_eps=log_eps,
        sparse=sparse,
    )

    sigma = sigma.to(device, non_blocking=True)
    num_workers = 0

    pin_memory = False

    dataset = TensorDataset(sigma)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(159753)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    bc = args.bc
    lc = args.lc
    
    model = UNetVAE(base_channels=bc, latent_channels=lc).to(device)
    # num params
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    model = torch.compile(model)

    loss_fn = TVHuberLoss3D(
        lam_tv=tv_lambda,
        beta=recon_beta,
        w_in=10.0,
        thresh=0.5,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=1e-3)
    amp_device_type = "cuda" if device.startswith("cuda") else "cpu"

    lr_tag = f"lr{lr:.0e}"
    beta_tag = f"beta{beta:.0e}"
    kl_tag = f"kl{kl_warmup_epochs}"
    bs_tag = f"bs{batch_size}"
    ep_tag = f"ep{epochs}"
    bc_tag = f"bc{bc}"
    lc_tag = f"lc{lc}"
    save_dir = f"{args.save_dir}_{lr_tag}_{beta_tag}_{kl_tag}_{bs_tag}_{ep_tag}_{bc_tag}_{lc_tag}"
    if sparse:
        save_dir += "_sparse"
    save_root = Path("saved_runs") / save_dir
    (save_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_root / "samples").mkdir(parents=True, exist_ok=True)

    with open(save_root / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sparse": sparse,
                "save_dir": save_dir,
                "ckpt": args.ckpt,
                "bc": bc,
                "lc": lc,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "beta": beta,
                "kl_warmup_epochs": kl_warmup_epochs,
            },
            f,
            indent=2,
        )

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_kl = float("inf")
    early_stopping_patience = 400
    epochs_without_improvement = 0
    last_epoch = start_epoch - 1
    ema_state_dict = None
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_val_kl = float(checkpoint.get("best_val_kl", best_val_kl))
        ema_state_dict = checkpoint.get("ema_state_dict")

    loss_history = {
        "train_total": [],
        "train_recon": [],
        "train_kl": [],
        "val_total": [],
        "val_recon": [],
        "val_kl": [],
    }

    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ema_model = AveragedModel(
        base_model,
        multi_avg_fn=get_ema_multi_avg_fn(0.999),
    ).to(device)
    if ema_state_dict is not None:
        ema_model.load_state_dict(ema_state_dict)

    epoch_bar = tqdm(range(start_epoch, epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        last_epoch = epoch
        kl_weight = beta * min(1.0, epoch / max(1, kl_warmup_epochs))
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        train_recon = torch.tensor(0.0, device=device)
        train_kl = torch.tensor(0.0, device=device)
        val_recon = torch.tensor(0.0, device=device)
        val_kl = torch.tensor(0.0, device=device)

        for (x,) in tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False):
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=amp_device_type, enabled=device.startswith("cuda"), dtype=torch.bfloat16):
                recon, mu, logvar = model(x)
                recon_loss = loss_fn(recon, x)
                kl_loss = kl_divergence(mu, logvar)
                loss = recon_loss + kl_weight * kl_loss

            if not torch.isfinite(loss):
                continue

            loss.backward()
            optimizer.step()
            ema_model.update_parameters(model._orig_mod if hasattr(model, "_orig_mod") else model)

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()

        model.eval()
        ema_model.module.eval()
        val_loss = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon, mu, logvar = ema_model.module(x)
                recon_loss = loss_fn(recon, x)
                kl_loss = kl_divergence(mu, logvar)
                val_loss += (recon_loss + kl_weight * kl_loss).item()
                val_recon += recon_loss.item()
                val_kl += kl_loss.item()

        train_loss = float((train_loss / max(1, len(train_loader))).item())
        val_loss = float((val_loss / max(1, len(val_loader))).item())
        avg_train_recon = float((train_recon / max(1, len(train_loader))).item())
        avg_train_kl = float((train_kl / max(1, len(train_loader))).item())
        avg_val_recon = float((val_recon / max(1, len(val_loader))).item())
        avg_val_kl = float((val_kl / max(1, len(val_loader))).item())
        epoch_bar.set_postfix(
            train=f"{train_loss:.6f}",
            recon=f"{avg_train_recon:.6f}",
            kl=f"{avg_train_kl:.6f}",
            val=f"{val_loss:.6f}",
            kl_weight=f"{kl_weight:.6f}",
        )

        loss_history["train_total"].append(train_loss)
        loss_history["train_recon"].append(avg_train_recon)
        loss_history["train_kl"].append(avg_train_kl)
        loss_history["val_total"].append(val_loss)
        loss_history["val_recon"].append(avg_val_recon)
        loss_history["val_kl"].append(avg_val_kl)

        plot_vae_loss_curves(
            loss_history,
            "VAE loss",
            out_path=str(save_root / "loss_curve.png"),
        )

        val_loss_improved = val_loss < best_val_loss
        track_kl_for_patience = epoch >= 400
        if track_kl_for_patience and not np.isfinite(best_val_kl):
            best_val_kl = avg_val_kl
        val_kl_improved = track_kl_for_patience and avg_val_kl < best_val_kl

        if val_kl_improved:
            best_val_kl = avg_val_kl
        if val_loss_improved:
            best_val_loss = val_loss
        if val_loss_improved or val_kl_improved:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model._orig_mod.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "log_eps": log_eps,
                "best_val_loss": float(best_val_loss),
                "best_val_kl": float(best_val_kl),
                "ema_state_dict": ema_model.state_dict(),
            }
            torch.save(checkpoint, save_root / "checkpoints" / "best_val_loss.pt")

        if val_loss_improved or val_kl_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 100 == 0 or epoch == epochs:
            sample = next(iter(val_loader))[0][:1].to(device)
            recon, _, _ = ema_model.module(sample)
            mu, _ = ema_model.module.encode(sample)
            prior_z = torch.randn_like(mu)
            prior_sample = ema_model.module.decode(prior_z)
            save_vae_slices(
                sample,
                recon,
                str(save_root / "samples" / f"recon_epoch_{epoch}.png"),
                xq,
                yq,
                zq,
                sigma_min,
                max_plot_points,
                generated=prior_sample,
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch} after {early_stopping_patience} epochs without validation improvement."
            )
            break

    final_checkpoint = {
        "epoch": last_epoch,
        "model_state_dict": model._orig_mod.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "log_eps": log_eps,
        "best_val_loss": float(best_val_loss),
        "best_val_kl": float(best_val_kl),
        "ema_state_dict": ema_model.state_dict(),
    }
    torch.save(final_checkpoint, save_root / "checkpoints" / "final_model.pt")


if __name__ == "__main__":
    main()
