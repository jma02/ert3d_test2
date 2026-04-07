import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from models.VAE.cnn import CNNVAE
from train.VAE.util import plot_vae_loss_curves, save_vae_slices, tv3d_iso


def load_sigma_dataset(
    path: Path,
    log_eps: float,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    sigma = data["Y"].astype(np.float32, copy=False)
    sigma = sigma[:, None, ...]
    sigma = np.log(sigma + log_eps)
    xq = data["xq"].astype(np.float32, copy=False)
    yq = data["yq"].astype(np.float32, copy=False)
    zq = data["zq"].astype(np.float32, copy=False)
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
    parser = argparse.ArgumentParser(description="Train a CNN VAE on sigma voxels.")
    parser.add_argument("--data_path", type=str, default="data/3dert_test2.npy")
    parser.add_argument("--save_dir", type=str, default="vae_sigma_cnn")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    device = args.device

    log_eps = 1e-6
    val_split = 0.05
    batch_size = 512
    epochs = 2000
    lr = 2e-4
    beta = 1e-4
    recon_beta = 0.5
    tv_lambda = 3e-4
    num_workers = 8
    sigma_min = float(np.log(log_eps))
    max_plot_points = 2000

    sigma, xq, yq, zq = load_sigma_dataset(
        Path(args.data_path),
        log_eps=log_eps,
    )

    dataset = TensorDataset(sigma)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(159753)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    model = CNNVAE().to(device)
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))
    amp_device_type = "cuda" if device.startswith("cuda") else "cpu"

    save_root = Path("saved_runs") / args.save_dir
    (save_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_root / "samples").mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_loss = float("inf")
    early_stopping_patience = 50
    epochs_without_improvement = 0
    last_epoch = start_epoch - 1
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    loss_history = {
        "train_total": [],
        "train_recon": [],
        "train_kl": [],
        "val_total": [],
        "val_recon": [],
        "val_kl": [],
    }

    epoch_bar = tqdm(range(start_epoch, epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        last_epoch = epoch
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        train_recon = torch.tensor(0.0, device=device)
        train_kl = torch.tensor(0.0, device=device)
        val_recon = torch.tensor(0.0, device=device)
        val_kl = torch.tensor(0.0, device=device)

        for (x,) in tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False):
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=amp_device_type, enabled=device.startswith("cuda")):
                recon, mu, logvar = model(x)
                recon_loss = F.smooth_l1_loss(recon, x, beta=recon_beta)
                kl_loss = kl_divergence(mu, logvar)
                tv_loss = tv3d_iso(recon)
                loss = recon_loss + beta * kl_loss + tv_lambda * tv_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()

        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                recon_loss = F.smooth_l1_loss(recon, x, beta=recon_beta)
                kl_loss = kl_divergence(mu, logvar)
                tv_loss = tv3d_iso(recon)
                val_loss += (recon_loss + beta * kl_loss + tv_lambda * tv_loss).item()
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "log_eps": log_eps,
                "best_val_loss": float(best_val_loss),
            }
            torch.save(checkpoint, save_root / "checkpoints" / "best_val_loss.pt")
        else:
            epochs_without_improvement += 1

        if epoch % 100 == 0 or epoch == epochs:
            sample = next(iter(val_loader))[0][:1].to(device)
            recon, _, _ = model(sample)
            mu, _ = model.encode(sample)
            prior_z = torch.randn_like(mu)
            prior_sample = model.decode(prior_z)
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
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "log_eps": log_eps,
        "best_val_loss": float(best_val_loss),
    }
    torch.save(final_checkpoint, save_root / "checkpoints" / "final_model.pt")


if __name__ == "__main__":
    main()