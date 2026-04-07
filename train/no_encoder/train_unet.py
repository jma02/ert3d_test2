import argparse
import json
import os

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.unet import Unet
from train.util import TVHuberLoss2D
from train.no_encoder.util import plot_no_encoder_loss_curves, save_no_encoder_slices


torch.manual_seed(159753)
np.random.seed(159753)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/3dert_test2_easy.npy")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="unet_easy_no_encoder")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--unet_ch", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    unet_ch = args.unet_ch
    lr = args.lr
    weight_decay = args.weight_decay
    log_eps = 1e-6
    device = torch.device(args.device)

    data = np.load(args.data, allow_pickle=True).item()
    x = torch.from_numpy(np.asarray(data["X"], dtype=np.float32)).to(device)
    y = torch.from_numpy(np.asarray(data["Y_easy"], dtype=np.float32)).to(device)

    y = torch.log(y + log_eps)

    n_samples, in_c, in_h, in_w = x.shape
    _, out_c, out_h, out_w = y.shape

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

    model = Unet(
        in_channels=in_c,
        out_channels=out_c,
        output_shape=(out_h, out_w),
        ch=unet_ch,
    ).to(device)
    model = torch.compile(model, mode="max-autotune")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay, fused=True)

    loss_fn = TVHuberLoss2D(
        lam_tv=3e-4,
        beta=0.5,
        tv="iso",
        tv_on="pred",
        w_in=10.0,
        thresh=0.5,
    )

    save_dir = args.save_dir
    save_dir += f"_lr{lr:.0e}_bs{batch_size}_ep{epochs}"
    save_dir += f"_unet_ch{unet_ch}_wd{weight_decay:.0e}"

    os.makedirs(f"saved_runs/{save_dir}", exist_ok=True)
    os.chdir(f"saved_runs/{save_dir}")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    with open("hparams.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt": args.ckpt,
                "save_dir": save_dir,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "unet_ch": unet_ch,
                "weight_decay": weight_decay,
            },
            f,
            indent=2,
        )

    ckpt = args.ckpt
    best_val_loss = float("inf")
    early_stopping_patience = 50
    epochs_without_improvement = 0

    if ckpt is not None:
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        log_step = int(checkpoint["log_step"])
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
    else:
        start_epoch = 0
        log_step = 0

    last_epoch = start_epoch - 1

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    pbar = tqdm(range(start_epoch, epochs + 1), desc="Epochs")
    for epoch in pbar:
        last_epoch = epoch
        model.train()
        train_running_loss = torch.tensor(0.0, device=device)
        train_batches = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)

            if torch.isnan(loss):
                continue

            loss.backward()
            optim.step()
            log_step += 1
            train_running_loss += loss.item()
            train_batches += 1

        model.eval()
        with torch.no_grad():
            val_running_loss = torch.tensor(0.0, device=device)
            val_batches = 0
            for x_batch, y_batch in val_loader:
                pred = model(x_batch)
                val_running_loss += loss_fn(pred, y_batch)
                val_batches += 1
            train_loss = float((train_running_loss / max(1, train_batches)).item())
            val_loss = float((val_running_loss / max(1, val_batches)).item())
            pbar.set_postfix(
                {
                    "train_loss": f"{train_loss:.5f}",
                    "val_loss": f"{val_loss:.5f}",
                }
            )

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
                checkpoint = {
                    "epoch": int(epoch),
                    "log_step": int(log_step),
                    "model_state_dict": model_to_save.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    "best_val_loss": float(best_val_loss),
                }
                torch.save(checkpoint, "checkpoints/best_val_loss.pt")
            else:
                epochs_without_improvement += 1

            if epoch % 50 == 0 or epoch == epochs:
                xq = data["xq"]
                zq = data["zq"]
                extent = (float(xq[0]), float(xq[-1]), float(zq[0]), float(zq[-1]))
                save_no_encoder_slices(
                    target=y_batch[0].detach().cpu().numpy(),
                    pred=pred[0].detach().cpu().numpy(),
                    out_path=f"samples/epoch_{epoch}.png",
                    extent=extent,
                )
                plot_no_encoder_loss_curves(
                    train_loss_history,
                    val_loss_history,
                    "Loss",
                    out_path="loss_curve.png",
                )

            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch} after {early_stopping_patience} epochs without validation improvement."
                )
                break

    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    final_checkpoint = {
        "epoch": int(last_epoch),
        "log_step": int(log_step),
        "model_state_dict": model_to_save.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "best_val_loss": float(best_val_loss),
    }
    torch.save(final_checkpoint, "checkpoints/final.pt")

if __name__ == "__main__":
    main()
