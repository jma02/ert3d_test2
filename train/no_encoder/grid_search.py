import argparse
import multiprocessing
import subprocess
from itertools import product


GRID_0 = {
    "lr": [1e-3, 5e-3],
    "epochs": [500],
    "batch_size": [64, 128, 256],
    "unet_ch": [32, 64],
    "weight_decay": [1e-3, 1e-4],
}

GRID_1 = {
    "lr": [1e-4, 5e-4],
    "epochs": [500],
    "batch_size": [64, 128, 256],
    "unet_ch": [32, 64],
    "weight_decay": [1e-3, 1e-4],
}

DEVICE_0_GRIDS = (GRID_0,)
DEVICE_1_GRIDS = (GRID_1,)


def run_grid(
    grids: tuple[dict[str, list], ...],
    device: str,
    args: argparse.Namespace,
    failures: list[str],
) -> None:
    for grid in grids:
        for lr, epochs, batch_size, unet_ch, weight_decay in product(
            grid["lr"],
            grid["epochs"],
            grid["batch_size"],
            grid["unet_ch"],
            grid["weight_decay"],
        ):
            cmd = [
                "uv",
                "run",
                "-m",
                "train.no_encoder.train_unet",
                "--data",
                args.data,
                "--device",
                device,
                "--epochs",
                str(epochs),
                "--lr",
                str(lr),
                "--save_dir",
                args.save_dir,
                "--batch_size",
                str(batch_size),
                "--unet_ch",
                str(unet_ch),
                "--weight_decay",
                str(weight_decay),
            ]
            print(" ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                failures.append(
                    "FAILED CONFIG: "
                    f"lr={lr} epochs={epochs} batch_size={batch_size} "
                    f"unet_ch={unet_ch} weight_decay={weight_decay} device={device}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/3dert_test2_easy.npy")
    parser.add_argument("--save_dir", type=str, default="grid_search_no_encoder/")
    parser.add_argument("--device_0", type=str, default="cuda:0")
    parser.add_argument("--device_1", type=str, default="cuda:1")
    args = parser.parse_args()

    manager = multiprocessing.Manager()
    failures = manager.list()
    proc_0 = multiprocessing.Process(
        target=run_grid,
        args=(DEVICE_0_GRIDS, args.device_0, args, failures),
    )
    proc_1 = multiprocessing.Process(
        target=run_grid,
        args=(DEVICE_1_GRIDS, args.device_1, args, failures),
    )
    proc_0.start()
    proc_1.start()
    proc_0.join()
    proc_1.join()
    if failures:
        print("The following runs failed:")
        for failure in list(failures):
            print(failure)
    else:
        print("All runs passed.")


if __name__ == "__main__":
    main()
