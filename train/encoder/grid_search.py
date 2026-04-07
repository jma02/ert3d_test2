import argparse
import multiprocessing
import subprocess
from itertools import product


# amazing language
SPARSE_LATENT_DATASET_GRID = {
    "sparse": "true",
    "antipodal_loss": ["true", "false"],
    "pixel_output": ["false"],
    "lr": [5e-4],
    "epochs": [800],
    "batch_size": [128],
    "unet_ch": [32, 64],
}
SPARSE_PIXEL_DATASET_GRID = {
    "sparse": "true",
    "antipodal_loss": ["true", "false"],
    "pixel_output": ["true"],
    "lr": [5e-4],
    "epochs": [800],
    "batch_size": [64],
    "unet_ch": [32],
}
DENSE_LATENT_DATASET_GRID = {
    "sparse": "false",
    "antipodal_loss": ["true", "false"],
    "pixel_output": ["false"],
    "lr": [5e-4],
    "epochs": [800],
    "batch_size": [64],
    "unet_ch": [32],
}
DENSE_PIXEL_DATASET_GRID = {
    "sparse": "false",
    "antipodal_loss": ["true", "false"],
    "pixel_output": ["true"],
    "lr": [5e-4],
    "epochs": [800],
    "batch_size": [32],
    "unet_ch": [32],
}

SPARSE_GRIDS = (SPARSE_LATENT_DATASET_GRID, SPARSE_PIXEL_DATASET_GRID)
DENSE_GRIDS = (DENSE_LATENT_DATASET_GRID, DENSE_PIXEL_DATASET_GRID)


def run_grid(
    grids: tuple[dict[str, list], ...],
    device: str,
    args: argparse.Namespace,
    failures: list[str],
) -> None:
    for grid in grids:
        sparse = grid["sparse"]
        for antipodal_loss, pixel_output, lr, epochs, batch_size, unet_ch in product(
            grid["antipodal_loss"],
            grid["pixel_output"],
            grid["lr"],
            grid["epochs"],
            grid["batch_size"],
            grid["unet_ch"],
        ):
            cmd = [
                "uv",
                "run",
                "-m",
                "train.encoder.train_unet",
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
                "--sparse",
                str(sparse),
                "--pixel_output",
                str(pixel_output),
                "--batch_size",
                str(batch_size),
                "--unet_ch",
                str(unet_ch),
                "--antipodal_loss",
                str(antipodal_loss),
            ]
            print(" ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                failures.append(
                    "FAILED CONFIG: "
                    f"sparse={sparse} pixel_output={pixel_output} "
                    f"antipodal_loss={antipodal_loss} lr={lr} "
                    f"epochs={epochs} batch_size={batch_size} unet_ch={unet_ch} "
                    f"device={device}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/3dert_test2.npy")
    parser.add_argument("--save_dir", type=str, default="grid_search_full_2/")
    parser.add_argument("--sparse_device", type=str, default="cuda:0")
    parser.add_argument("--dense_device", type=str, default="cuda:1")
    args = parser.parse_args()

    manager = multiprocessing.Manager()
    failures = manager.list()
    sparse_proc = multiprocessing.Process(
        target=run_grid,
        args=(SPARSE_GRIDS, args.sparse_device, args, failures),
    )
    dense_proc = multiprocessing.Process(
        target=run_grid,
        args=(DENSE_GRIDS, args.dense_device, args, failures),
    )
    sparse_proc.start()
    dense_proc.start()
    sparse_proc.join()
    dense_proc.join()
    if failures:
        print("The following runs failed:")
        for failure in list(failures):
            print(failure)
    else:
        print("All runs passed.")


if __name__ == "__main__":
    main()
