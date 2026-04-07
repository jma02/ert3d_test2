import argparse
import multiprocessing
import subprocess
from itertools import product

from tqdm import tqdm


SPARSE_VAE_GRID = {
    "sparse": "true",
    "lr": [2e-4],
    "epochs": [1200],
    "batch_size": [128],
    "beta": [5e-2],
    "kl_warmup_epochs": [400],
    "bc": [8, 16],
    "lc": [4, 8],
}

DENSE_VAE_GRID = {
    "sparse": "false",
    "lr": [2e-4],
    "epochs": [1200],
    "batch_size": [128],
    "beta": [5e-2],
    "kl_warmup_epochs": [400],
    "bc": [8, 16],
    "lc": [4, 8],
}

SPARSE_GRIDS = (SPARSE_VAE_GRID,)
DENSE_GRIDS = (DENSE_VAE_GRID,)


def run_grid(
    grids: tuple[dict[str, list], ...],
    device: str,
    args: argparse.Namespace,
    failures: list[str],
) -> None:
    configs: list[tuple[str, float, int, int, float, int, int, int]] = []
    for grid in grids:
        sparse = grid["sparse"]
        configs.extend(
            (sparse, lr, epochs, batch_size, beta, kl_warmup_epochs, bc, lc)
            for lr, epochs, batch_size, beta, kl_warmup_epochs, bc, lc in product(
                grid["lr"],
                grid["epochs"],
                grid["batch_size"],
                grid["beta"],
                grid["kl_warmup_epochs"],
                grid["bc"],
                grid["lc"],
            )
        )

    progress_label = "sparse" if any(config[0] == "true" for config in configs) else "dense"
    for sparse, lr, epochs, batch_size, beta, kl_warmup_epochs, bc, lc in tqdm(
        configs,
        desc=f"{progress_label} grid",
        position=0 if progress_label == "sparse" else 1,
        leave=True,
    ):
            cmd = [
                "uv",
                "run",
                "-m",
                "train.VAE.train_VAE_unet",
                "--data_path",
                args.data_path,
                "--device",
                device,
                "--epochs",
                str(epochs),
                "--lr",
                str(lr),
                "--beta",
                str(beta),
                "--kl_warmup_epochs",
                str(kl_warmup_epochs),
                "--save_dir",
                args.save_dir,
                "--sparse",
                str(sparse),
                "--batch_size",
                str(batch_size),
                "--bc",
                str(bc),
                "--lc",
                str(lc),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                stdout = exc.stdout.strip()
                stderr = exc.stderr.strip()
                failures.append(
                    "FAILED CONFIG: "
                    f"sparse={sparse} lr={lr} beta={beta} "
                    f"kl_warmup_epochs={kl_warmup_epochs} epochs={epochs} "
                    f"batch_size={batch_size} bc={bc} lc={lc} "
                    f"device={device}\n"
                    f"COMMAND: {' '.join(cmd)}\n"
                    f"STDOUT:\n{stdout if stdout else '<empty>'}\n"
                    f"STDERR:\n{stderr if stderr else '<empty>'}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/3dert_test2.npy")
    parser.add_argument("--save_dir", type=str, default="grid_search_vae_2/")
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
