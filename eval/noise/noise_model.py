import torch
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_NOISE_LEVELS: list[float] = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
NOISE_SEED: int = 12345


def apply_ert_noise(
    x: torch.Tensor,
    noise_level: float,
    seed: int = NOISE_SEED,
) -> torch.Tensor:
    """Apply MATLAB-style ERT measurement noise.

    noise model (fraction convention: 1.0 = 100% noise):
        std = noise_level * |U| + (noise_level / 2) * globalmax_per_sample
        noised = U + std * randn(...)

    MATLAB reference used percentage convention (noisepercentage=1 means 1%),
    so MATLAB's noisepercentage=1 corresponds to noise_level=0.01 here.

    Args:
        x: Measurement tensor, shape (N, C, H, W) or (N, ...).
        noise_level: Noise fraction (1.0 = 100%).
        seed: RNG seed for reproducibility.
    """
    # per-sample global max: max(|x_i|) for each sample
    globalmax = x.abs().flatten(1).max(dim=1).values  # (N,)
    # reshape for broadcasting
    extra_dims = x.ndim - 1
    globalmax = globalmax.reshape(-1, *([1] * extra_dims))

    std = noise_level * x.abs() + (noise_level / 2.0) * globalmax

    gen = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)

    return x + std * noise


def add_noise_to_loader(
    test_loader: DataLoader,
    noise_level: float,
    seed: int = NOISE_SEED,
) -> DataLoader:
    """Re-wrap a DataLoader with noised X tensors, keeping Y unchanged."""
    xs, ys = [], []
    for x_batch, y_batch in test_loader:
        xs.append(x_batch)
        ys.append(y_batch)
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)

    x_noised = apply_ert_noise(x_all, noise_level, seed=seed)

    return DataLoader(
        TensorDataset(x_noised, y_all),
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
