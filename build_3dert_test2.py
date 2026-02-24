#!/usr/bin/env python3
"""Build ERT dataset pairs (X, Y) from sample*.mat files.

X: injection measurements shaped (3, 5, 5) per sample
Y: voxelized conductivity on a uniform rectangular prism covering
   the 3 surface lines and subsurface conductivity region.

Saves: 3dert_test2.npy (a dict with X, Y and metadata)
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from tqdm import tqdm


@dataclass
class GridSpec:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    surface_params: dict


def surface_z(x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
    dx_main = params["dx_main"]
    dy_main = params["dy_main"]
    k = params["k"]
    lx = dx_main / 2.0
    ly = dy_main / 2.0
    return 3.0 * np.sin(7.0 * x / dx_main) * np.sin(8.0 * (y - 0.25) / dy_main) * np.exp(
        -k * ((lx - x) ** 2 + (ly - y) ** 2)
    )


def build_grid_from_sample1(sample1_path: str, nx: int, ny: int, nz: int, pad_xy: float, air_margin: float) -> GridSpec:
    d = loadmat(sample1_path, squeeze_me=True, struct_as_record=False)
    u = d["u"]
    g = np.asarray(d["g"])

    lines_xyz = np.vstack(
        [
            np.asarray(u.line1.electrodes_xyz),
            np.asarray(u.line2.electrodes_xyz),
            np.asarray(u.line3.electrodes_xyz),
        ]
    )

    xmin, xmax = lines_xyz[:, 0].min(), lines_xyz[:, 0].max()
    ymin, ymax = lines_xyz[:, 1].min(), lines_xyz[:, 1].max()

    xr = xmax - xmin
    yr = ymax - ymin
    xpad = pad_xy * xr
    ypad = pad_xy * yr

    x = np.linspace(xmin - xpad, xmax + xpad, nx, dtype=np.float32)
    y = np.linspace(ymin - ypad, ymax + ypad, ny, dtype=np.float32)

    z_top = float(lines_xyz[:, 2].max() + air_margin)
    z_bottom = float(g[:, 2].min())
    z = np.linspace(z_bottom, z_top, nz, dtype=np.float32)

    return GridSpec(
        x=x,
        y=y,
        z=z,
        surface_params={"dx_main": 30.0, "dy_main": 15.0, "k": 0.0025},
    )


def extract_x(u_obj) -> np.ndarray:
    blocks = []
    for line_name in ("line1", "line2", "line3"):
        line = getattr(u_obj, line_name)
        inj = line.inj
        if not isinstance(inj, np.ndarray):
            inj = np.array([inj], dtype=object)
        rows = [np.asarray(item.U).reshape(-1).astype(np.float32) for item in inj]
        block = np.stack(rows, axis=0)  # (5, 5)
        blocks.append(block)
    x = np.stack(blocks, axis=0)  # (3, 5, 5)
    return x


def idw_knn_interp(g: np.ndarray, sigma: np.ndarray, q: np.ndarray, k: int, power: float, workers: int) -> np.ndarray:
    tree = cKDTree(g)
    dist, idx = tree.query(q, k=max(1, k), workers=workers)
    if k == 1:
        return sigma[idx]

    eps = 1e-8
    w = 1.0 / np.maximum(dist, eps) ** power
    w_sum = np.sum(w, axis=1, keepdims=True)
    w = w / np.maximum(w_sum, eps)
    vals = np.sum(w * sigma[idx], axis=1)
    return vals


def voxelize_sigma(
    g: np.ndarray,
    sigma: np.ndarray,
    grid: GridSpec,
    air_value: float,
    knn_k: int = 8,
    idw_power: float = 2.0,
    smooth_sigma: float = 0.0,
    workers: int = 8,
) -> np.ndarray:
    xx, yy, zz = np.meshgrid(grid.x, grid.y, grid.z, indexing="xy")  # (ny, nx, nz)
    q = np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])

    vals = idw_knn_interp(g, sigma, q, k=knn_k, power=idw_power, workers=workers).astype(np.float32)

    vals = vals.reshape(xx.shape)

    surf = surface_z(xx, yy, grid.surface_params)
    air_mask = zz > surf
    vals[air_mask] = np.float32(air_value)

    if smooth_sigma > 0:
        solid_mask = ~air_mask
        solid_vals = vals.copy()
        solid_vals[air_mask] = 0.0
        weight = solid_mask.astype(np.float32)
        num = gaussian_filter(solid_vals, sigma=smooth_sigma)
        den = gaussian_filter(weight, sigma=smooth_sigma)
        smooth = num / np.maximum(den, 1e-8)
        vals[solid_mask] = smooth[solid_mask]

    # return as (nz, ny, nx) for Conv3D-friendly depth-first ordering
    return np.transpose(vals, (2, 0, 1))


def plot_examples(out_dir: str, x_data: np.ndarray, y_data: np.ndarray, grid: GridSpec, n_plot: int = 3) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n_plot = min(n_plot, x_data.shape[0])
    for i in range(n_plot):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        im0 = axes[0].imshow(x_data[i, 0], cmap="viridis", aspect="auto")
        axes[0].set_title("X: line1 (inj x meas)")
        axes[0].set_xlabel("measurement")
        axes[0].set_ylabel("injection")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(x_data[i, 1], cmap="viridis", aspect="auto")
        axes[1].set_title("X: line2 (inj x meas)")
        axes[1].set_xlabel("measurement")
        axes[1].set_ylabel("injection")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)

        mid_y = y_data.shape[2] // 2
        im2 = axes[2].imshow(y_data[i, :, mid_y, :],
                             extent=[grid.x.min(), grid.x.max(), grid.z.min(), grid.z.max()],
                             origin="lower", cmap="plasma", aspect="auto")
        axes[2].set_title("Y: sigma voxel (XZ @ mid-Y)")
        axes[2].set_xlabel("X")
        axes[2].set_ylabel("Z")
        fig.colorbar(im2, ax=axes[2], fraction=0.046)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"example_{i+1}.png"), dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Users/jma2/Downloads/samples_onlysurface")
    parser.add_argument("--output", default="3dert_test2.npy")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--pad-xy", type=float, default=0.10)
    parser.add_argument("--air-margin", type=float, default=0.3)
    parser.add_argument("--air-value", type=float, default=0.0)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--smooth-sigma", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "sample*.mat")))
    if not files:
        raise FileNotFoundError(f"No sample*.mat found in {args.data_dir}")

    sample1_path = os.path.join(args.data_dir, "sample1.mat")
    if not os.path.exists(sample1_path):
        sample1_path = files[0]

    ref = loadmat(sample1_path, squeeze_me=True, struct_as_record=False)
    if "g" not in ref:
        raise KeyError(f"Reference sample {sample1_path} must contain 'g'.")
    g_ref = np.asarray(ref["g"], dtype=np.float32)

    grid = build_grid_from_sample1(
        sample1_path,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        pad_xy=args.pad_xy,
        air_margin=args.air_margin,
    )

    x_list = []
    y_list = []

    iterator = tqdm(files, desc="Processing samples", unit="sample")
    for i, fpath in enumerate(iterator, start=1):
        d = loadmat(fpath, squeeze_me=True, struct_as_record=False)
        x = extract_x(d["u"])
        g = np.asarray(d["g"], dtype=np.float32) if "g" in d else g_ref
        sigma = np.asarray(d["sigma_model"], dtype=np.float32).reshape(-1)
        if sigma.shape[0] != g.shape[0]:
            raise ValueError(
                f"Point/value mismatch in {os.path.basename(fpath)}: "
                f"len(sigma_model)={sigma.shape[0]} vs len(g)={g.shape[0]}"
            )
        y = voxelize_sigma(
            g,
            sigma,
            grid,
            air_value=args.air_value,
            knn_k=args.knn_k,
            idw_power=args.idw_power,
            smooth_sigma=args.smooth_sigma,
            workers=args.workers,
        )

        x_list.append(x)
        y_list.append(y)

    x_data = np.stack(x_list, axis=0).astype(np.float32)  # (N,3,5,5)
    y_data = np.stack(y_list, axis=0).astype(np.float32)  # (N,nz,ny,nx)

    payload = {
        "X": x_data,
        "Y": y_data,
        "xq": grid.x,
        "yq": grid.y,
        "zq": grid.z,
        "shape_X": x_data.shape,
        "shape_Y": y_data.shape,
        "notes": "X=(N,3,5,5) [line,inj,meas], Y=(N,nz,ny,nx) conductivity voxels with air",
        "interp": "idw_knn",
        "knn_k": args.knn_k,
        "idw_power": args.idw_power,
        "smooth_sigma": args.smooth_sigma,
        "workers": args.workers,
    }

    out_path = os.path.join(args.data_dir, args.output)
    np.save(out_path, payload, allow_pickle=True)
    print(f"saved: {out_path}")
    print(f"X shape: {x_data.shape}")
    print(f"Y shape: {y_data.shape}")

    plot_examples(os.path.join(args.data_dir, "dataset_preview"), x_data, y_data, grid, n_plot=3)
    print("saved plots in dataset_preview/")


if __name__ == "__main__":
    main()
