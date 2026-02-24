#!/usr/bin/env python3
"""Visualize 3dert_test2.npy with 3D point cloud views."""

from __future__ import annotations

import argparse
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npy", default="/Users/jma2/Downloads/samples_onlysurface/3dert_test2.npy")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--data-dir", default="/Users/jma2/Downloads/samples_onlysurface")
    p.add_argument("--max-points", type=int, default=8000)
    p.add_argument("--sigma-min", type=float, default=None, help="Filter points below this sigma")
    p.add_argument("--out", default="/Users/jma2/Downloads/samples_onlysurface/dataset_preview/pointcloud_sample0.png")
    p.add_argument(
        "--interactive-html",
        default="/Users/jma2/Downloads/samples_onlysurface/dataset_preview/pointcloud_sample0.html",
        help="Path to save interactive Plotly 3D HTML (requires plotly)",
    )
    p.add_argument("--no-interactive", action="store_true", help="Skip interactive HTML export")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    d = np.load(args.npy, allow_pickle=True).item()
    X = d["X"]
    Y = d["Y"]
    xq = d["xq"]
    yq = d["yq"]
    zq = d["zq"]

    i = args.sample_idx
    sample_files = sorted(glob.glob(os.path.join(args.data_dir, "sample*.mat")))
    if i < 0 or i >= len(sample_files):
        raise IndexError(f"sample-idx {i} out of range for {len(sample_files)} sample files")
    sample_file = sample_files[i]
    s = loadmat(sample_file, squeeze_me=True, struct_as_record=False)
    u = s["u"]

    line_colors = {"line1": "r", "line2": "g", "line3": "b"}
    overlay = {}
    for line_name in ("line1", "line2", "line3"):
        line = getattr(u, line_name)
        electrodes = np.asarray(line.electrodes_xyz, dtype=np.float32)
        inj = line.inj
        if not isinstance(inj, np.ndarray):
            inj = np.array([inj], dtype=object)
        segs = []
        for it in inj:
            pin = np.asarray(it.Iin_xyz, dtype=np.float32).reshape(-1)
            pout = np.asarray(it.Iout_xyz, dtype=np.float32).reshape(-1)
            segs.append((pin, pout))
        overlay[line_name] = {"electrodes": electrodes, "segments": segs}

    vol = Y[i]  # (nz, ny, nx)

    zz, yy, xx = np.meshgrid(zq, yq, xq, indexing="ij")

    sigma = vol.reshape(-1)
    x = xx.reshape(-1)
    y = yy.reshape(-1)
    z = zz.reshape(-1)

    if args.sigma_min is not None:
        m = sigma >= args.sigma_min
    else:
        # Drop air voxels at 0 by default
        m = sigma > 0

    x = x[m]
    y = y[m]
    z = z[m]
    sigma = sigma[m]

    if sigma.size == 0:
        raise RuntimeError("No voxels passed the sigma filter.")

    if sigma.size > args.max_points:
        idx = np.linspace(0, sigma.size - 1, args.max_points).astype(int)
        x = x[idx]
        y = y[idx]
        z = z[idx]
        sigma = sigma[idx]

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    sc1 = ax1.scatter(x, y, z, c=sigma, s=4, cmap="plasma", alpha=0.8)
    for line_name in ("line1", "line2", "line3"):
        e = overlay[line_name]["electrodes"]
        c = line_colors[line_name]
        ax1.plot(e[:, 0], e[:, 1], e[:, 2], color=c, linewidth=2.0, label=f"{line_name} electrodes")
        for pin, pout in overlay[line_name]["segments"]:
            ax1.plot([pin[0], pout[0]], [pin[1], pout[1]], [pin[2], pout[2]], color=c, alpha=0.45, linewidth=1.0)
    ax1.set_title(f"Y voxel point cloud (sample {i})")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc1, ax=ax1, fraction=0.03, pad=0.08, label="sigma")

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(X[i, 0], cmap="viridis", aspect="auto")
    ax2.set_title("X channel: line1 (inj x meas)")
    ax2.set_xlabel("measurement")
    ax2.set_ylabel("injection")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"Saved plot: {args.out}")
    print(f"Sample {i}: kept {sigma.size} points")

    if not args.no_interactive:
        if px is None:
            print("Plotly not installed; skipping interactive HTML. Install with: uv add plotly")
        else:
            fig3d = px.scatter_3d(
                x=x,
                y=y,
                z=z,
                color=sigma,
                opacity=0.8,
                color_continuous_scale="Plasma",
                title=f"Interactive sigma point cloud (sample {i})",
            )
            fig3d.update_traces(marker={"size": 2})
            color_map = {"line1": "red", "line2": "green", "line3": "blue"}
            for line_name in ("line1", "line2", "line3"):
                e = overlay[line_name]["electrodes"]
                c = color_map[line_name]
                fig3d.add_scatter3d(
                    x=e[:, 0], y=e[:, 1], z=e[:, 2],
                    mode="lines+markers",
                    line={"color": c, "width": 6},
                    marker={"size": 4, "color": c},
                    name=f"{line_name} electrodes",
                )
                for j, (pin, pout) in enumerate(overlay[line_name]["segments"]):
                    fig3d.add_scatter3d(
                        x=[pin[0], pout[0]], y=[pin[1], pout[1]], z=[pin[2], pout[2]],
                        mode="lines",
                        line={"color": c, "width": 3, "dash": "dot"},
                        opacity=0.6,
                        name=f"{line_name} injection",
                        showlegend=(j == 0),
                    )
            os.makedirs(os.path.dirname(args.interactive_html), exist_ok=True)
            fig3d.write_html(args.interactive_html, include_plotlyjs="cdn")
            print(f"Saved interactive HTML: {args.interactive_html}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
