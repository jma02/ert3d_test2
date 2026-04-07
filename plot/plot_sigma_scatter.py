#!/usr/bin/env python3
"""
Dense vs sparse scatter plots + easy-dataset 3-slice 3D view, using Plotly.

Dense:  full Y volume, log(sigma), Blues colorscale.
Sparse: same but yq[::2] (every other y-slice, matching the sparse training set).
Easy:   3dert_test2_easy.npy -- the 3 electrode-line slices rendered as
        curtains of points at their true y-coordinates in 3D space.

Electrode overlay (requires --mat) is always drawn on top as the last traces.
Outputs are saved as interactive HTML files.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.io import loadmat


LOG_EPS = 1e-6
SIGMA_MIN_LOG = float(np.log(LOG_EPS))
LINE_COLORS = {"line1": "red", "line2": "green", "line3": "blue"}


def _sample_points(vol_flat, xx_flat, yy_flat, zz_flat, mask, max_points):
    sigma = vol_flat[mask]
    x = xx_flat[mask]
    y = yy_flat[mask]
    z = zz_flat[mask]
    if sigma.size > max_points:
        idx = np.linspace(0, sigma.size - 1, max_points).astype(int)
        sigma, x, y, z = sigma[idx], x[idx], y[idx], z[idx]
    return x, y, z, sigma


def _rgba_colors(sigma, vmin, vmax, alpha_min=0.01, alpha_max=0.8):
    cmap = plt.get_cmap("PuBu")
    denom = vmax - vmin if vmax > vmin else 1.0
    t = (sigma - vmin) / denom                          # 0=low, 1=high
    alpha = np.clip(t * alpha_max, alpha_min, alpha_max)  # high sigma → more opaque
    rgba = []
    for ti, ai in zip(t, alpha):
        r, g, b, _ = cmap(float(ti))
        rgba.append(f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{ai:.3f})")
    return rgba


def _sigma_scatter_trace(x, y, z, sigma, name, vmin, vmax):
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=2,
            color=_rgba_colors(sigma, vmin, vmax),
        ),
        name=name,
    )


def _electrode_traces(overlay):
    traces = []
    for line_name, line_data in overlay.items():
        c = LINE_COLORS[line_name]
        e = line_data["electrodes"]
        # Electrode positions: solid line + circle markers
        traces.append(go.Scatter3d(
            x=e[:, 0], y=e[:, 1], z=e[:, 2],
            mode="lines+markers",
            line=dict(color=c, width=4),
            marker=dict(size=4, color=c, symbol="circle"),
            name=f"{line_name} electrodes",
        ))
        # Injection pairs: diamond markers at endpoints + dashed line
        for j, (pin, pout) in enumerate(line_data["segments"]):
            traces.append(go.Scatter3d(
                x=[pin[0], pout[0]], y=[pin[1], pout[1]], z=[pin[2], pout[2]],
                mode="lines+markers",
                line=dict(color=c, width=3, dash="dash"),
                marker=dict(size=7, color=c, symbol="diamond", opacity=0.9),
                name=f"{line_name} injection",
                showlegend=(j == 0),
            ))
    return traces


def _save_fig(fig, path):
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"Saved: {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npy", default="data/3dert_test2.npy")
    p.add_argument("--easy-npy", default="data/3dert_test2_easy.npy")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--mat", default="data/sample1.mat",
                   help="Path to a sample*.mat file for electrode geometry overlay (optional)")
    p.add_argument("--max-points", type=int, default=1000000)
    p.add_argument("--out-dir", default="plot_outputs")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.npy, allow_pickle=True).item()
    Y = d["Y"]
    xq = d["xq"]
    yq = d["yq"]
    zq = d["zq"]
    i = args.sample_idx

    overlay = None
    if args.mat is not None:
        u = loadmat(args.mat, squeeze_me=True, struct_as_record=False)["u"]
        overlay = {}
        for line_name in ("line1", "line2", "line3"):
            line = getattr(u, line_name)
            electrodes = np.asarray(line.electrodes_xyz, dtype=np.float32)
            inj = line.inj
            if not isinstance(inj, np.ndarray):
                inj = np.array([inj], dtype=object)
            segs = [(np.asarray(it.Iin_xyz, dtype=np.float32).reshape(-1),
                     np.asarray(it.Iout_xyz, dtype=np.float32).reshape(-1)) for it in inj]
            overlay[line_name] = {"electrodes": electrodes, "segments": segs}

    # ------------------------------------------------------------------
    # Dense vs Sparse
    # ------------------------------------------------------------------
    vol_full = Y[i]  # (nz, ny, nx)
    log_vol_full = np.log(vol_full + LOG_EPS)

    zz, yy, xx = np.meshgrid(zq, yq, xq, indexing="ij")
    log_flat_full = log_vol_full.reshape(-1)
    dense_mask = log_flat_full > max(SIGMA_MIN_LOG, float(np.quantile(log_flat_full, 0.05)))
    xd, yd, zd, sd = _sample_points(log_flat_full, xx.reshape(-1), yy.reshape(-1), zz.reshape(-1), dense_mask, args.max_points)

    yq_sparse = yq[::2]
    log_vol_sparse = log_vol_full[:, ::2, :]
    zz_s, yy_s, xx_s = np.meshgrid(zq, yq_sparse, xq, indexing="ij")
    log_flat_sparse = log_vol_sparse.reshape(-1)
    sparse_mask = log_flat_sparse > max(SIGMA_MIN_LOG, float(np.quantile(log_flat_sparse, 0.05)))
    xs, ys, zs, ss = _sample_points(log_flat_sparse, xx_s.reshape(-1), yy_s.reshape(-1), zz_s.reshape(-1), sparse_mask, args.max_points)

    vmin = float(min(sd.min(), ss.min()))
    vmax = float(max(sd.max(), ss.max()))

    for tag, xp, yp, zp, sp, label in (
        ("dense",  xd, yd, zd, sd, "Dense"),
        ("sparse", xs, ys, zs, ss, "Sparse (y[::2])"),
    ):
        traces = [_sigma_scatter_trace(xp, yp, zp, sp, label, vmin, vmax)]
        if overlay is not None:
            traces += _electrode_traces(overlay)
        fig = go.Figure(traces)
        fig.update_layout(title=f"{label} – sample {i}", scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
        _save_fig(fig, os.path.join(args.out_dir, f"{tag}_scatter_sample{i}.html"))

    # ------------------------------------------------------------------
    # Easy dataset: 3 slices in 3D space
    # ------------------------------------------------------------------
    if args.easy_npy is not None:
        de = np.load(args.easy_npy, allow_pickle=True).item()
        Y_easy = de["Y_easy"]   # (N, 3, nz, nx)
        line_y = de["line_y"]   # (3,) y-coordinate of each slice

        vol_easy = Y_easy[i]    # (3, nz, nx)
        log_vol_easy = np.log(vol_easy + LOG_EPS)
        vmin_e = float(log_vol_easy.min())
        vmax_e = float(log_vol_easy.max())

        traces_e = []
        zz_e, xx_e = np.meshgrid(zq, xq, indexing="ij")
        for k in range(3):
            log_flat_e = log_vol_easy[k].reshape(-1)
            y_val = float(line_y[k])
            yy_e_flat = np.full(log_flat_e.shape, y_val, dtype=np.float32)
            e_mask = log_flat_e > max(SIGMA_MIN_LOG, float(np.quantile(log_flat_e, 0.05)))
            xe, ye, ze, se = _sample_points(log_flat_e, xx_e.reshape(-1), yy_e_flat, zz_e.reshape(-1), e_mask, args.max_points)
            traces_e.append(go.Scatter3d(
                x=xe, y=ye, z=ze,
                mode="markers",
                marker=dict(
                    size=2,
                    color=_rgba_colors(se, vmin_e, vmax_e),
                ),
                name=f"line{k+1} (y={y_val:.1f})",
            ))

        if overlay is not None:
            traces_e += _electrode_traces(overlay)
        fig_e = go.Figure(traces_e)
        fig_e.update_layout(
            title=f"Easy dataset – 3 slices in 3D (sample {i})",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        )
        _save_fig(fig_e, os.path.join(args.out_dir, f"easy_slices_3d_sample{i}.html"))


if __name__ == "__main__":
    main()
