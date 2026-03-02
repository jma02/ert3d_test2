#!/usr/bin/env python3
"""Build an easier dataset by slicing Y directly under each injection line."""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.io import loadmat


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/Users/jma2/Downloads/samples_onlysurface")
    p.add_argument("--input", default="3dert_test2.npy")
    p.add_argument("--output", default="3dert_test2_easy.npy")
    args = p.parse_args()

    npy_path = os.path.join(args.data_dir, args.input)
    d = np.load(npy_path, allow_pickle=True).item()
    X = d["X"]
    Y = d["Y"]  # (N, nz, ny, nx)
    xq = d["xq"]
    yq = d["yq"]
    zq = d["zq"]

    sample1_path = os.path.join(args.data_dir, "sample1.mat")
    s = loadmat(sample1_path, squeeze_me=True, struct_as_record=False)
    u = s["u"]

    line_y = []
    line_y_idx = []
    for line_name in ("line1", "line2", "line3"):
        line = getattr(u, line_name)
        electrodes = np.asarray(line.electrodes_xyz, dtype=np.float32)
        y_mean = float(np.mean(electrodes[:, 1]))
        idx = int(np.argmin(np.abs(yq - y_mean)))
        line_y.append(y_mean)
        line_y_idx.append(idx)

    y_slices = []
    for idx in line_y_idx:
        y_slices.append(Y[:, :, idx, :])  # (N, nz, nx)
    Y_easy = np.stack(y_slices, axis=1).astype(np.float32)  # (N, 3, nz, nx)

    payload = {
        "X": X,
        "Y_easy": Y_easy,
        "xq": xq,
        "yq": yq,
        "zq": zq,
        "line_y": np.asarray(line_y, dtype=np.float32),
        "line_y_idx": np.asarray(line_y_idx, dtype=np.int32),
        "shape_X": X.shape,
        "shape_Y_easy": Y_easy.shape,
        "notes": "Y_easy slices at nearest y-index under line1/line2/line3",
    }

    out_path = os.path.join(args.data_dir, args.output)
    np.save(out_path, payload, allow_pickle=True)
    print(f"saved: {out_path}")
    print(f"X shape: {X.shape}")
    print(f"Y_easy shape: {Y_easy.shape}")
    print(f"line_y_idx: {line_y_idx}")


if __name__ == "__main__":
    main()
