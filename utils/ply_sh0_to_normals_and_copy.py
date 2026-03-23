#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement


def compute_normals_from_sh0(f_dc0: np.ndarray, f_dc1: np.ndarray, f_dc2: np.ndarray, mode: str) -> np.ndarray:
    # Stack SH DC coefficients
    v = np.stack([f_dc0, f_dc1, f_dc2], axis=1).astype(np.float32)

    if mode == "rgb":
        # Convert SH DC to RGB like renderer: color = clamp(SH_C0 * sh + 0.5, 0)
        # SH_C0 constant from 3DGS implementation
        SH_C0 = 0.28209479177387814
        rgb = np.maximum(SH_C0 * v + 0.5, 0.0)
        # Map [0,1] -> [-1,1]
        n = rgb * 2.0 - 1.0
    else:
        # Directly use DC vector
        n = v

    # Normalize, avoid division by zero
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    n = n / norm
    return n.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Overwrite nx,ny,nz in a PLY using SH degree-0 (f_dc_0..2) and copy to destination")
    parser.add_argument("--src", required=True, help="Source point_cloud.ply path")
    parser.add_argument("--dst", required=True, help="Destination PLY path (e.g., dataset/base/points3d.ply)")
    parser.add_argument("--mode", choices=["rgb", "dc"], default="rgb", help="How to map SH0 to normals: rgb (default) or dc")
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)

    if not os.path.isfile(src):
        print(f"ERROR: Source PLY not found: {src}", file=sys.stderr)
        sys.exit(2)

    try:
        ply = PlyData.read(src)
    except Exception as e:
        print(f"ERROR: Failed to read PLY: {e}", file=sys.stderr)
        sys.exit(3)

    vert = ply["vertex"].data
    names = vert.dtype.names

    required_dc = ["f_dc_0", "f_dc_1", "f_dc_2"]
    for n in required_dc:
        if n not in names:
            print(f"ERROR: Missing required SH0 field '{n}' in PLY", file=sys.stderr)
            sys.exit(4)

    f_dc0 = np.asarray(vert["f_dc_0"])
    f_dc1 = np.asarray(vert["f_dc_1"])
    f_dc2 = np.asarray(vert["f_dc_2"])
    new_normals = compute_normals_from_sh0(f_dc0, f_dc1, f_dc2, args.mode)

    # Build new dtype: keep all existing fields; ensure nx,ny,nz exist (float32)
    new_dtype = []
    for name in names:
        new_dtype.append((name, vert.dtype.fields[name][0]))
    if "nx" not in names:
        new_dtype.append(("nx", "f4"))
    if "ny" not in names:
        new_dtype.append(("ny", "f4"))
    if "nz" not in names:
        new_dtype.append(("nz", "f4"))

    N = len(vert)
    new_rec = np.empty(N, dtype=new_dtype)

    # Copy over existing fields
    for name in names:
        new_rec[name] = vert[name]

    # Overwrite or set normals
    new_rec["nx"] = new_normals[:, 0]
    new_rec["ny"] = new_normals[:, 1]
    new_rec["nz"] = new_normals[:, 2]

    # Describe and write
    new_el = PlyElement.describe(new_rec, "vertex")
    out_ply = PlyData([new_el], text=False)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    out_ply.write(dst)
    print(f"[ok] Wrote PLY with normals from SH0 to: {dst}")


if __name__ == "__main__":
    main()


