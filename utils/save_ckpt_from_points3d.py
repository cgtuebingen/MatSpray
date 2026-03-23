#!/usr/bin/env python3
import argparse
import os
import torch

# Ensure project root is on path if needed
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scene.gaussian_model import GaussianModel


def main():
    parser = argparse.ArgumentParser(description="Create a training checkpoint from points3d.ply or a 3DGS point_cloud.ply without running iterations")
    parser.add_argument("-s", "--source_path", help="Dataset base path (contains points3d.ply). Ignored if --ply is set.")
    parser.add_argument("-m", "--model_path", required=True, help="Output model directory for checkpoint")
    parser.add_argument("-i", "--iteration", type=int, default=30000, help="Iteration number to embed in checkpoint filename")
    parser.add_argument("--sh-degree", type=int, default=0, help="SH degree (default 0)")
    parser.add_argument("--ply", type=str, default=None, help="Explicit path to PLY file (e.g. .../iteration_30000/point_cloud.ply). If set, -s is ignored.")
    parser.add_argument("--convert-rgb-to-normals", action="store_true",
                        help="If set, ignore stored normals and derive normals from RGB as normal = normalize(rgb*2-1).")
    args = parser.parse_args()

    if args.ply:
        src_ply = os.path.abspath(args.ply)
        if not os.path.isfile(src_ply):
            raise FileNotFoundError(f"PLY not found at {src_ply}")
    else:
        if not args.source_path:
            parser.error("Either -s/--source_path or --ply is required")
        src_ply = os.path.join(args.source_path, "points3d.ply")
        if not os.path.isfile(src_ply):
            raise FileNotFoundError(f"points3d.ply not found at {src_ply}")

    os.makedirs(args.model_path, exist_ok=True)

    # Initialize GaussianModel with given SH degree, non-PBR
    gaussians = GaussianModel(sh_degree=args.sh_degree, render_type='render')
    # Special conversion mode: treat RGB as normals when requested
    if args.convert_rgb_to_normals:
        gaussians.convert_rgb_to_normals = True
    gaussians.load_ply(src_ply)
    # Initialize a minimal optimizer so capture() can serialize optimizer state
    gaussians.optimizer = torch.optim.Adam([gaussians._xyz], lr=0.0)

    ckpt_path = os.path.join(args.model_path, f"chkpnt{args.iteration}.pth")
    torch.save((gaussians.capture(), args.iteration), ckpt_path)
    print(f"[ok] Wrote checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()


