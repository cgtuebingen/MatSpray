# Patches for External Dependencies

MatSpray requires modifications to two external repositories.
This directory contains patch files that apply the necessary changes.

## Base Commits

| Repository | URL | Commit |
|---|---|---|
| DiffusionRenderer (SVD) | <https://github.com/nv-tlabs/diffusion-renderer> | `0a6d71d69d81cc5d9ab9a5599dee4216cb3fc237` |
| 3D Gaussian Splatting | <https://github.com/graphdeco-inria/gaussian-splatting> | `54c035f7834b564019656c3e3fcc3646292f727d` |

## Patch Files

### 1. `gaussian-splatting-modifications.patch`

Modifies the original 3D Gaussian Splatting code to:

- **Mask GT instead of rendered output** (`train.py`): When alpha masks are present, the
  ground-truth image is composited against the background colour rather than zeroing out
  the rendered image. This prevents floaters in transparent regions.
- **Gracefully handle different rasterizer APIs** (`gaussian_renderer/__init__.py`): Builds
  rasterisation settings via kwargs and catches `TypeError` for versions that lack
  `antialiasing` or `debug` fields, and handles rasterisers that return 2 vs 3 values.
- **Fallback PLY creation** (`scene/__init__.py`): If the source PLY file is missing (e.g.,
  Blender datasets without a pre-existing point cloud), the scene creates one from the
  in-memory point cloud.
- **Store image path on Camera** (`scene/cameras.py`): Exposes the original file path on
  the camera object for downstream use.

**Apply:**

```bash
cd /path/to/gaussian-splatting
git checkout 54c035f7834b564019656c3e3fcc3646292f727d
git apply /path/to/matspray/patches/gaussian-splatting-modifications.patch
```

### 2. `diffusion-renderer-svd-modifications.patch`

Modifies tracked files in the DiffusionRenderer SVD repository:

- **Config** (`configs/rgbx_inference.yaml`): Adds `srgb_to_linear` flag and environment
  orientation controls (`env_rot`, `env_pitch`).
- **Inference script** (`inference_svd_xrgb.py`): Replaces the folder-based video grouping
  with frame-number discovery (`r_<n>_rgb.png` or `r_<n>.png`), adds per-frame saving of
  predictions and environment backgrounds, passes camera poses and frame indices to the
  environment map processor.
- **Environment projection** (`utils/utils_env_proj.py`): Accepts `frame_indices` and
  `env_total_frames` for globally consistent light rotation, applies a Blender-to-renderer
  coordinate transform, and indexes camera poses by frame number.
- **Inference utilities** (`utils/utils_rgbx_inference.py`): Adds natural-sort for frame
  ordering so frames 1, 2, …, 10 sort correctly.

**Apply:**

```bash
cd /path/to/diffusion-renderer
git checkout 0a6d71d69d81cc5d9ab9a5599dee4216cb3fc237
git apply /path/to/matspray/patches/diffusion-renderer-svd-modifications.patch
```

### 3. `diffusion-renderer-inference-improved.patch`

Creates a **new file** `inference_svd_rgbx_improved.py` by deriving it from the original
`inference_svd_xrgb.py`. This is the inference entry point called by the MatSpray training
scripts. Key differences from the original:

- Simplified pipeline without environment-map conditioning (material prediction only).
- sRGB ↔ linear colour-space conversions for physically correct material maps.
- Per-image processing with automatic pass detection (`basecolor`, `normal`, `roughness`,
  `metallic`).
- Chunk-based batching for memory efficiency.

**Apply:**

```bash
cd /path/to/diffusion-renderer
# After applying the previous patch:
cp inference_svd_xrgb.py inference_svd_rgbx_improved.py
patch inference_svd_rgbx_improved.py < /path/to/matspray/patches/diffusion-renderer-inference-improved.patch
```

## Quick Setup (All Patches at Once)

```bash
# Set paths
export GS_METHOD_DIR=/path/to/gaussian-splatting
export DIFFUSION_RENDERER_DIR=/path/to/diffusion-renderer
export MATSPRAY_DIR=/path/to/matspray

# Clone and patch Gaussian Splatting
git clone git@github.com:graphdeco-inria/gaussian-splatting.git "$GS_METHOD_DIR"
cd "$GS_METHOD_DIR"
git checkout 54c035f7834b564019656c3e3fcc3646292f727d
git apply "$MATSPRAY_DIR/patches/gaussian-splatting-modifications.patch"

# Clone and patch DiffusionRenderer
git clone https://github.com/nv-tlabs/diffusion-renderer.git "$DIFFUSION_RENDERER_DIR"
cd "$DIFFUSION_RENDERER_DIR"
git checkout 0a6d71d69d81cc5d9ab9a5599dee4216cb3fc237
git apply "$MATSPRAY_DIR/patches/diffusion-renderer-svd-modifications.patch"
cp inference_svd_xrgb.py inference_svd_rgbx_improved.py
patch inference_svd_rgbx_improved.py < "$MATSPRAY_DIR/patches/diffusion-renderer-inference-improved.patch"
```
