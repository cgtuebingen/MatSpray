# MatSpray: Fusing 2D Material World Knowledge on 3D Geometry

<p align="center">
  <a href="https://matspray.jdihlmann.com/"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2512.18314"><img src="https://img.shields.io/badge/arXiv-2512.18314-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv"></a>
  <a href="https://arxiv.org/pdf/2512.18314"><img src="https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge" alt="Paper PDF"></a>
  <a href="https://drive.google.com/drive/folders/1znN_KllBKllIY_1PLZUHbnfHsB6KNifR?usp=sharing"><img src="https://img.shields.io/badge/Materials-Google_Drive-orange?style=for-the-badge" alt="Materials"></a>
</p>

<p align="center">
Philipp Langsteiner, Jan-Niklas Dihlmann, Hendrik P.A. Lensch<br>
University of T&uuml;bingen
</p>

Demo video: https://github.com/user-attachments/assets/95c2114e-e33a-46ed-933a-3a2ec153f6ca

![Alt text](media/teaser.gif)

This is the official code for **MatSpray**, a framework for fusing 2D material world knowledge from diffusion models into 3D Gaussian Splatting geometry to obtain relightable assets with physically based materials. Our method leverages pretrained 2D diffusion-based material predictors to generate per-view material maps (base color, roughness, metallic) and integrates them into a 3D Gaussian representation via Gaussian ray tracing. A lightweight Neural Merger refines the estimates for multi-view consistency and physical accuracy, enabling high-quality relightable 3D reconstruction.

This codebase is built on top of [Relightable 3D Gaussian](https://nju-3dv.github.io/projects/Relightable3DGaussian/) by Gao et al. (ECCV 2024), with the OptiX ray-tracing renderer adapted from the [OptiX 7 Course](https://github.com/ingowald/optix7course) by Ingo Wald, and parts of the deferred rendering pipeline taken from [SSS GS](https://sss.jdihlmann.com/) by Dihlmann et al. We thank the authors of these projects for making their code available.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Python Environment (venv)](#2-python-environment-venv)
  - [3. Install PyTorch](#3-install-pytorch)
  - [4. Install Python Dependencies](#4-install-python-dependencies)
  - [5. Build CUDA Extensions](#5-build-cuda-extensions)
  - [6. Install OptiX Renderer](#6-install-optix-renderer)
  - [7. Install External Tools](#7-install-external-tools)
- [Dataset Preparation](#dataset-preparation)
- [Rendering Ground Truth with Blender](#rendering-ground-truth-with-blender)
- [Environment Variables Reference](#environment-variables-reference)
- [Training Scripts Guide](#training-scripts-guide)
- [Running the Pipeline](#running-the-pipeline)
- [Evaluation](#evaluation)
- [Relighting and Composition](#relighting-and-composition)
- [GUI](#gui)
- [Verification Checklist](#verification-checklist)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## System Requirements

### Hardware

- **GPU**: NVIDIA GPU with compute capability >= 7.0 (RTX 2070 or newer; tested on RTX 3090 24 GB)
- **RAM**: 32 GB+ recommended
- **Disk**: ~50 GB free for datasets + outputs

### Software

| Dependency | Version | Notes |
|---|---|---|
| Linux | Ubuntu 20.04+ | Tested on Ubuntu 22.04 |
| NVIDIA Driver | >= 525 | Must support CUDA 11.8 |
| CUDA Toolkit | 11.8 | [Download](https://developer.nvidia.com/cuda-11-8-0-download-archive) |
| Python | 3.10 | Other 3.x versions may work but are untested |
| CMake | >= 3.26 | Required for OptiX build |
| GCC / G++ | 11 or 12 | C++17 support required |

Install system packages (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y git cmake ninja-build build-essential libglfw3-dev \
    python3-dev python3-venv bc
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repo-url>
cd MA_Philipp_Langsteiner
```

All commands below assume you are in the repository root.

### 2. Python Environment (venv)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

> Add `source /absolute/path/to/MA_Philipp_Langsteiner/venv/bin/activate` to your `.bashrc` so it activates automatically.

### 3. Install PyTorch

Install PyTorch with CUDA 11.8 support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

> For other CUDA versions see [PyTorch Get Started](https://pytorch.org/get-started/locally/).

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs all needed packages:

| Package | Purpose |
|---|---|
| `torch_scatter` | Scatter operations for 3DGS |
| `kornia` | Differentiable CV ops, depth-to-normal |
| `OpenEXR` / `pyexr` | EXR image I/O for HDR data |
| `lpips` / `scikit-image` | Perceptual quality metrics (LPIPS, PSNR, SSIM) |
| `dearpygui` | GUI visualization |
| `tensorboard` | Training monitoring |
| `numpy`, `scipy`, `opencv-python`, `pillow`, `matplotlib`, `plyfile`, `imageio`, `tqdm` | General utilities |

### 5. Build CUDA Extensions

First, clone nvdiffrast (not bundled in this repo):

```bash
git clone https://github.com/NVlabs/nvdiffrast.git
```

Then build and install all CUDA extensions from the repository root:

```bash
pip install ./submodules/simple-knn
pip install ./bvh
pip install ./r3dg-rasterization
pip install ./nvdiffrast
pip install ./gs_sss_rasterization
```

> If builds fail, ensure `CUDA_HOME` is set: `export CUDA_HOME=/usr/local/cuda`

### 6. Install OptiX Renderer

The `Optix/` directory contains a custom OptiX 7 ray-tracing renderer (`SampleRenderer`) used for intersection tracing during training and relighting. It requires three external dependencies: the OptiX SDK, the Slang shader compiler, and nanobind.

#### 6a. NVIDIA OptiX SDK 7.4.0

1. Download **OptiX 7.4.0** from [NVIDIA OptiX](https://developer.nvidia.com/designworks/optix/download) (requires NVIDIA developer account).
2. Run the self-extracting installer:

```bash
chmod +x NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64.sh
./NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64.sh --prefix=$HOME/optix
```

3. Set the environment variable (add to `.bashrc`):

```bash
export OptiX_INSTALL_DIR=$HOME/optix/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64
```

4. Verify:

```bash
ls $OptiX_INSTALL_DIR/include/optix.h
```

#### 6b. Slang Shader Compiler

1. Download Slang **v2025.2.2** from [GitHub Releases](https://github.com/shader-slang/slang/releases):

```bash
wget https://github.com/shader-slang/slang/releases/download/v2025.2.2/slang-2025.2.2-linux-x86_64.tar.gz
mkdir -p $HOME/slang
tar -xzf slang-2025.2.2-linux-x86_64.tar.gz -C $HOME/slang
```

2. Set the environment variable (add to `.bashrc`):

```bash
export SLANG_DIR=$HOME/slang/slang-2025.2.2-linux-x86_64
```

3. Verify:

```bash
$SLANG_DIR/bin/slangc --version
```

#### 6c. nanobind + scikit-build-core

```bash
pip install nanobind scikit-build-core
```

#### 6d. Build

```bash
cd Optix
mkdir -p build && cd build
cmake .. -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../..
```

The built `SampleRenderer*.so` module lands in `Optix/build/`. The codebase automatically adds it to `sys.path` at runtime.

#### 6e. Verify OptiX

```bash
python -c "import sys; sys.path.insert(0, 'Optix/build'); import SampleRenderer; print('OK: SampleRenderer loaded')"
```

### 7. Install External Tools

#### Blender (for ground truth rendering)

Blender renders ground truth material maps (base color, normals, roughness, metallic, depth) from `.blend` scene files.

```bash
# Download and extract (example for Blender 4.0)
wget https://download.blender.org/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz
tar -xJf blender-4.0.0-linux-x64.tar.xz -C $HOME/
export BLENDER_BIN=$HOME/blender-4.0.0-linux-x64/blender
```

Or if Blender is installed system-wide, just set `export BLENDER_BIN=blender`.

#### COLMAP (for custom / real-world datasets)

```bash
sudo apt install -y colmap
```

Or build from source: [COLMAP Installation](https://colmap.github.io/install.html).

#### External Repositories (optional)

These are only needed for the material/depth estimation stages (`--run-rgb2x`, `--run-marigold`, `--run-diffusion-renderer`):

| Repository | Env Variable | Purpose |
|---|---|---|
| [RGB2X](https://github.com/zheng95z/rgb2x) | `RGB2X_DIR` | Material property prediction from RGB |
| [DiffusionRenderer](https://github.com/nv-tlabs/diffusion-renderer) | `DIFFUSION_RENDERER_DIR` | SVD-based material decomposition |
| [Marigold](https://github.com/prs-eth/Marigold) | `MARIGOLD_DIR` | Monocular depth estimation |
| [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | `GS_METHOD_DIR` | Base 3DGS (only for the COLMAP external-GS path) |

Clone each and set its environment variable. Each has its own installation instructions.

> **Important:** Both DiffusionRenderer and Gaussian Splatting require modifications to work with MatSpray. We provide ready-made patch files in the [`patches/`](patches/) directory. See the sections below and [`patches/README.md`](patches/README.md) for details.

#### DiffusionRenderer Setup (Important)

DiffusionRenderer is used to generate per-view material maps (base color, roughness, metallic, normals) from input images. These material maps serve as supervision for the training pipeline. **You must run DiffusionRenderer before starting training** (via the `--run-diffusion-renderer` flag or manually) so that the material maps are available in the dataset directory.

> **WARNING: Use the SVD version of DiffusionRenderer only. The COSMOS version does NOT work and will produce incorrect material predictions. Make sure you check out and set up the SVD-based variant.**

Setup:

1. Clone the DiffusionRenderer repository at the correct commit and apply our patches:

```bash
git clone https://github.com/nv-tlabs/diffusion-renderer.git "$DIFFUSION_RENDERER_DIR"
cd "$DIFFUSION_RENDERER_DIR"
git checkout 0a6d71d69d81cc5d9ab9a5599dee4216cb3fc237

# Apply MatSpray modifications to existing files
git apply /path/to/matspray/patches/diffusion-renderer-svd-modifications.patch

# Create the improved inference script used by our training pipeline
cp inference_svd_xrgb.py inference_svd_rgbx_improved.py
patch inference_svd_rgbx_improved.py < /path/to/matspray/patches/diffusion-renderer-inference-improved.patch
```

2. Create a separate virtual environment for DiffusionRenderer (it has its own dependencies):

```bash
python3 -m venv venv_dr
source venv_dr/bin/activate
pip install -r requirements.txt
# Follow the repo's own installation instructions for model weights etc.
deactivate
```

3. Set the environment variable:

```bash
export DIFFUSION_RENDERER_DIR=/path/to/diffusion-renderer
```

4. The training scripts call DiffusionRenderer automatically when `--run-diffusion-renderer` is passed. It runs `inference_svd_rgbx_improved.py` with these defaults:

```bash
python inference_svd_rgbx_improved.py \
    --config configs/rgbx_inference.yaml \
    inference_input_dir="<dataset>/base/train/images_bg/" \
    inference_save_dir="<dataset>/base/train/" \
    inference_n_frames=20 \
    inference_n_steps=20 \
    model_passes="['basecolor','normal','metallic','roughness']" \
    inference_res="[512,512]" \
    chunk_mode="all"
```

The output material maps are written into the dataset's `train/` directory (e.g., `train/diffusionrenderer_baseColor/`, `train/diffusionrenderer_roughness/`, etc.) and are picked up automatically by `train.py` during the supervision stages.

#### Gaussian Splatting Setup (COLMAP Workflows Only)

If you use scripts that run external 3D Gaussian Splatting (e.g., `run_gaussian_splatting_colmap.sh`), you need to clone and patch the original repository:

```bash
git clone git@github.com:graphdeco-inria/gaussian-splatting.git "$GS_METHOD_DIR"
cd "$GS_METHOD_DIR"
git checkout 54c035f7834b564019656c3e3fcc3646292f727d

# Apply MatSpray modifications
git apply /path/to/matspray/patches/gaussian-splatting-modifications.patch
```

The patch makes the following changes:
- **Masking fix** (`train.py`): Alpha masks are applied to the ground-truth image (composited against the background) instead of zeroing out the rendered output. This prevents floaters in transparent regions.
- **Rasterizer compatibility** (`gaussian_renderer/__init__.py`): Gracefully handles different rasterizer API versions.
- **PLY fallback** (`scene/__init__.py`): Creates an input PLY from in-memory point clouds when no PLY file exists on disk (e.g., Blender datasets).

> **WARNING — Masking during external 3DGS training:** If you use any external Gaussian Splatting implementation, make sure masks are applied to the **ground-truth image only**, not to the 3DGS rendered output. Masking the rendered output causes floaters. Our patch handles this correctly.

---

## Dataset Preparation

### NeRF Synthetic

Download from [Google Drive](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi?usp=drive_link) (provided by [NeRF](https://github.com/bmild/nerf)).

Expected structure:

```
<ROOT_DIR>/                          # e.g., /data/datasets/nerf_synthetic/
├── lego/
│   ├── base/
│   │   ├── train/
│   │   │   ├── images/              # training RGB images
│   │   │   └── images_bg/           # images with background (after Blender rendering)
│   │   ├── test/
│   │   │   └── images/
│   │   ├── transforms_train.json
│   │   └── transforms_test.json
│   ├── blend_files/
│   │   └── lego.blend               # Blender scene for GT rendering
│   └── env_maps/
│       ├── envmap3.exr
│       ├── envmap6.exr
│       ├── envmap12.exr
│       └── envmap24.exr
├── chair/
├── drums/
├── ...
```

### COLMAP / Navi (Real-World)

Use `script/navi_dataset_prep.bash` to prepare from raw Navi data:

```bash
NAVI_RAW_DIR=/path/to/navi/v1.2/ \
NAVI_DATASET_DIR=/path/to/datasets/navi/ \
bash script/navi_dataset_prep.bash
```

This runs: image downsampling + masking -> COLMAP feature extraction -> matching -> mapping.

Expected structure after preparation:

```
<ROOT_DIR>/                          # e.g., /data/datasets/navi/
├── <object_name>/
│   └── base/
│       ├── images/                  # input RGB images
│       ├── images_bg/               # background-masked images
│       ├── sparse/0/                # COLMAP sparse reconstruction
│       │   ├── cameras.bin
│       │   ├── images.bin
│       │   └── points3D.bin
│       ├── cameras.json
│       └── points3d.ply
├── blend_files/
│   └── <object_name>.blend
└── env_maps/
```

### Pre-processed DTU

Download from [here](https://box.nju.edu.cn/f/d9858b670ab9480fb526/?dl=1):

```
datasets/neilfpp/data_dtu/
├── DTU_scan24/
│   └── inputs/
│       ├── depths/
│       ├── images/
│       ├── model/
│       ├── normals/
│       ├── pmasks/
│       └── sfm_scene.json
├── ...
```

### Pre-processed Tanks and Temples

Download from [here](https://box.nju.edu.cn/f/f0b094cc22bf4934a000/?dl=1). Same structure as DTU under `datasets/neilfpp/data_tnt/`.

### Synthetic4Relight

Download from [Google Drive](https://drive.google.com/file/d/1wWWu7EaOxtVq8QNalgs6kDqsiAm7xsRh/view?usp=sharing) (provided by [InvRender](https://github.com/zju3dv/InvRender)).

### Ground Points for Composition

For multi-object composition, download the ground plane PLY from [here](https://box.nju.edu.cn/f/c51d9de245f04d0fb872/?dl=1) and place it at `./point/ground.ply`.

---

## Rendering Ground Truth with Blender

The Blender scripts in `blender_scripts/` render ground truth material maps from `.blend` scene files. This step is needed when you want to train with GT supervision (the `--render-gts` flag in training scripts).

### Train-View Rendering

```bash
$BLENDER_BIN --background <dataset>/blend_files/<object>.blend \
    --python blender_scripts/blender_render_script_train.py \
    -- --object_name <object> --output_dir <dataset>/
```

### Test-View Rendering (Multiple Environment Maps)

```bash
$BLENDER_BIN --background <dataset>/blend_files/<object>.blend \
    --python blender_scripts/blender_render_script_test.py \
    -- --env_dir <dataset>/env_maps/ --results_path <dataset>/<object>
```

### Output After Rendering

```
<object>/base/
├── train/
│   ├── images/                        # rendered RGB
│   ├── images_bg/                     # with background
│   ├── base_basecolor/                # base color AOV
│   ├── base_metallicroughness/        # metallic + roughness AOV
│   ├── base_normal/                   # camera-space normals
│   ├── base_depth/                    # depth maps
│   └── transforms_train.json
└── test/
    ├── images/
    └── ...  (same AOVs as train)
```

### Headless GPU Rendering

For running Blender without a display (e.g., on a server):

```bash
export PYOPENGL_PLATFORM=egl
export CYCLES_CUDA_EXTRA_CFLAGS="-I${CUDA_HOME:-/usr/local/cuda}/include"
```

---

## Environment Variables Reference

All configurable paths use environment variables with sensible defaults. Set them before running any script.

### Required

| Variable | Description | Example |
|---|---|---|
| `ROOT_DIR` | Dataset root directory | `/data/datasets/nerf_synthetic/` |
| `OUTPUT_DIR` | Output directory for results | `/data/results/nerf_synthetic/` |
| `CUDA_HOME` | CUDA Toolkit path | `/usr/local/cuda` |
| `OptiX_INSTALL_DIR` | OptiX SDK path | `$HOME/optix/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64` |
| `SLANG_DIR` | Slang compiler path | `$HOME/slang/slang-2025.2.2-linux-x86_64` |

### Optional

| Variable | Description | Default |
|---|---|---|
| `BLENDER_BIN` | Blender binary | `blender` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | all GPUs |
| `RGB2X_DIR` | RGB2X repo path | `/path/to/external/rgbx` |
| `MARIGOLD_DIR` | Marigold repo path | `/path/to/external/Marigold` |
| `DIFFUSION_RENDERER_DIR` | diffusion-renderer repo path | `/path/to/external/diffusion-renderer` |
| `GS_METHOD_DIR` | gaussian-splatting repo path | `/path/to/external/gaussian-splatting` |
| `RESULTS_DIR` | Used by `evaluation/` scripts | `/path/to/results/` |

### Recommended `.bashrc` Setup

```bash
# CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# OptiX + Slang
export OptiX_INSTALL_DIR=$HOME/optix/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64
export SLANG_DIR=$HOME/slang/slang-2025.2.2-linux-x86_64
export PATH=$SLANG_DIR/bin:$PATH

# Blender (if not system-installed)
export BLENDER_BIN=$HOME/blender/blender

# Datasets (adjust to your setup)
export ROOT_DIR=/data/datasets/nerf_synthetic/
export OUTPUT_DIR=/data/results/nerf_synthetic/
```

---

## Training Scripts Guide

> **Ready-to-use scripts are already provided in the `script/` folder.** You do not need to write your own training loops. Simply set the environment variables (`ROOT_DIR`, `OUTPUT_DIR`, etc.), choose the script matching your dataset, select which stages to run via flags, and execute. The scripts handle the entire pipeline from ground truth rendering through training to relighting evaluation.

The `script/` directory contains end-to-end pipeline scripts. Each one handles a specific dataset type and runs through: ground truth rendering, material estimation, 3DGS training, NeILF BRDF decomposition, supervision training, and relighting evaluation.

### Overview

| Script | Dataset | Description |
|---|---|---|
| `create_gt_and_run_training_nerf.bash` | NeRF Synthetic | Lego, chair, drums, etc. Synthetic scenes with known geometry. |
| `create_gt_and_run_training_COLMAP.bash` | COLMAP / Navi | Real-world objects reconstructed with COLMAP. Supports pre-processed GS outputs. |
| `create_gt_and_run_training_aria.bash` | Aria | Aria dataset scenes. Expects external 3DGS PLY. |
| `create_gt_and_run_training_specular.bash` | Specular | Specular/reflective objects. Expects external 3DGS PLY. |
| `create_gt_and_run_training_stanford_orb.bash` | Stanford ORB | Stanford ORB objects (imported as OBJ in Blender). |
| `create_gt_and_run_training_LERF_CLEAN.sh` | LERF | LERF dataset scenes. |
| `navi_dataset_prep.bash` | Navi raw data | Downsamples images, applies masks, runs COLMAP SfM. |
| `run_gaussian_splatting_colmap.sh` | Any COLMAP | Runs external gaussian-splatting repo on a COLMAP dataset. |
| `run_gaussian_splatting_nerf.sh` | Any NeRF | Runs external gaussian-splatting repo on NeRF-format data. |

### Pipeline Stages

Each main training script (`create_gt_and_run_training_*.bash`) runs through these stages **in order**, controlled by flags:

```
1. --render-gts           Blender headless rendering of ground truth material maps
                          (base color, normals, roughness, metallic, depth)

2. --run-rgb2x            RGB2X material prediction from rendered images
   --run-marigold         Marigold monocular depth estimation
   --run-diffusion-renderer  Diffusion-based material decomposition
   (--srgb-to-linear)     Enable sRGB-to-linear conversion for the above

3. --run-base-3dgs        Stage 1: Train base 3D Gaussian Splatting (30k iterations)
                          Produces chkpnt30000.pth and point cloud

4. --run-base-neilf       Stage 2: Train NeILF model for BRDF decomposition (40k iterations)
                          Uses 3DGS checkpoint as initialization
                          Runs evaluation + relighting with 4 environment maps

5. --run-neilf-supervision  Stage 3: NeILF with material supervision losses
                            Uses RGB2X/diffusion-renderer material maps as supervision
                            Tests multiple loss weight configurations

6. --run-pure-supervision   Variant: pure supervision (no NeILF self-supervision)

7. --run-projected-average  Projected average baseline

8. --run-mlp              Stage 4: Train MLP-based material model on top of deferred shading
                          (enabled by default)
```

> **Important: Stage ordering matters.** Stages 1-2 (rendering + material estimation) must complete before stages 3+ (training) can start, because training relies on the material maps as supervision input. If you run DiffusionRenderer separately (outside the script), make sure its output maps are in the dataset directory before launching training.

> **Masking warning for external Gaussian Splatting:** If you use an external Gaussian Splatting implementation (not the one built into this framework) for the base 3DGS stage, make sure the mask is applied **only to the ground truth image** and **not** to the 3DGS rasterization output. Masking the rasterized output will cause the optimizer to push Gaussians into masked regions to minimize the loss, creating **floaters** (stray Gaussians in empty space). The correct approach is to mask only the GT side of the loss so the 3DGS output is compared against the masked ground truth.

### How to Use

**Basic usage (NeRF Synthetic):**

```bash
source venv/bin/activate

export ROOT_DIR=/data/datasets/nerf_synthetic/
export OUTPUT_DIR=/data/results/nerf_synthetic/

# Run everything from rendering to MLP training
bash script/create_gt_and_run_training_nerf.bash \
    --render-gts \
    --run-diffusion-renderer \
    --run-base-3dgs \
    --run-base-neilf \
    --run-neilf-supervision \
    --run-mlp
```

**COLMAP dataset with pre-processed GS:**

```bash
export ROOT_DIR=/data/datasets/navi/
export OUTPUT_DIR=/data/results/navi/

bash script/create_gt_and_run_training_COLMAP.bash \
    --use-preprocessed-gs \
    --gs-ply /path/to/gs_outputs/{scene}/point_cloud/iteration_30000/point_cloud.ply \
    --cameras-json /path/to/gs_outputs/{scene}/cameras.json \
    --run-base-3dgs \
    --run-base-neilf \
    --run-mlp
```

> The `{scene}` template is automatically expanded to each scene name during iteration.

**Navi dataset preparation:**

```bash
NAVI_RAW_DIR=/path/to/navi/raw/ \
NAVI_DATASET_DIR=/path/to/datasets/navi/ \
bash script/navi_dataset_prep.bash
```

**External Gaussian Splatting (COLMAP format):**

```bash
bash script/run_gaussian_splatting_colmap.sh \
    --method-dir /path/to/gaussian-splatting \
    --data-dir /data/datasets/navi/object_name/base \
    --output-root /data/results/navi/object_name \
    --iters 30000
```

### Editing Scene Lists

Each script has a `list=()` array near the top that controls which scenes to process. Edit this to select scenes:

```bash
# In create_gt_and_run_training_nerf.bash:
list=("lego")                                    # single scene
list=("chair" "drums" "ficus" "hotdog")          # multiple scenes
list=("lego" "materials" "mic" "ship")           # all NeRF synthetic
```

### Editing Model Variants

The `models=()` array selects which material estimation method's outputs to use for supervision:

```bash
models=("diffusionrenderer")   # use diffusion-renderer outputs
models=("rgb2x")               # use RGB2X outputs
models=("base")                # use base (no external supervision)
```

---

## Running the Pipeline

### Individual Commands

Instead of using the full scripts, you can run each step manually:

**Stage 1 -- Base 3DGS:**

```bash
python train.py --eval \
    -s $ROOT_DIR/lego/base/ \
    -m $OUTPUT_DIR/lego/3dgs \
    --lambda_normal_render_depth 0.01 \
    --lambda_normal_smooth 0.01 \
    --lambda_mask_entropy 0.1 \
    --save_training_vis
```

**Stage 2 -- NeILF (BRDF decomposition):**

```bash
python train.py --eval \
    -s $ROOT_DIR/lego/base/ \
    -m $OUTPUT_DIR/lego/neilf \
    -c $OUTPUT_DIR/lego/3dgs/chkpnt30000.pth \
    -t neilf --sample_num 32 \
    --iterations 40000 \
    --position_lr_init 0.000016 \
    --position_lr_final 0.00000016 \
    --lambda_light 0.01 \
    --lambda_env_smooth 0.01
```

**Stage 3 -- NeILF with Material Supervision:**

```bash
python train.py --eval \
    -s $ROOT_DIR/lego/base/ \
    -m $OUTPUT_DIR/lego/neilf_supervised \
    -c $OUTPUT_DIR/lego/3dgs/chkpnt30000.pth \
    --base_color_folder /train/diffusionrenderer_baseColor/ \
    --roughness_folder /train/diffusionrenderer_roughness/ \
    --metallic_folder /train/diffusionrenderer_metallic/ \
    --lambda_base_color_buffer 1.0 \
    --lambda_roughness_buffer 1.0 \
    --lambda_metallic_buffer 1.0 \
    -t neilf --sample_num 32 \
    --iterations 40000
```

---

## Evaluation

**Novel View Synthesis:**

```bash
# Stage 1 (3DGS)
python eval_nvs.py --eval \
    -m $OUTPUT_DIR/lego/3dgs \
    -c $OUTPUT_DIR/lego/3dgs/chkpnt30000.pth

# Stage 2 (NeILF)
python eval_nvs.py --eval \
    -m $OUTPUT_DIR/lego/neilf \
    -c $OUTPUT_DIR/lego/neilf/chkpnt40000.pth \
    -t neilf
```

**Average Metrics Across Scenes:**

```bash
RESULTS_DIR=/data/results/navi python evaluation/calculate_average_metrics.py
RESULTS_DIR=/data/results/navi python evaluation/calculate_average_timings.py
```

---

## Relighting and Composition

**Object relighting:**

```bash
python relighting.py \
    -co "$ROOT_DIR/lego/envmap12/" \
    --envmap_path "env_map/envmap12.exr" \
    --output "$OUTPUT_DIR/lego/neilf/envmap12" \
    --ply_path "$OUTPUT_DIR/lego/neilf/point_cloud/iteration_40000/point_cloud.ply" \
    --sample_num 64 \
    --video
```

An example usage of this can be found in the `script/` folder in the `create_gt_and_...` scripts (they call `relighting.py` with the correct dataset directories and arguments).

---

## GUI

Visualize trained models interactively:

```bash
# 3D Gaussian Splatting
python gui.py -m $OUTPUT_DIR/lego/3dgs -t render

# Relightable 3D Gaussian (NeILF)
python gui.py -m $OUTPUT_DIR/lego/neilf -t neilf
```

You can also add `--gui` to any `train.py` command to monitor training live.

---

## Verification Checklist

Run through this to confirm everything works:

```bash
source venv/bin/activate

# PyTorch + CUDA
python -c "import torch; assert torch.cuda.is_available(); print('OK: PyTorch', torch.__version__)"

# CUDA extensions
python -c "from simple_knn._C import distCUDA2; print('OK: simple-knn')"
python -c "import bvh; print('OK: bvh')"
python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings; print('OK: r3dg-rasterization')"
python -c "import nvdiffrast.torch; print('OK: nvdiffrast')"

# OptiX renderer
python -c "import sys; sys.path.insert(0, 'Optix/build'); import SampleRenderer; print('OK: SampleRenderer')"

# Python dependencies
python -c "import kornia, lpips, OpenEXR, pyexr, skimage; print('OK: all Python deps')"

# Entrypoints
python train.py --help
python eval_nvs.py --help
python relighting.py --help
```

---

## Troubleshooting

### CUDA version mismatch

```bash
nvcc --version                                        # CUDA Toolkit
nvidia-smi                                            # driver CUDA version
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA version
```

All must be compatible (e.g., driver supports >= 11.8, toolkit is 11.8, PyTorch was built for 11.8).

### OptiX: "OptiX headers not found"

Ensure `OptiX_INSTALL_DIR` is set and valid:

```bash
echo $OptiX_INSTALL_DIR
ls $OptiX_INSTALL_DIR/include/optix.h
```

### Slang: "slangc not found"

Ensure `SLANG_DIR` is set:

```bash
echo $SLANG_DIR
ls $SLANG_DIR/bin/slangc
```

### CUDA extension build errors

- Check `CUDA_HOME`: `echo $CUDA_HOME && ls $CUDA_HOME/bin/nvcc`
- GCC compatibility: CUDA 11.8 supports GCC up to 11
- Explicit arch list: `export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"`

### Blender headless fails

```bash
export PYOPENGL_PLATFORM=egl
export CYCLES_CUDA_EXTRA_CFLAGS="-I${CUDA_HOME:-/usr/local/cuda}/include"
$BLENDER_BIN --version
```

### SampleRenderer import fails

1. Check the `.so` exists: `ls Optix/build/SampleRenderer*.so`
2. Ensure `LD_LIBRARY_PATH` includes `$CUDA_HOME/lib64` and `Optix/build/`
3. Rebuild: `cd Optix/build && make -j && cd ../..`

### torch_scatter install fails

```bash
pip install torch_scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu118.html
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceeding{
matspray,
author =
{Langsteiner, Philipp and Dihlmann, Jan-Niklas and Lensch, Hendrik P.A.},
title =
{MatSpray: Fusing 2D Material World Knowledge on 3D Geometry},
booktitle =
{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month =
{June},
year =
{2026}
}
```

## Acknowledgements

This codebase builds on several excellent projects:

- **[Relightable 3D Gaussian](https://nju-3dv.github.io/projects/Relightable3DGaussian/)** (Gao et al., ECCV 2024) -- the core framework for relightable Gaussian Splatting that this code is based on.
- **[OptiX 7 Course](https://github.com/ingowald/optix7course)** (Wald, Siggraph 2019/2020) -- the foundation for the OptiX ray-tracing renderer in the `Optix/` directory.
- **[SSS GS](https://sss.jdihlmann.com/)** (Dihlmann et al., NeurIPS 2024) -- parts of our deferred rendering pipeline are adapted from their subsurface scattering Gaussian Splatting work.
- **[DiffusionRenderer](https://github.com/nv-tlabs/diffusion-renderer)** (Liang, Wang et al., CVPR 2025 Oral) -- the SVD-based material predictor used to generate per-view PBR material maps for supervision.

```bibtex
@inproceedings{R3DG2024,
    author    = {Gao, Jian and Gu, Chun and Lin, Youtian and Li, Zhihao and Zhu, Hao and Cao, Xun and Zhang, Li and Yao, Yao},
    title     = {Relightable 3D Gaussians: Realistic Point Cloud Relighting with BRDF Decomposition and Ray Tracing},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2024},
}

@inproceedings{sss_gs,
    author    = {Dihlmann, Jan-Niklas and Engelhardt, Andreas and Majumdar, Arjun and Braun, Raphael and Lensch, Hendrik P.A.},
    title     = {Subsurface Scattering for Gaussian Splatting},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year      = {2024},
}

@inproceedings{DiffusionRenderer,
    author    = {Ruofan Liang and Zan Gojcic and Huan Ling and Jacob Munkberg and
        Jon Hasselgren and Zhi-Hao Lin and Jun Gao and Alexander Keller and
        Nandita Vijaykumar and Sanja Fidler and Zian Wang},
    title     = {DiffusionRenderer: Neural Inverse and Forward Rendering with Video Diffusion Models},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
}
```
