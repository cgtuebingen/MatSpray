#!/bin/bash

set -e

# Initialize flags to false
RENDER_GTS=false
RUN_BASE_3DGS=false
RUN_BASE_NEILF=false
RUN_NEILF_SUPERVISION=false
RUN_PURE_SUPERVISION=false
RUN_PROJECTED_AVERAGE=false
RUN_MLP=true

# Optional: Diffusion Renderer
RUN_DIFFUSION_RENDERER=false
SRGB_TO_LINEAR=false

# Optional: use preprocessed GS outputs (single cameras.json + 30000-iter point_cloud.ply)
USE_PREPROCESSED_GS=true
GS_PLY_PATH=""
CAMERAS_JSON_PATH=""

SKIP_FIRST_TRAINING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --render-gts|--render_gts)
            RENDER_GTS=true
            shift
            ;;
        --use-preprocessed-gs|--use_preprocessed_gs)
            USE_PREPROCESSED_GS=true
            shift
            ;;
        --gs-ply|--gs_ply)
            GS_PLY_PATH="$2"
            shift 2
            ;;
        --cameras-json|--cameras_json)
            CAMERAS_JSON_PATH="$2"
            shift 2
            ;;
        --run-diffusion-renderer|--run_diffusion_renderer)
            RUN_DIFFUSION_RENDERER=true
            shift
            ;;
        --srgb-to-linear|--srgb_to_linear)
            SRGB_TO_LINEAR=true
            shift
            ;;
        --run-base-3dgs|--run_base_3dgs)
            RUN_BASE_3DGS=true
            shift
            ;;
        --run-base-neilf|--run_base_neilf)
            RUN_BASE_NEILF=true
            shift
            ;;
        --run-neilf-supervision|--run_neilf_supervision)
            RUN_NEILF_SUPERVISION=true
            shift
            ;;
        --run-pure-supervision|--run_pure_supervision)
            RUN_PURE_SUPERVISION=true
            shift
            ;;
        --run-projected-average|--run_projected_average)
            RUN_PROJECTED_AVERAGE=true
            shift
            ;;
        --run-mlp|--run_mlp)
            RUN_MLP=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --render-gts              Generate ground truth data with Blender"
            echo "  --run-base-3dgs          Run base 3DGS training"
            echo "  --run-base-neilf         Run base NeILF training"
            echo "  --run-neilf-supervision  Run NeILF with supervision"
            echo "  --run-pure-supervision   Run pure supervision"
            echo "  --run-projected-average  Run projected average"
            echo "  --run-mlp                Run MLP training"
            echo "  --help, -h               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display which flags are enabled
echo "=== Configuration ==="
echo "Render GTs: $RENDER_GTS"
echo "Run Diffusion Renderer: $RUN_DIFFUSION_RENDERER"
echo "Run Base 3DGS: $RUN_BASE_3DGS"
echo "Run Base NeILF: $RUN_BASE_NEILF"
echo "Run NeILF Supervision: $RUN_NEILF_SUPERVISION"
echo "Run Pure Supervision: $RUN_PURE_SUPERVISION"
echo "Run Projected Average: $RUN_PROJECTED_AVERAGE"
echo "Run MLP: $RUN_MLP"
echo "sRGB to Linear: $SRGB_TO_LINEAR"
echo "Use preprocessed GS (PLY+single transforms.json): $USE_PREPROCESSED_GS"
if [ "$USE_PREPROCESSED_GS" = true ]; then
    echo "  GS PLY (override, optional): ${GS_PLY_PATH:-<auto>}"
    echo "  cameras.json (override, optional): ${CAMERAS_JSON_PATH:-<auto>}"
fi
echo "===================="

# Autodetect Python interpreter
if command -v python3 >/dev/null 2>&1; then
	PY="$(command -v python3)"
else
	echo "ERROR: No python3 interpreter found" >&2; exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BLENDER_BIN="${BLENDER_BIN:-blender}"

# Set CUDA memory allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU Environment Setup for Headless Blender
export PYOPENGL_PLATFORM=egl
export CYCLES_CUDA_EXTRA_CFLAGS="-I${CUDA_HOME:-/usr/local/cuda}/include"

# Check GPU availability
echo "=== GPU Check ==="
nvidia-smi || echo "WARNING: nvidia-smi not available"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=================="

cd "$PROJECT_ROOT/Optix/build"
make -j
cd -

# Base directories (COLMAP dataset layout)
root_dir="${ROOT_DIR:-/path/to/datasets/navi/}"
output_dir="${OUTPUT_DIR:-/path/to/results/navi/}"
list=("pumpkin_showpiece_s" "3d_dollhouse_sink" "box_multi_door_colored" "bunny_racer" "can_kernel_corn" "chicken_racer" "circo_fish_toothbrush_holder_14995988" "dino_4" "dino_5" "duck_bath_yellow_s" "fire_engine_toy_red_yellow_s" "garbage_truck_green_toy_s" "hand_drill_cordless_blue_black" "hut_mushrooms_showpiece" "ice_cream_cart_showpiece" "keywest_showpiece_s" "paper_weight_flowers_showpiece" "remote_control_toy_car_s" "schleich_african_black_rhino" "schleich_bald_eagle" "schleich_hereford_bull" "schleich_lion_action_figure" "schleich_spinosaurus_action_figure" "school_bus" "soldier_wood_showpiece" "steps_small_showpiece" "tractor_green_showpiece" "tumbler_air_balloon" "water_gun_toy_green" "water_gun_toy_white" "water_gun_toy_yellow" "weisshai_great_white_shark" "welcome_sign_mushrooms" "well_with_leaf_roof_showpiece" "bottle_vitamin_d_tablets")

# Timing functions
start_timer() {
    echo $(date +%s.%N)
}

end_timer() {
    local start_time=$1
    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)
    echo "$elapsed"
}

write_timing() {
    local dataset=$1
    local step_name=$2
    local elapsed_time=$3
    local timing_dir="$output_dir/$dataset/timings"
    mkdir -p "$timing_dir"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $step_name: ${elapsed_time}s" >> "$timing_dir/timings.txt"
    echo "$step_name,${elapsed_time}" >> "$timing_dir/timings.csv"
}

# Initialize CSV header for timings
init_timing_csv() {
    local dataset=$1
    local timing_dir="$output_dir/$dataset/timings"
    mkdir -p "$timing_dir"
    if [ ! -f "$timing_dir/timings.csv" ]; then
        echo "step_name,elapsed_seconds" > "$timing_dir/timings.csv"
    fi
}

# Set the list of models to iterate through
models=("diffusionrenderer")

# Generate ground truth data with Blender first
if [ "$RENDER_GTS" = true ]; then
    echo "====== ENABLED: Generating ground truth data ======"
    for i in "${list[@]}"; do
        echo "====== Generating ground truth for dataset: $i ======"
        init_timing_csv "$i"
        
        blend_file_name="$i"
        
        echo "Running Blender render for $i..."
        start_time=$(start_timer)
        "$BLENDER_BIN" --background $root_dir/blend_files/$blend_file_name.blend --python blender_scripts/blender_render_script_test.py -- --env_dir $root_dir/env_maps/ --results_path $root_dir$i
        elapsed=$(end_timer "$start_time")
        write_timing "$i" "render_gt_$i" "$elapsed"

        echo "Ground truth generation completed for $i"
    done
else
    echo "====== SKIPPED: Generating ground truth data (use --render-gts to enable) ======"
fi

echo "====== Starting model training pipeline ======"

# Track PLY files used per scene for verification
declare -A PLY_FILES_USED

FIRST_ELEMENT=true
for i in "${list[@]}"; do
    echo "====== Processing dataset: $i ======"
    init_timing_csv "$i"

    # Optional: skip all training for the first dataset when SKIP_FIRST_TRAINING=true
    SKIP_TRAINING=false
    if [ "$FIRST_ELEMENT" = true ] && [ "$SKIP_FIRST_TRAINING" = true ]; then
        SKIP_TRAINING=true
        echo "[TEST] Skipping training calls for first element: $i"
    fi
    FIRST_ELEMENT=false

    # Input images (COLMAP layout)
    input_dir="$root_dir/$i/base/images_bg/"
    output_base_dir="$root_dir/$i/base/"

    # Run Diffusion Renderer to generate material properties
    if [ "$RUN_DIFFUSION_RENDERER" = true ]; then
        echo "====== ENABLED: Running Diffusion Renderer for $i ======"
        start_time=$(start_timer)
        cd "${DIFFUSION_RENDERER_DIR:-/path/to/external/diffusion-renderer}"
        
        # Build the command with conditional srgb_to_linear parameter
        cmd="$PY inference_svd_rgbx_improved.py --config configs/rgbx_inference.yaml \
            inference_input_dir=\"$input_dir\" \
            inference_save_dir=\"$output_base_dir\" \
            inference_n_frames=20 inference_n_steps=20 model_passes=\"['basecolor','normal','metallic','roughness']\" inference_res=\"[512,512]\" chunk_mode=\"all\""
        
        # Add srgb_to_linear parameter if flag is enabled
        if [ "$SRGB_TO_LINEAR" = true ]; then
            cmd="$cmd srgb_to_linear=true"
            echo "  Enabling sRGB to linear transformation"
        fi
        
        # Execute the command
        eval "$cmd"
        
        cd -
        elapsed=$(end_timer "$start_time")
        write_timing "$i" "diffusion_renderer_$i" "$elapsed"
        echo "Diffusion Renderer processing completed for $i"
    else
        echo "====== SKIPPED: Diffusion Renderer processing (use --run-diffusion-renderer to enable) ======"
    fi

    # Now continue with the NeILF processing
    echo "Running 3DGS and NeILF processing for $i..."

    # Sync cameras/PLY into dataset base if produced under output_dir
    start_time=$(start_timer)
    if [ -f "$output_dir/$i/base/cameras.json" ] && [ ! -f "$root_dir/$i/base/cameras.json" ]; then
        echo "[sync] Copying cameras.json to dataset: $root_dir/$i/base/"
        mkdir -p "$root_dir/$i/base"
        cp -n "$output_dir/$i/base/cameras.json" "$root_dir/$i/base/cameras.json" || true
    fi
    if [ -f "$output_dir/$i/base/points3d.ply" ] && [ ! -f "$root_dir/$i/base/points3d.ply" ]; then
        echo "[sync] Copying points3d.ply to dataset: $root_dir/$i/base/"
        mkdir -p "$root_dir/$i/base"
        cp -n "$output_dir/$i/base/points3d.ply" "$root_dir/$i/base/points3d.ply" || true
    fi
    # If only iteration PLY exists, copy it as points3d.ply
    if [ ! -f "$root_dir/$i/base/points3d.ply" ]; then
        if [ -f "$output_dir/$i/base/point_cloud/iteration_30000/point_cloud.ply" ]; then
            echo "[sync] Seeding dataset points3d.ply from iteration_30000 point_cloud.ply"
            cp -n "$output_dir/$i/base/point_cloud/iteration_30000/point_cloud.ply" "$root_dir/$i/base/points3d.ply" || true
        fi
    fi
    elapsed=$(end_timer "$start_time")
    if (( $(echo "$elapsed > 0.001" | bc -l) )); then
        write_timing "$i" "sync_cameras_ply_$i" "$elapsed"
    fi

    # Base 3DGS (paths use COLMAP layout under each scene)
    if [ "$RUN_BASE_3DGS" = true ]; then
        echo "====== ENABLED: Running base 3DGS for $i ======"
        GS_OUTPUT_DIR="$output_dir/$i/base"
        mkdir -p "$GS_OUTPUT_DIR"

        CURRENT_DATASET="$i"
        CURRENT_DATA_DIR="$root_dir/$CURRENT_DATASET/base"
        CURRENT_OUTPUT_ROOT="$output_dir/$CURRENT_DATASET"

        echo "[3DGS] Processing dataset: $CURRENT_DATASET"
        echo "[3DGS] Data directory: $CURRENT_DATA_DIR"
        echo "[3DGS] Output root: $CURRENT_OUTPUT_ROOT"
        echo "[3DGS] Expected output: $GS_OUTPUT_DIR"

        # Expected GS outputs for downstream steps
        if [ -d "$GS_OUTPUT_DIR" ] && [ -f "$GS_OUTPUT_DIR/cameras.json" ]; then
            echo "[3DGS] Gaussian Splatting output verified in: $GS_OUTPUT_DIR"
        else
            echo "[3DGS] WARNING: Expected output directory not found or incomplete: $GS_OUTPUT_DIR"
            echo "[3DGS] Checking for output in alternative locations..."
            WRONG_LOCATION="$output_dir/bottle_vitamin_d_tablets/base"
            if [ -d "$WRONG_LOCATION" ] && [ "$CURRENT_DATASET" != "bottle_vitamin_d_tablets" ]; then
                echo "[3DGS] ERROR: Output found in wrong location: $WRONG_LOCATION"
                echo "[3DGS] Current dataset: $CURRENT_DATASET"
                echo "[3DGS] Attempting to move output to correct location..."
                if [ ! -d "$GS_OUTPUT_DIR" ]; then
                    mkdir -p "$(dirname "$GS_OUTPUT_DIR")"
                    mv "$WRONG_LOCATION" "$GS_OUTPUT_DIR" 2>/dev/null || {
                        echo "[3DGS] Failed to move, copying instead..."
                        cp -r "$WRONG_LOCATION" "$GS_OUTPUT_DIR"
                    }
                    echo "[3DGS] Moved/copied output to correct location: $GS_OUTPUT_DIR"
                fi
            fi
        fi

        if [ "$USE_PREPROCESSED_GS" = true ]; then
            echo "[preprocessed] Using existing cameras.json and GS PLY for $i"
            
            DATASET_DIR=""
            PLY_SRC=""
            CURRENT_CAMERAS_JSON_PATH=""

            ORIGINAL_CAMERAS_JSON_PATH="${CAMERAS_JSON_PATH}"
            ORIGINAL_GS_PLY_PATH="${GS_PLY_PATH}"
            
            # cameras.json directory; override may use {scene} template
            if [ -n "$ORIGINAL_CAMERAS_JSON_PATH" ]; then
                T_OVERRIDE="${ORIGINAL_CAMERAS_JSON_PATH//\{scene\}/$i}"
                if [ -d "$T_OVERRIDE" ]; then
                    if [ -f "$T_OVERRIDE/cameras.json" ]; then
                        DATASET_DIR="$T_OVERRIDE"
                        CURRENT_CAMERAS_JSON_PATH="$T_OVERRIDE/cameras.json"
                    else
                        echo "[error] Directory provided to --cameras-json lacks cameras.json: $T_OVERRIDE"
                        exit 2
                    fi
                elif [ -f "$T_OVERRIDE" ]; then
                    DATASET_DIR="$(dirname "$T_OVERRIDE")"
                    CURRENT_CAMERAS_JSON_PATH="$T_OVERRIDE"
                else
                    echo "[warn] Provided --cameras-json (after template) not found: $T_OVERRIDE"
                fi
            fi
            if [ -z "$DATASET_DIR" ]; then
                if [ -f "$output_dir/$i/base/cameras.json" ]; then
                    DATASET_DIR="$output_dir/$i/base"
                    CURRENT_CAMERAS_JSON_PATH="$DATASET_DIR/cameras.json"
                elif [ -f "$root_dir/$i/base/cameras.json" ]; then
                    DATASET_DIR="$root_dir/$i/base"
                    CURRENT_CAMERAS_JSON_PATH="$DATASET_DIR/cameras.json"
                elif [ -f "$root_dir/$i/cameras.json" ]; then
                    DATASET_DIR="$root_dir/$i"
                    CURRENT_CAMERAS_JSON_PATH="$DATASET_DIR/cameras.json"
                else
                    DATASET_DIR="$root_dir/$i/base"
                fi
            fi
            if [ -z "$CURRENT_CAMERAS_JSON_PATH" ] || [ ! -f "$CURRENT_CAMERAS_JSON_PATH" ]; then
                echo "[error] cameras.json not found for scene '$i' (expected near: $DATASET_DIR)"
                exit 2
            fi
            echo "[preprocessed] Dataset directory: $DATASET_DIR"
            echo "[preprocessed] cameras.json: $CURRENT_CAMERAS_JSON_PATH"
            echo "[preprocessed] Scene: $i"

            # Images next to cameras.json (train/test) for Scene
            mkdir -p "$DATASET_DIR/train/images" "$DATASET_DIR/test/images"
            IMAGE_SOURCE=""
            if [ -d "$DATASET_DIR/images" ]; then
                IMAGE_SOURCE="$DATASET_DIR/images"
            elif [ -d "$root_dir/$i/base/images" ]; then
                IMAGE_SOURCE="$root_dir/$i/base/images"
            elif [ -d "$root_dir/$i/images" ]; then
                IMAGE_SOURCE="$root_dir/$i/images"
            else
                echo "[warn] No images directory found near cameras.json or root_dir; proceeding anyway."
            fi

            if [ -n "$IMAGE_SOURCE" ] && [ -z "$(ls -A "$DATASET_DIR/train/images" 2>/dev/null)" ]; then
                echo "[preprocessed] Populating train/ and test/ images from: $IMAGE_SOURCE"
                shopt -s nullglob
                for f in "$IMAGE_SOURCE"/*; do
                    base="$(basename "$f")"
                    cp -n "$f" "$DATASET_DIR/train/images/$base" || true
                    cp -n "$f" "$DATASET_DIR/test/images/$base" || true
                done
                shopt -u nullglob
            else
                echo "[preprocessed] train/images already populated or no source; skipping image copy."
            fi

            if [ -n "$ORIGINAL_GS_PLY_PATH" ]; then
                G_OVERRIDE="${ORIGINAL_GS_PLY_PATH//\{scene\}/$i}"
                echo "[preprocessed] Resolving PLY from GS_PLY_PATH template: $ORIGINAL_GS_PLY_PATH -> $G_OVERRIDE"
                if [ -d "$G_OVERRIDE" ]; then
                    mapfile -t _pcd_candidates < <(find "$G_OVERRIDE" -type f -path "*/point_cloud/iteration_*/point_cloud.ply" 2>/dev/null | sort -V || true)
                    if [[ ${#_pcd_candidates[@]} -gt 0 ]]; then
                        PLY_SRC="${_pcd_candidates[-1]}"
                    fi
                elif [ -f "$G_OVERRIDE" ]; then
                    PLY_SRC="$G_OVERRIDE"
                else
                    echo "[warn] Provided --gs-ply (after template) not found: $G_OVERRIDE"
                    echo "[warn] Falling back to auto-detection for scene '$i'"
                fi
            fi
            if [ -z "$PLY_SRC" ] || [ ! -f "$PLY_SRC" ]; then
                SCENE_PLY_DIR="$output_dir/$i/base"
                echo "[preprocessed] Auto-detecting PLY for scene '$i'"
                echo "[preprocessed]   Searching in: $SCENE_PLY_DIR"
                
                if [ ! -d "$SCENE_PLY_DIR" ]; then
                    echo "[warn] Scene PLY directory does not exist: $SCENE_PLY_DIR"
                fi
                
                if [ -f "$SCENE_PLY_DIR/point_cloud/iteration_30000/point_cloud.ply" ]; then
                    PLY_SRC="$SCENE_PLY_DIR/point_cloud/iteration_30000/point_cloud.ply"
                    echo "[preprocessed] Found iteration_30000 PLY for scene '$i'"
                elif [ -d "$SCENE_PLY_DIR/point_cloud" ]; then
                    echo "[preprocessed] Searching for latest iteration PLY in: $SCENE_PLY_DIR/point_cloud"
                    mapfile -t _pcd_candidates < <(find "$SCENE_PLY_DIR/point_cloud" -maxdepth 3 -type f -name "point_cloud.ply" 2>/dev/null | sort -V || true)
                    if [[ ${#_pcd_candidates[@]} -gt 0 ]]; then
                        PLY_SRC="${_pcd_candidates[-1]}"
                        echo "[preprocessed] Found latest iteration PLY for scene '$i': ${#_pcd_candidates[@]} candidates"
                        echo "[preprocessed]   Selected: $PLY_SRC"
                    else
                        echo "[warn] No PLY candidates found in $SCENE_PLY_DIR/point_cloud"
                    fi
                else
                    echo "[warn] Point cloud directory does not exist: $SCENE_PLY_DIR/point_cloud"
                fi
            fi
            if [ -z "$PLY_SRC" ] || [ ! -f "$PLY_SRC" ]; then
                echo "[error] Could not locate GS point_cloud.ply for scene '$i'"
                echo "[error]   Searched in: $output_dir/$i/base/point_cloud/"
                echo "[error]   Set --gs-ply with {scene} template or ensure PLY exists for each scene"
                exit 3
            fi
            echo "[preprocessed] Using GS PLY for scene '$i': $PLY_SRC"

            if [ -f "$PLY_SRC" ]; then
                PLY_CHECKSUM=$(md5sum "$PLY_SRC" | cut -d' ' -f1)
                PLY_FILES_USED["$i"]="$PLY_SRC|$PLY_CHECKSUM"
                echo "[preprocessed] PLY checksum for scene '$i': $PLY_CHECKSUM"
            else
                echo "[error] PLY file does not exist: $PLY_SRC"
                exit 3
            fi

            cp -f "$PLY_SRC" "$DATASET_DIR/points3d.ply"
            echo "[preprocessed] Wrote: $DATASET_DIR/points3d.ply"

            if [ "$SKIP_TRAINING" = false ]; then
                start_time=$(start_timer)
                $PY train.py --eval \
                    -s "$DATASET_DIR" \
                    -m "$output_dir/$i/3dgs" \
                    --lambda_normal_render_depth 0.01 \
                    --lambda_normal_smooth 0.01 \
                    --lambda_mask_entropy 0.1 \
                    --save_training_vis \
                    --lambda_depth_var 1e-2 \
                    --enable_timelapse \
                    --save_timelapse \
                    --timelapse_interval 10 \
                    --timelapse_fps 10 \
                    --timelapse_camera_index 0
                elapsed=$(end_timer "$start_time")
                write_timing "$i" "train_3dgs_preprocessed_$i" "$elapsed"
            else
                echo "[TEST] SKIPPED: train.py call for $i"
            fi
        else
            if [ "$SKIP_TRAINING" = false ]; then
                start_time=$(start_timer)
                $PY train.py --eval \
                    -s $root_dir/$i/base/ \
                    -m $output_dir/$i/3dgs \
                    --lambda_normal_render_depth 0.01 \
                    --lambda_normal_smooth 0.01 \
                    --lambda_mask_entropy 0.1 \
                    --save_training_vis \
                    --lambda_depth_var 1e-2 \
                    --enable_timelapse \
                    --save_timelapse \
                    --timelapse_interval 10 \
                    --timelapse_fps 10 \
                    --timelapse_camera_index 0
                elapsed=$(end_timer "$start_time")
                write_timing "$i" "train_3dgs_$i" "$elapsed"
            else
                echo "[TEST] SKIPPED: train.py call for $i"
            fi
        fi
        echo "Base 3DGS training completed for $i"
    else
        echo "====== SKIPPED: Base 3DGS training (use --run-base-3dgs to enable) ======"
    fi

    # Base NeILF
    if [ "$RUN_BASE_NEILF" = true ]; then
        echo "====== ENABLED: Running base NeILF for $i ======"
        if [ "$SKIP_TRAINING" = false ]; then
            start_time=$(start_timer)
            $PY train.py --eval \
            -s $root_dir/$i/base/ \
            -m $output_dir/$i/neilf \
            -c $output_dir/$i/3dgs/chkpnt30000.pth \
            --save_training_vis \
            --position_lr_init 0.000016 \
            --position_lr_final 0.00000016 \
            --normal_lr 0.001 \
            --sh_lr 0.00025 \
            --opacity_lr 0.005 \
            --scaling_lr 0.0005 \
            --rotation_lr 0.0001 \
            --iterations 40000 \
            --lambda_base_color_smooth 0 \
            --lambda_roughness_smooth 0 \
            --lambda_light_smooth 0 \
            --lambda_light 0.01 \
            --lambda_base_color_buffer 0.0 \
            --lambda_roughness_buffer 0.0 \
            --lambda_diffuse_buffer 0.0 \
            -t neilf --sample_num 32 \
            --save_training_vis_iteration 200 \
            --lambda_env_smooth 0.01 
            elapsed=$(end_timer "$start_time")
            write_timing "$i" "train_base_neilf_$i" "$elapsed"

            start_time=$(start_timer)
            $PY eval_nvs.py --eval \
                -m $output_dir/${i}/neilf \
                -c $output_dir/${i}/neilf/chkpnt40000.pth \
                -t neilf
            elapsed=$(end_timer "$start_time")
            write_timing "$i" "eval_base_neilf_$i" "$elapsed"

            echo "Running relighting for ${i} with neilf..."
            echo "Relighting for ${i} with neilf completed"
        else
            echo "[TEST] SKIPPED: Base NeILF training calls for $i"
        fi
    else
        echo "====== SKIPPED: Base NeILF training (use --run-base-neilf to enable) ======"
    fi

    # Process each model with NeILF supervision
    if [ "$RUN_NEILF_SUPERVISION" = true ]; then
        echo "====== ENABLED: Processing NeILF with supervision ======"
        for model_name in "${models[@]}"; do
            echo "Processing NeILF for model: $model_name"

            # Define all NeILF configurations for the current model
            declare -A neilf_configs
            neilf_configs=(
                ["neilf_BaseColor_Roughness_Metallic_${model_name}"]="1.0 1.0 1.0"
            )

            # Run for each NeILF configuration
            for neilf_name in "${!neilf_configs[@]}"; do
                echo "Running $neilf_name for $i..."

                # Get the buffer values for this configuration
                read -r base_color_buffer roughness_buffer metallic_buffer <<< "${neilf_configs[$neilf_name]}"

                if [ "$SKIP_TRAINING" = false ]; then
                    start_time=$(start_timer)
                    $PY train.py --eval \
                        -s $root_dir/$i/ \
                        -m $output_dir/$i/$neilf_name \
                        -c $output_dir/$i/3dgs/chkpnt30000.pth \
                        --base_color_folder /${model_name,,}_baseColor/ \
                        --roughness_folder /${model_name,,}_roughness/ \
                        --metallic_folder /${model_name,,}_metallic/ \
                        --save_training_vis \
                        --position_lr_init 0.0 \
                        --position_lr_final 0.0 \
                        --normal_lr 0.001 \
                        --sh_lr 0.00025 \
                        --opacity_lr 0.0 \
                        --scaling_lr 0.0 \
                        --rotation_lr 0.0 \
                        --iterations 40000 \
                        --lambda_base_color_smooth 0 \
                        --lambda_roughness_smooth 0 \
                        --lambda_light_smooth 0 \
                        --lambda_light 0.01 \
                        --lambda_base_color_buffer $base_color_buffer \
                        --lambda_roughness_buffer $roughness_buffer \
                        --lambda_metallic_buffer $metallic_buffer \
                        -t neilf --sample_num 32 \
                        --save_training_vis_iteration 200 \
                        --lambda_env_smooth 0.01 
                    elapsed=$(end_timer "$start_time")
                    write_timing "$i" "train_${neilf_name}_$i" "$elapsed"

                    start_time=$(start_timer)
                    $PY eval_nvs.py --eval \
                        -m $output_dir/${i}/$neilf_name \
                        -c $output_dir/${i}/$neilf_name/chkpnt40000.pth \
                        -t neilf
                    elapsed=$(end_timer "$start_time")
                    write_timing "$i" "eval_${neilf_name}_$i" "$elapsed"

                    # Run relighting for different environment maps
                    echo "Running relighting for ${i} with $neilf_name..."
                    # Define environment maps as an array
                    envmaps=("envmap3" "envmap6" "envmap12" "envmap24")

                    # Iterate through each environment map
                    for envmap in "${envmaps[@]}"; do
                        echo "Processing ${i} with $neilf_name and $envmap..."

                        # Ensure cameras.json exists in the envmap directory by seeding it from known locations
                        env_dir="$root_dir/$i/$envmap"
                        mkdir -p "$env_dir"
                        if [ ! -f "$env_dir/cameras.json" ]; then
                            for src in "$root_dir/$i/base/cameras.json" "$root_dir/$i/cameras.json" "$output_dir/$i/base/cameras.json"; do
                                if [ -f "$src" ]; then
                                    cp -f "$src" "$env_dir/cameras.json"
                                    echo "[relighting] Seeded cameras.json to $env_dir from $src"
                                    break
                                fi
                            done
                        fi

                        start_time=$(start_timer)
                        $PY relighting.py \
                            -co "$root_dir/$i/$envmap/" \
                            --video \
                            --output "$output_dir/$i/$neilf_name/$envmap" \
                            --envmap_path "env_map/$envmap.exr" \
                            --capture_list "pbr_env,normal,base_color,visibility,roughness,metallic,diffuse,specular" \
                            --sample_num 64 \
                            --ply_path "$output_dir/$i/$neilf_name/point_cloud/iteration_40000/point_cloud.ply" \
                            --gt_path "$root_dir/$i/$envmap/test/images/"
                        elapsed=$(end_timer "$start_time")
                        write_timing "$i" "relighting_${neilf_name}_${envmap}_$i" "$elapsed"

                        echo "Finished processing ${i} with $neilf_name and $envmap"
                    done
                    echo "Relighting for ${i} with $neilf_name completed"
                else
                    echo "[TEST] SKIPPED: NeILF supervision training calls for $neilf_name ($i)"
                fi
            done
        done
    else
        echo "====== SKIPPED: NeILF with supervision (use --run-neilf-supervision to enable) ======"
    fi

    # Process each model with pure supervision
    if [ "$RUN_PURE_SUPERVISION" = true ]; then
        echo "====== ENABLED: Processing pure supervision ======"
        for model_name in "${models[@]}"; do
            echo "Processing NeILF for model: $model_name"

            # Define all NeILF configurations for the current model
            declare -A neilf_configs
            neilf_configs=(
                ["pure_supervision_BaseColor_Roughness_Metallic_${model_name}"]="1.0 1.0 1.0"
            )

            # Run for each NeILF configuration
            for neilf_name in "${!neilf_configs[@]}"; do
                echo "Running $neilf_name for $i..."

                # Get the buffer values for this configuration
                read -r base_color_buffer roughness_buffer metallic_buffer <<< "${neilf_configs[$neilf_name]}"

                if [ "$SKIP_TRAINING" = false ]; then
                    start_time=$(start_timer)
                    $PY train.py --eval \
                        -s $root_dir/$i/ \
                        -m $output_dir/$i/$neilf_name \
                        -c $output_dir/$i/3dgs/chkpnt30000.pth \
                        --base_color_folder /${model_name,,}_baseColor/ \
                        --roughness_folder /${model_name,,}_roughness/ \
                        --metallic_folder /${model_name,,}_metallic/ \
                        --save_training_vis \
                        --position_lr_init 0.0 \
                        --position_lr_final 0.0 \
                        --normal_lr 0.001 \
                        --sh_lr 0.00025 \
                        --opacity_lr 0.0 \
                        --scaling_lr 0.0 \
                        --rotation_lr 0.0 \
                        --iterations 40000 \
                        --lambda_base_color_smooth 0 \
                        --lambda_roughness_smooth 0 \
                        --lambda_light_smooth 0 \
                        --lambda_light 0.01 \
                        --lambda_base_color_buffer $base_color_buffer \
                        --lambda_roughness_buffer $roughness_buffer \
                        --lambda_metallic_buffer $metallic_buffer \
                        --detach_metallic \
                        --detach_roughness \
                        --detach_base_color \
                        -t neilf --sample_num 32 \
                        --save_training_vis_iteration 200 \
                        --lambda_env_smooth 0.01 
                    elapsed=$(end_timer "$start_time")
                    write_timing "$i" "train_${neilf_name}_$i" "$elapsed"

                    start_time=$(start_timer)
                    $PY eval_nvs.py --eval \
                        -m $output_dir/${i}/$neilf_name \
                        -c $output_dir/${i}/$neilf_name/chkpnt40000.pth \
                        -t neilf
                    elapsed=$(end_timer "$start_time")
                    write_timing "$i" "eval_${neilf_name}_$i" "$elapsed"

                    # Run relighting for different environment maps
                    echo "Running relighting for ${i} with $neilf_name..."
                    # Define environment maps as an array
                    envmaps=("envmap3" "envmap6" "envmap12" "envmap24")

                    # Iterate through each environment map
                    for envmap in "${envmaps[@]}"; do
                        echo "Processing ${i} with $neilf_name and $envmap..."

                        # Ensure cameras.json exists in the envmap directory by seeding it from known locations
                        env_dir="$root_dir/$i/$envmap"
                        mkdir -p "$env_dir"
                        if [ ! -f "$env_dir/cameras.json" ]; then
                            for src in "$root_dir/$i/base/cameras.json" "$root_dir/$i/cameras.json" "$output_dir/$i/base/cameras.json"; do
                                if [ -f "$src" ]; then
                                    cp -f "$src" "$env_dir/cameras.json"
                                    echo "[relighting] Seeded cameras.json to $env_dir from $src"
                                    break
                                fi
                            done
                        fi

                        start_time=$(start_timer)
                        $PY relighting.py \
                            -co "$root_dir/$i/$envmap/" \
                            --video \
                            --output "$output_dir/$i/$neilf_name/$envmap" \
                            --envmap_path "env_map/$envmap.exr" \
                            --capture_list "pbr_env,normal,base_color,visibility,roughness,metallic,diffuse,specular" \
                            --sample_num 64 \
                            --ply_path "$output_dir/$i/$neilf_name/point_cloud/iteration_40000/point_cloud.ply" \
                            --gt_path "$root_dir/$i/$envmap/test/images/"
                        elapsed=$(end_timer "$start_time")
                        write_timing "$i" "relighting_${neilf_name}_${envmap}_$i" "$elapsed"

                        echo "Finished processing ${i} with $neilf_name and $envmap"
                    done
                    echo "Relighting for ${i} with $neilf_name completed"
                else
                    echo "[TEST] SKIPPED: Pure supervision training calls for $neilf_name ($i)"
                fi
            done
        done
    else
        echo "====== SKIPPED: Pure supervision (use --run-pure-supervision to enable) ======"
    fi

    # Process each model with projected average
    if [ "$RUN_PROJECTED_AVERAGE" = true ]; then
        echo "====== ENABLED: Processing projected average ======"
        for model_name in "${models[@]}"; do
            echo "Processing NeILF for model: $model_name"

            # Define all NeILF configurations for the current model
            declare -A neilf_configs
            neilf_configs=(
                ["projected_Averaged_BaseColor_Roughness_Metallic_${model_name}"]="0.0 0.0 0.0"
            )

            # Run for each NeILF configuration
            for neilf_name in "${!neilf_configs[@]}"; do
                echo "Running $neilf_name for $i..."

                # Get the buffer values for this configuration
                read -r base_color_buffer roughness_buffer metallic_buffer <<< "${neilf_configs[$neilf_name]}"

                if [ "$SKIP_TRAINING" = false ]; then
                    start_time=$(start_timer)
                    $PY train.py --eval \
                        -s $root_dir/$i/ \
                        -m $output_dir/$i/$neilf_name \
                        -c $output_dir/$i/3dgs/chkpnt30000.pth \
                        --base_color_folder /${model_name,,}_baseColor/ \
                        --roughness_folder /${model_name,,}_roughness/ \
                        --metallic_folder /${model_name,,}_metallic/ \
                        --save_training_vis \
                        --position_lr_init 0.0 \
                        --position_lr_final 0.0 \
                        --normal_lr 0.001 \
                        --sh_lr 0.00025 \
                        --opacity_lr 0.0 \
                        --scaling_lr 0.0 \
                        --rotation_lr 0.0 \
                        --iterations 40000 \
                        --lambda_base_color_smooth 0 \
                        --lambda_roughness_smooth 0 \
                        --lambda_light_smooth 0 \
                        --lambda_light 0.01 \
                        --lambda_base_color_buffer $base_color_buffer \
                        --lambda_roughness_buffer $roughness_buffer \
                        --lambda_metallic_buffer $metallic_buffer \
                        --detach_metallic \
                        --detach_roughness \
                        --detach_base_color \
                        -t neilf --sample_num 32 \
                        --save_training_vis_iteration 200 \
                        --lambda_env_smooth 0.01 \
                        --perform_intersection_tracing 
                    elapsed=$(end_timer "$start_time")
                    write_timing "$i" "train_${neilf_name}_$i" "$elapsed"

                    start_time=$(start_timer)
                    $PY eval_nvs.py --eval \
                        -m $output_dir/${i}/$neilf_name \
                        -c $output_dir/${i}/$neilf_name/chkpnt40000.pth \
                        -t neilf
                    elapsed=$(end_timer "$start_time")
                    write_timing "$i" "eval_${neilf_name}_$i" "$elapsed"

                    # Run relighting for different environment maps
                    echo "Running relighting for ${i} with $neilf_name..."
                    # Define environment maps as an array
                    envmaps=("envmap3" "envmap6" "envmap12" "envmap24")

                    # Iterate through each environment map
                    for envmap in "${envmaps[@]}"; do
                        echo "Processing ${i} with $neilf_name and $envmap..."

                        # Ensure cameras.json exists in the envmap directory by seeding it from known locations
                        env_dir="$root_dir/$i/$envmap"
                        mkdir -p "$env_dir"
                        if [ ! -f "$env_dir/cameras.json" ]; then
                            for src in "$root_dir/$i/base/cameras.json" "$root_dir/$i/cameras.json" "$output_dir/$i/base/cameras.json"; do
                                if [ -f "$src" ]; then
                                    cp -f "$src" "$env_dir/cameras.json"
                                    echo "[relighting] Seeded cameras.json to $env_dir from $src"
                                    break
                                fi
                            done
                        fi

                        start_time=$(start_timer)
                        $PY relighting.py \
                            -co "$root_dir/$i/$envmap/" \
                            --video \
                            --output "$output_dir/$i/$neilf_name/$envmap" \
                            --envmap_path "env_map/$envmap.exr" \
                            --capture_list "pbr_env,normal,base_color,visibility,roughness,metallic,diffuse,specular" \
                            --sample_num 64 \
                            --ply_path "$output_dir/$i/$neilf_name/point_cloud/iteration_40000/point_cloud.ply" \
                            --gt_path "$root_dir/$i/$envmap/test/images/"
                        elapsed=$(end_timer "$start_time")
                        write_timing "$i" "relighting_${neilf_name}_${envmap}_$i" "$elapsed"

                        echo "Finished processing ${i} with $neilf_name and $envmap"
                    done
                    echo "Relighting for ${i} with $neilf_name completed"
                else
                    echo "[TEST] SKIPPED: Projected average training calls for $neilf_name ($i)"
                fi
            done
        done
    else
        echo "====== SKIPPED: Projected average (use --run-projected-average to enable) ======"
    fi

    # Process each model with MLPs
    if [ "$RUN_MLP" = true ]; then
        echo "====== ENABLED: Processing MLP NeILF for model: $model_name ======"
        for model_name in "${models[@]}"; do
            echo "Processing MLP NeILF for model: $model_name"

            # Define all NeILF configurations for the current model
            declare -A neilf_configs
            neilf_configs=(
                ["mlp_BaseColor_Roughness_Metallic_Deferred_${model_name}"]="1.0 1.0 1.0"
            )

            # Run for each NeILF configuration
            for neilf_name in "${!neilf_configs[@]}"; do
                echo "Running $neilf_name for $i..."

                # Get the buffer values for this configuration
                read -r base_color_buffer roughness_buffer metallic_buffer <<< "${neilf_configs[$neilf_name]}"

                if [ "$SKIP_TRAINING" = false ]; then
                    start_time=$(start_timer)
                    $PY train.py --eval \
                        -s $root_dir/$i/base/ \
                        -m $output_dir/$i/$neilf_name \
                        -c $output_dir/$i/3dgs/chkpnt30000.pth \
                        --base_color_folder /${model_name,,}_baseColor/ \
                        --roughness_folder /${model_name,,}_roughness/ \
                        --metallic_folder /${model_name,,}_metallic/ \
                        --save_training_vis \
                        --position_lr_init 0.0 \
                        --position_lr_final 0.0 \
                        --normal_lr 0.0 \
                        --sh_lr 0.00025 \
                        --opacity_lr 0.0 \
                        --scaling_lr 0.0 \
                        --rotation_lr 0.0 \
                        --iterations 40000 \
                        --lambda_base_color_smooth 0 \
                        --lambda_roughness_smooth 0 \
                        --lambda_light_smooth 0 \
                        --lambda_light 0.01 \
                        --lambda_base_color_buffer $base_color_buffer \
                        --lambda_roughness_buffer $roughness_buffer \
                        --lambda_metallic_buffer $metallic_buffer \
                        -t neilf_deferred --sample_num 64 \
                        --save_training_vis_iteration 200 \
                        --lambda_env_smooth 0.01 \
                        --perform_intersection_tracing \
                        --base_color_lr 0.001 \
                        --roughness_lr 0.001 \
                        --metallic_lr 0.001 \
                        --use_mlp \
                        --save_interval 1000 \
                        --checkpoint_interval 1000 
                    elapsed=$(end_timer "$start_time")
                    write_timing "$i" "train_${neilf_name}_$i" "$elapsed"

                    # Run relighting for different environment maps
                    echo "Running relighting for ${i} with $neilf_name..."
                    # Define environment maps as an array
                    envmaps=("envmap3" "envmap6" "envmap12" "envmap24")

                    # Iterate through each environment map
                    for envmap in "${envmaps[@]}"; do
                        echo "Processing ${i} with $neilf_name and $envmap..."

                        # Ensure cameras.json exists in the envmap directory by seeding it from known locations
                        env_dir="$root_dir/$i/$envmap"
                        mkdir -p "$env_dir"
                        if [ ! -f "$env_dir/cameras.json" ]; then
                            for src in "$root_dir/$i/base/cameras.json" "$root_dir/$i/cameras.json" "$output_dir/$i/base/cameras.json"; do
                                if [ -f "$src" ]; then
                                    cp -f "$src" "$env_dir/cameras.json"
                                    echo "[relighting] Seeded cameras.json to $env_dir from $src"
                                    break
                                fi
                            done
                        fi

                        start_time=$(start_timer)
                        $PY relighting.py \
                            -co "$root_dir/$i/$envmap/" \
                            --video \
                            --output "$output_dir/$i/$neilf_name/$envmap" \
                            --envmap_path "env_map/$envmap.exr" \
                            --capture_list "pbr_env,normal,base_color,visibility,roughness,metallic,diffuse,specular" \
                            --sample_num 512 \
                            --ply_path "$output_dir/$i/$neilf_name/point_cloud/iteration_40000/point_cloud.ply" \
                            --gt_path "$root_dir/$i/$envmap/test/images/" \
                            -t neilf_deferred
                        elapsed=$(end_timer "$start_time")
                        write_timing "$i" "relighting_${neilf_name}_${envmap}_$i" "$elapsed"

                        echo "Finished processing ${i} with $neilf_name and $envmap"
                    done
                    echo "Relighting for ${i} with $neilf_name completed"
                else
                    echo "[TEST] SKIPPED: MLP training calls for $neilf_name ($i)"
                fi
            done
        done
    else
        echo "====== SKIPPED: Processing MLP NeILF (use --run-mlp to enable) ======"
    fi

    echo "====== Completed all processing for dataset: $i ======"
done

# Verification: Check that each scene used a unique PLY file
echo ""
echo "====== PLY File Usage Verification ======"
if [ ${#PLY_FILES_USED[@]} -eq 0 ]; then
    echo "[verify] No PLY files were tracked (USE_PREPROCESSED_GS may be false)"
else
    # Track checksums to detect duplicates
    declare -A CHECKSUM_TO_SCENES
    DUPLICATE_FOUND=false
    
    for scene in "${!PLY_FILES_USED[@]}"; do
        IFS='|' read -r ply_path checksum <<< "${PLY_FILES_USED[$scene]}"
        echo "[verify] Scene '$scene':"
        echo "         PLY: $ply_path"
        echo "         Checksum: $checksum"
        
        # Check for duplicate checksums
        if [ -n "${CHECKSUM_TO_SCENES[$checksum]}" ]; then
            DUPLICATE_FOUND=true
            echo "[WARNING] Scene '$scene' uses the SAME PLY as: ${CHECKSUM_TO_SCENES[$checksum]}"
            echo "          This indicates a bug - each scene should use its own PLY file!"
        else
            CHECKSUM_TO_SCENES["$checksum"]="$scene"
        fi
    done
    
    if [ "$DUPLICATE_FOUND" = false ]; then
        echo "[verify] ✓ SUCCESS: All scenes used unique PLY files (${#PLY_FILES_USED[@]} scenes verified)"
    else
        echo "[verify] ✗ FAILURE: Some scenes are sharing PLY files - this is a bug!"
        exit 1
    fi
fi
echo "=========================================="
echo ""

echo "All processing completed." 