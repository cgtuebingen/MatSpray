#!/bin/bash

# Ensure the script runs with bash (re-exec if invoked with sh)
if [ -z "$BASH_VERSION" ]; then
	exec /usr/bin/env bash "$0" "$@"
fi

set -e

# Autodetect Python interpreter
if command -v python3 >/dev/null 2>&1; then
	PY="$(command -v python3)"
else
	echo "ERROR: No python3 interpreter found" >&2; exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BLENDER_BIN="${BLENDER_BIN:-blender}"

# Initialize flags to false
RENDER_GTS=false
SRGB_TO_LINEAR=false
RUN_DIFFUSION_RENDERER=true
RUN_BASE_3DGS=true
RUN_BASE_NEILF=true
RUN_NEILF_SUPERVISION=false
RUN_PURE_SUPERVISION=false
RUN_PROJECTED_AVERAGE=false
RUN_MLP=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --render-gts|--render_gts)
            RENDER_GTS=true
            shift
            ;;
        --run-diffusion-renderer|--run_diffusion_renderer)
            RUN_DIFFUSION_RENDERER=true
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
        --srgb-to-linear|--srgb_to_linear)
            SRGB_TO_LINEAR=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --render-gts              Generate ground truth data with Blender"
            echo "  --run-diffusion-renderer Run diffusion renderer processing"
            echo "  --run-base-3dgs          Run base 3DGS training"
            echo "  --run-base-neilf         Run base NeILF training"
            echo "  --run-neilf-supervision  Run NeILF with supervision"
            echo "  --run-pure-supervision   Run pure supervision"
            echo "  --run-projected-average  Run projected average"
            echo "  --run-mlp                Run MLP training"
            echo "  --srgb-to-linear        Enable sRGB to linear transformation for Diffusion Renderer"
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
echo "===================="

# Set CUDA memory allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_SHOW_CPP_STACKTRACES=1

# GPU Environment Setup for Headless Blender
export PYOPENGL_PLATFORM=egl
export CYCLES_CUDA_EXTRA_CFLAGS="-I${CUDA_HOME:-/usr/local/cuda}/include"

export DOTNV_EPS=0.001
export DOTNV_TOL=0.2

# Check GPU availability
echo "=== GPU Check ==="
nvidia-smi || echo "WARNING: nvidia-smi not available"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=================="

# Base directories
root_dir="${ROOT_DIR:-/path/to/datasets/stanford_orb}"
output_dir="${OUTPUT_DIR:-/path/to/results/stanford_orb}"
nerf_synthetic_dir="${NERF_SYNTHETIC_DIR:-/path/to/datasets/nerf_synthetic}"

# Copy env_maps from nerf_synthetic if they don't exist
env_maps_dir="$root_dir/env_maps"
if [ ! -d "$env_maps_dir" ]; then
    echo "Copying env_maps from nerf_synthetic to stanford_orb..."
    mkdir -p "$env_maps_dir"
    if [ -d "$nerf_synthetic_dir/env_maps" ]; then
        cp -r "$nerf_synthetic_dir/env_maps"/* "$env_maps_dir/"
        echo "Successfully copied env_maps"
    else
        echo "WARNING: nerf_synthetic/env_maps not found at $nerf_synthetic_dir/env_maps"
    fi
else
    echo "env_maps directory already exists at $env_maps_dir"
fi

# List of stanford_orb objects
list=("baking" "ball" "blocks" "cactus" "car" "chips" "cup" "curry" "gnome" "grogu" "pepsi" "pitcher" "salt" "teapot")

# Model id for supervision / MLP run names
models=("diffusionrenderer")

# Generate ground truth data with Blender first
if [ "$RENDER_GTS" = true ]; then
    echo "====== ENABLED: Generating ground truth data ======"
    for i in "${list[@]}"; do
        echo "====== Generating ground truth for dataset: $i ======"
        
        # Find the first scene for this object (any scene will do)
        scene_name=$(find "$root_dir" -type d -name "${i}_scene*" | head -1 | xargs basename)
        
        if [ -z "$scene_name" ]; then
            echo "ERROR: No scene found for object $i, skipping..."
            continue
        fi
        
        obj_file="$root_dir/$scene_name/mesh_blender/mesh.obj"
        
        if [ ! -f "$obj_file" ]; then
            echo "ERROR: OBJ file not found: $obj_file"
            continue
        fi
        
        echo "Running Blender render for $i..."
        "$BLENDER_BIN" --background --python "$PROJECT_ROOT/blender_scripts/blender_render_script_stanford_orb.py" -- --obj_file "$obj_file" --env_dir "$env_maps_dir" --results_path "$root_dir$i"
        
        echo "Ground truth generation completed for $i"
    done
else
    echo "====== SKIPPED: Generating ground truth data (use --render-gts to enable) ======"
fi

echo "====== Starting model training pipeline ======"

    for i in "${list[@]}"; do
    echo "====== Processing dataset: $i ======"

    # Set input directory for images
    input_dir="$root_dir/$i/base/train/images_bg/"
    output_base_dir="$root_dir/$i/base/train/"

    # Run Diffusion Renderer to generate material properties
    if [ "$RUN_DIFFUSION_RENDERER" = true ]; then
        echo "====== ENABLED: Running Diffusion Renderer for $i ======"
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
        
        echo "Diffusion Renderer processing completed for $i"
        cd -
    else
        echo "====== SKIPPED: Diffusion Renderer processing (use --run-diffusion-renderer to enable) ======"
    fi

    # Now continue with the NeILF processing
    echo "Running 3DGS and NeILF processing for $i..."

    # First run the base 3DGS model
    if [ "$RUN_BASE_3DGS" = true ]; then

        echo "====== ENABLED: Running base 3DGS for $i ======"

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

        echo "Base 3DGS training completed for $i"
    else
        echo "====== SKIPPED: Base 3DGS training (use --run-base-3dgs to enable) ======"
    fi

    # Run base NeILF
    if [ "$RUN_BASE_NEILF" = true ]; then
        echo "====== ENABLED: Running base NeILF for $i ======"
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

        $PY eval_nvs.py --eval \
            -m $output_dir/${i}/neilf \
            -c $output_dir/${i}/neilf/chkpnt40000.pth \
            -t neilf

        # Run relighting for different environment maps
        echo "Running relighting for ${i} with neilf..."
        # Define environment maps as an array
        envmaps=("envmap3" "envmap6" "envmap12" "envmap24")

        # Iterate through each environment map
        for envmap in "${envmaps[@]}"; do
            echo "Processing ${i} with neilf and $envmap..."

            $PY relighting.py \
                -co "$root_dir/$i/$envmap/" \
                --video \
                --output "$output_dir/$i/neilf/$envmap" \
                --envmap_path "env_map/$envmap.exr" \
                --capture_list "pbr_env,normal,base_color,visibility,roughness,metallic,diffuse,specular" \
                --sample_num 64 \
                --ply_path "$output_dir/$i/neilf/point_cloud/iteration_40000/point_cloud.ply" \
                --gt_path "$root_dir/$i/$envmap/test/images/"

            echo "Finished processing ${i} with neilf and $envmap"
        done
        echo "Relighting for ${i} with neilf completed"
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

                $PY train.py --eval \
                    -s $root_dir/$i/base/ \
                    -m $output_dir/$i/$neilf_name \
                    -c $output_dir/$i/3dgs/chkpnt30000.pth \
                    --base_color_folder /train/${model_name,,}_baseColor/ \
                    --roughness_folder /train/${model_name,,}_roughness/ \
                    --metallic_folder /train/${model_name,,}_metallic/ \
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

                $PY eval_nvs.py --eval \
                    -m $output_dir/${i}/$neilf_name \
                    -c $output_dir/${i}/$neilf_name/chkpnt40000.pth \
                    -t neilf

                # Run relighting for different environment maps
                echo "Running relighting for ${i} with $neilf_name..."
                # Define environment maps as an array
                envmaps=("envmap3" "envmap6" "envmap12" "envmap24")

                # Iterate through each environment map
                for envmap in "${envmaps[@]}"; do
                    echo "Processing ${i} with $neilf_name and $envmap..."

                    $PY relighting.py \
                        -co "$root_dir/$i/$envmap/" \
                        --video \
                        --output "$output_dir/$i/$neilf_name/$envmap" \
                        --envmap_path "env_map/$envmap.exr" \
                        --capture_list "pbr_env,normal,base_color,visibility,roughness,metallic,diffuse,specular" \
                        --sample_num 64 \
                        --ply_path "$output_dir/$i/$neilf_name/point_cloud/iteration_40000/point_cloud.ply" \
                        --gt_path "$root_dir/$i/$envmap/test/images/"

                    echo "Finished processing ${i} with $neilf_name and $envmap"
                done
                echo "Relighting for ${i} with $neilf_name completed"
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

                $PY train.py --eval \
                    -s $root_dir/$i/base/ \
                    -m $output_dir/$i/$neilf_name \
                    -c $output_dir/$i/3dgs/chkpnt30000.pth \
                    --base_color_folder /train/${model_name,,}_baseColor/ \
                    --roughness_folder /train/${model_name,,}_roughness/ \
                    --metallic_folder /train/${model_name,,}_metallic/ \
                    --save_training_vis \
            --detect_anomaly \
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

                $PY eval_nvs.py --eval \
                    -m $output_dir/${i}/$neilf_name \
                    -c $output_dir/${i}/$neilf_name/chkpnt40000.pth \
                    -t neilf

                # Run relighting for different environment maps
                echo "Running relighting for ${i} with $neilf_name..."
                # Define environment maps as an array
                envmaps=("envmap3" "envmap6" "envmap12" "envmap24")

                # Iterate through each environment map
                for envmap in "${envmaps[@]}"; do
                    echo "Processing ${i} with $neilf_name and $envmap..."

                    $PY relighting.py \
                        -co "$root_dir/$i/$envmap/" \
                        --video \
                        --output "$output_dir/$i/$neilf_name/$envmap" \
                        --envmap_path "env_map/$envmap.exr" \
                        --capture_list "pbr_env,normal,base_color,visibility,roughness,metallic,diffuse,specular" \
                        --sample_num 64 \
                        --ply_path "$output_dir/$i/$neilf_name/point_cloud/iteration_40000/point_cloud.ply" \
                        --gt_path "$root_dir/$i/$envmap/test/images/"

                    echo "Finished processing ${i} with $neilf_name and $envmap"
                done
                echo "Relighting for ${i} with $neilf_name completed"
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

                $PY train.py --eval \
                    -s $root_dir/$i/base/ \
                    -m $output_dir/$i/$neilf_name \
                    -c $output_dir/$i/3dgs/chkpnt30000.pth \
                    --base_color_folder /train/${model_name,,}_baseColor/ \
                    --roughness_folder /train/${model_name,,}_roughness/ \
                    --metallic_folder /train/${model_name,,}_metallic/ \
                    --normals_folder /train/${model_name,,}_normals/ \
                    --save_training_vis \
                    --detect_anomaly \
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

                $PY eval_nvs.py --eval \
                    -m $output_dir/${i}/$neilf_name \
                    -c $output_dir/${i}/$neilf_name/chkpnt40000.pth \
                    -t neilf

                # Run relighting for different environment maps
                echo "Running relighting for ${i} with $neilf_name..."
                # Define environment maps as an array
                envmaps=("envmap3" "envmap6" "envmap12" "envmap24")

                # Iterate through each environment map
                for envmap in "${envmaps[@]}"; do
                    echo "Processing ${i} with $neilf_name and $envmap..."

                    $PY relighting.py \
                        -co "$root_dir/$i/$envmap/" \
                        --video \
                        --output "$output_dir/$i/$neilf_name/$envmap" \
                        --envmap_path "env_map/$envmap.exr" \
                        --capture_list "pbr_env,normal,base_color,visibility,roughness,metallic,diffuse,specular" \
                        --sample_num 64 \
                        --ply_path "$output_dir/$i/$neilf_name/point_cloud/iteration_40000/point_cloud.ply" \
                        --gt_path "$root_dir/$i/$envmap/test/images/" \
                        -t neilf_deferred

                    echo "Finished processing ${i} with $neilf_name and $envmap"
                done
                echo "Relighting for ${i} with $neilf_name completed"
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
                ["mlp_softmax_baseline_${model_name}"]="1.0 1.0 1.0"
            )

            # Run for each NeILF configuration
            for neilf_name in "${!neilf_configs[@]}"; do
                echo "Running $neilf_name for $i..."

                # Get the buffer values for this configuration
                read -r base_color_buffer roughness_buffer metallic_buffer <<< "${neilf_configs[$neilf_name]}"

                $PY train.py --eval \
                    -s $root_dir/$i/base/ \
                    -m $output_dir/$i/$neilf_name \
                    -c $output_dir/$i/3dgs/chkpnt30000.pth \
                    --base_color_folder /train/${model_name,,}_baseColor/ \
                    --roughness_folder /train/${model_name,,}_roughness/ \
                    --metallic_folder /train/${model_name,,}_metallic/ \
                    --save_training_vis \
                    --position_lr_init 0.000016 \
                    --position_lr_final 0.00000016 \
                    --normal_lr 0.0 \
                    --sh_lr 0.00025 \
                    --opacity_lr 0.005 \
                    --scaling_lr 0.0005 \
                    --rotation_lr 0.0001 \
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
                    --use_mlp

                # Run relighting for different environment maps
                echo "Running relighting for ${i} with $neilf_name..."
                # Define environment maps as an array
                envmaps=("envmap3" "envmap6" "envmap12" "envmap24")

                # Iterate through each environment map
                for envmap in "${envmaps[@]}"; do
                    echo "Processing ${i} with $neilf_name and $envmap..."

                    $PY relighting.py \
                        -co "$root_dir/$i/$envmap/" \
                        --video \
                        --output "$output_dir/$i/$neilf_name/$envmap" \
                        --envmap_path "env_map/$envmap.exr" \
                        --capture_list "pbr_env,normal,base_color,visibility,roughness,metallic,diffuse,specular" \
                        --sample_num 800 \
                        --ply_path "$output_dir/$i/$neilf_name/point_cloud/iteration_40000/point_cloud.ply" \
                        --gt_path "$root_dir/$i/$envmap/test/images/" \
                        -t neilf_deferred_importance

                    echo "Finished processing ${i} with $neilf_name and $envmap"
                done
                echo "Relighting for ${i} with $neilf_name completed"
            done
        done
    else
        echo "====== SKIPPED: Processing MLP NeILF (use --run-mlp to enable) ======"
    fi

    echo "====== Completed all processing for dataset: $i ======"
done

echo "All processing completed."
