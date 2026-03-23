import array
import json
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from random import randint

import Imath
import OpenEXR
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render_fn_dict
from scene import Scene, GaussianModel
from scene.direct_light_map import DirectLightMap
from scene.mlp import BaseColorMLP, CombinedMLP, MetallicMLP, NormalMLP, RoughnessMLP
from torchvision.utils import make_grid, save_image
from utils.general_utils import safe_state
from utils.graphics_utils import load_and_transform_diffusion_renderer_image, rgb_to_srgb
from utils.loss_utils import l1_loss, scale_invariant_l1_loss
from utils.system_utils import prepare_output_and_logger
from utils.training.gradient_logging import log_mlp_gradients
from utils.training.intersection_tracing import (
    load_intersection_data,
    perform_intersection_tracing_for_all_training_images,
)
from utils.training.normal_utils import get_camera_to_world_rotation_matrix, load_normal_image
from utils.training.render_debug import create_timelapse_video, render_timelapse_frame
from utils.training.reporting import eval_render, save_training_vis, training_report


def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, is_pbr=False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    """
    Setup Gaussians
    """
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    scene = Scene(dataset, gaussians)
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)
        
        # Load MLP data from checkpoint if MLPs are enabled
        if is_pbr and args.use_mlp:
            checkpoint_dir = os.path.dirname(args.checkpoint)
            mlp_data_dir = os.path.join(checkpoint_dir, "mlp_data")
            if os.path.exists(mlp_data_dir):
                try:
                    # Load MLP metadata
                    import json
                    with open(os.path.join(mlp_data_dir, "mlp_metadata.json"), 'r') as f:
                        mlp_metadata = json.load(f)
                    
                    if mlp_metadata.get('has_mlps', False):
                        # Load projected data
                        projected_base_colors = np.load(os.path.join(mlp_data_dir, "projected_base_colors.npy"))
                        projected_roughness = np.load(os.path.join(mlp_data_dir, "projected_roughness.npy"))
                        projected_metallic = np.load(os.path.join(mlp_data_dir, "projected_metallic.npy"))
                        
                        # Convert to tensors and store (not trainable)
                        gaussians.projected_base_colors = torch.tensor(projected_base_colors, dtype=torch.float, device="cuda", requires_grad=False)
                        gaussians.projected_roughness = torch.tensor(projected_roughness, dtype=torch.float, device="cuda", requires_grad=False)
                        gaussians.projected_metallic = torch.tensor(projected_metallic, dtype=torch.float, device="cuda", requires_grad=False)
                        
                        # Load normal MLP data if it exists and normals_folder is provided
                        if mlp_metadata.get('has_normal_mlp', False) and getattr(args, 'normals_folder', None) is not None:
                            try:
                                projected_normals = np.load(os.path.join(mlp_data_dir, "projected_normals.npy"))
                                gaussians.projected_normals = torch.tensor(projected_normals, dtype=torch.float, device="cuda", requires_grad=False)
                                print(f"  - Normal MLP data loaded, shape: {gaussians.projected_normals.shape}")
                            except Exception as e:
                                print(f"Warning: Failed to load normal MLP data: {e}")
                        else:
                            print("  - Skipping normal MLP data load (no normals_folder provided)")
                        
                        # Initialize MLPs
                        num_train_images = mlp_metadata['num_train_images']
                        net_width = mlp_metadata.get('net_width', 64)
                        
                        # Check if using combined MLP - prefer args flag, fallback to metadata
                        use_combined_mlp = getattr(args, 'combined_mlp', False) or mlp_metadata.get('use_combined_mlp', False)
                        gaussians.use_combined_mlp = use_combined_mlp
                        
                        if use_combined_mlp:
                            gaussians.combined_mlp = CombinedMLP(
                                num_train_images=num_train_images, net_width=net_width,
                                enable_gradient_logging=args.enable_gradient_logging
                            ).cuda()
                            gaussians.base_color_mlp = None
                            gaussians.roughness_mlp = None
                            gaussians.metallic_mlp = None
                        else:
                            gaussians.base_color_mlp = BaseColorMLP(
                                num_train_images=num_train_images, net_width=net_width,
                                enable_gradient_logging=args.enable_gradient_logging
                            ).cuda()
                            gaussians.roughness_mlp = RoughnessMLP(
                                num_train_images=num_train_images, net_width=net_width,
                                enable_gradient_logging=args.enable_gradient_logging
                            ).cuda()
                            gaussians.metallic_mlp = MetallicMLP(
                                num_train_images=num_train_images, net_width=net_width,
                                enable_gradient_logging=args.enable_gradient_logging
                            ).cuda()
                        
                        # Initialize normal MLP if data exists and normals_folder provided
                        if mlp_metadata.get('has_normal_mlp', False) and getattr(args, 'normals_folder', None) is not None:
                            gaussians.normal_mlp = NormalMLP(
                                num_train_images=num_train_images, net_width=net_width,
                                enable_gradient_logging=args.enable_gradient_logging
                            ).cuda()
                        
                        # Load MLP weights if they exist
                        try:
                            if use_combined_mlp:
                                gaussians.combined_mlp.load_state_dict(torch.load(os.path.join(mlp_data_dir, "combined_mlp_weights.pth")), strict=False)
                            else:
                                gaussians.base_color_mlp.load_state_dict(torch.load(os.path.join(mlp_data_dir, "base_color_mlp_weights.pth")), strict=False)
                                gaussians.roughness_mlp.load_state_dict(torch.load(os.path.join(mlp_data_dir, "roughness_mlp_weights.pth")), strict=False)
                                gaussians.metallic_mlp.load_state_dict(torch.load(os.path.join(mlp_data_dir, "metallic_mlp_weights.pth")), strict=False)
                            if mlp_metadata.get('has_normal_mlp', False) and getattr(args, 'normals_folder', None) is not None:
                                gaussians.normal_mlp.load_state_dict(torch.load(os.path.join(mlp_data_dir, "normal_mlp_weights.pth")), strict=False)
                            print("✓ MLP weights loaded successfully")
                        except Exception as e:
                            print(f"Warning: Failed to load MLP weights: {e}")
                            print("MLPs will be initialized with random weights")
                        
                        print(f"✓ MLP projected data and weights loaded from checkpoint {mlp_data_dir}")
                        print(f"  - Number of training images: {num_train_images}")
                        print(f"  - Network width: {net_width}")
                        print(f"  - Using combined MLP: {use_combined_mlp}")
                        print(f"  - Projected data shapes: {gaussians.projected_base_colors.shape}, {gaussians.projected_roughness.shape}, {gaussians.projected_metallic.shape}")
                        if mlp_metadata.get('has_normal_mlp', False):
                            print(f"  - Normal MLP enabled")
                        
                except Exception as e:
                    print(f"Warning: Failed to load MLP data from checkpoint {mlp_data_dir}: {e}")
                    print("Model will be loaded without MLPs")

    elif scene.loaded_iter:
        gaussians.load_ply(os.path.join(dataset.model_path,
                                        "point_cloud",
                                        "iteration_" + str(scene.loaded_iter),
                                        "point_cloud.ply"))
        
        # Always attach MLPs and projected data if is_pbr and args.use_mlp
        if is_pbr and args.use_mlp:
            ply_dir = os.path.dirname(os.path.join(dataset.model_path,
                                                  "point_cloud",
                                                  "iteration_" + str(scene.loaded_iter),
                                                  "point_cloud.ply"))
            mlp_data_dir = os.path.join(ply_dir, "mlp_data")
            if os.path.exists(mlp_data_dir):
                try:
                    import json
                    with open(os.path.join(mlp_data_dir, "mlp_metadata.json"), 'r') as f:
                        mlp_metadata = json.load(f)
                    if mlp_metadata.get('has_mlps', False):
                        projected_base_colors = np.load(os.path.join(mlp_data_dir, "projected_base_colors.npy"))
                        projected_roughness = np.load(os.path.join(mlp_data_dir, "projected_roughness.npy"))
                        projected_metallic = np.load(os.path.join(mlp_data_dir, "projected_metallic.npy"))
                        gaussians.projected_base_colors = torch.tensor(projected_base_colors, dtype=torch.float, device="cuda", requires_grad=False)
                        gaussians.projected_roughness = torch.tensor(projected_roughness, dtype=torch.float, device="cuda", requires_grad=False)
                        gaussians.projected_metallic = torch.tensor(projected_metallic, dtype=torch.float, device="cuda", requires_grad=False)
                        
                        # Load normal MLP data if it exists and normals_folder is provided
                        if mlp_metadata.get('has_normal_mlp', False) and getattr(args, 'normals_folder', None) is not None:
                            try:
                                projected_normals = np.load(os.path.join(mlp_data_dir, "projected_normals.npy"))
                                gaussians.projected_normals = torch.tensor(projected_normals, dtype=torch.float, device="cuda", requires_grad=False)
                                print(f"  - Normal MLP data loaded, shape: {gaussians.projected_normals.shape}")
                            except Exception as e:
                                print(f"Warning: Failed to load normal MLP data: {e}")
                        else:
                            print("  - Skipping normal MLP data load (no normals_folder provided)")
                        
                        num_train_images = mlp_metadata['num_train_images']
                        net_width = mlp_metadata.get('net_width', 64)
                        
                        # Check if using combined MLP - prefer args flag, fallback to metadata
                        use_combined_mlp = getattr(args, 'combined_mlp', False) or mlp_metadata.get('use_combined_mlp', False)
                        gaussians.use_combined_mlp = use_combined_mlp
                        
                        if use_combined_mlp:
                            gaussians.combined_mlp = CombinedMLP(num_train_images=num_train_images, net_width=net_width, enable_gradient_logging=args.enable_gradient_logging).cuda()
                            gaussians.base_color_mlp = None
                            gaussians.roughness_mlp = None
                            gaussians.metallic_mlp = None
                        else:
                            gaussians.base_color_mlp = BaseColorMLP(num_train_images=num_train_images, net_width=net_width, enable_gradient_logging=args.enable_gradient_logging).cuda()
                            gaussians.roughness_mlp = RoughnessMLP(num_train_images=num_train_images, net_width=net_width, enable_gradient_logging=args.enable_gradient_logging).cuda()
                            gaussians.metallic_mlp = MetallicMLP(num_train_images=num_train_images, net_width=net_width, enable_gradient_logging=args.enable_gradient_logging).cuda()
                        
                        # Initialize normal MLP if data exists and normals_folder provided
                        if mlp_metadata.get('has_normal_mlp', False) and getattr(args, 'normals_folder', None) is not None:
                            gaussians.normal_mlp = NormalMLP(num_train_images=num_train_images, net_width=net_width, enable_gradient_logging=args.enable_gradient_logging).cuda()
                        
                        try:
                            if use_combined_mlp:
                                gaussians.combined_mlp.load_state_dict(torch.load(os.path.join(mlp_data_dir, "combined_mlp_weights.pth")), strict=False)
                            else:
                                gaussians.base_color_mlp.load_state_dict(torch.load(os.path.join(mlp_data_dir, "base_color_mlp_weights.pth")), strict=False)
                                gaussians.roughness_mlp.load_state_dict(torch.load(os.path.join(mlp_data_dir, "roughness_mlp_weights.pth")), strict=False)
                                gaussians.metallic_mlp.load_state_dict(torch.load(os.path.join(mlp_data_dir, "metallic_mlp_weights.pth")), strict=False)
                            if mlp_metadata.get('has_normal_mlp', False) and getattr(args, 'normals_folder', None) is not None:
                                gaussians.normal_mlp.load_state_dict(torch.load(os.path.join(mlp_data_dir, "normal_mlp_weights.pth")), strict=False)
                            print("✓ MLP weights loaded successfully (post-ply)")
                        except Exception as e:
                            print(f"Warning: Failed to load MLP weights (post-ply): {e}")
                            print("MLPs will be initialized with random weights (post-ply)")
                        print(f"✓ MLP projected data and weights loaded from {mlp_data_dir} (post-ply)")
                        print(f"  - Number of training images: {num_train_images}")
                        print(f"  - Network width: {net_width}")
                        print(f"  - Using combined MLP: {use_combined_mlp}")
                        print(f"  - Projected data shapes: {gaussians.projected_base_colors.shape}, {gaussians.projected_roughness.shape}, {gaussians.projected_metallic.shape}")
                        if mlp_metadata.get('has_normal_mlp', False):
                            print(f"  - Normal MLP enabled")
                        gaussians.quick_debug_mlp_status()
                except Exception as e:
                    print(f"Warning: Failed to load MLP data from {mlp_data_dir} (post-ply): {e}")
                    print("Model will be loaded without MLPs (post-ply)")

    else:
        # Prefer scene-provided point cloud, but fall back to dataset points3d.ply if needed
        ply_path = os.path.join(dataset.source_path, "points3d.ply")
        if scene.scene_info is not None and scene.scene_info.point_cloud is not None:
            gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)
        elif os.path.isfile(ply_path):
            print(f"Loading point cloud from {ply_path}")
            gaussians.load_ply(ply_path)
        else:
            raise FileNotFoundError(f"No valid point cloud found; expected {ply_path}")

    """
    Setup MLPs for base_color, roughness, and metallic (BEFORE training setup)
    """
    if is_pbr and args.use_mlp:
        print("Initializing MLPs for base_color, roughness, and metallic...")
        
        # Get number of training images
        num_train_images = len(scene.getTrainCameras())
        print(f"Number of training images: {num_train_images}")
        
        # Check if using combined MLP or separate MLPs
        use_combined_mlp = getattr(args, 'combined_mlp', False)
        gaussians.use_combined_mlp = use_combined_mlp
        
        if use_combined_mlp:
            print("Using combined MLP for base_color, roughness, and metallic")
            combined_mlp = CombinedMLP(num_train_images=num_train_images, net_width=64, enable_gradient_logging=args.enable_gradient_logging).cuda()
            
            # Load combined MLP weights if they exist
            if hasattr(gaussians, 'mlp_weights_path'):
                try:
                    combined_mlp.load_state_dict(torch.load(os.path.join(gaussians.mlp_weights_path, "combined_mlp_weights.pth")), strict=False)
                    print("✓ Combined MLP weights loaded successfully")
                except Exception as e:
                    print(f"Warning: Failed to load combined MLP weights: {e}")
                    print("Combined MLP will be initialized with random weights")
            
            gaussians.combined_mlp = combined_mlp
            # Set separate MLPs to None to indicate they're not used
            gaussians.base_color_mlp = None
            gaussians.roughness_mlp = None
            gaussians.metallic_mlp = None
        else:
            print("Using separate MLPs for base_color, roughness, and metallic")
            # Initialize MLPs with the correct number of training images and enable gradient logging
            base_color_mlp = BaseColorMLP(num_train_images=num_train_images, net_width=64, enable_gradient_logging=args.enable_gradient_logging).cuda()
            roughness_mlp = RoughnessMLP(num_train_images=num_train_images, net_width=64, enable_gradient_logging=args.enable_gradient_logging).cuda()
            metallic_mlp = MetallicMLP(num_train_images=num_train_images, net_width=64, enable_gradient_logging=args.enable_gradient_logging).cuda()
            
            # Load MLP weights if they exist
            if hasattr(gaussians, 'mlp_weights_path'):
                try:
                    base_color_mlp.load_state_dict(torch.load(os.path.join(gaussians.mlp_weights_path, "base_color_mlp_weights.pth")), strict=False)
                    roughness_mlp.load_state_dict(torch.load(os.path.join(gaussians.mlp_weights_path, "roughness_mlp_weights.pth")), strict=False)
                    metallic_mlp.load_state_dict(torch.load(os.path.join(gaussians.mlp_weights_path, "metallic_mlp_weights.pth")), strict=False)
                    print("✓ MLP weights loaded successfully")
                except Exception as e:
                    print(f"Warning: Failed to load MLP weights: {e}")
                    print("MLPs will be initialized with random weights")
            
            gaussians.base_color_mlp = base_color_mlp
            gaussians.roughness_mlp = roughness_mlp
            gaussians.metallic_mlp = metallic_mlp
        
        # Normal MLP is always separate (not part of combined MLP)
        normal_mlp = None
        if getattr(args, 'normals_folder', None) is not None:
            normal_mlp = NormalMLP(num_train_images=num_train_images, net_width=64, enable_gradient_logging=args.enable_gradient_logging).cuda()
            if hasattr(gaussians, 'mlp_weights_path'):
                try:
                    normal_mlp.load_state_dict(torch.load(os.path.join(gaussians.mlp_weights_path, "normal_mlp_weights.pth")), strict=False)
                except Exception as e:
                    print(f"Warning: Failed to load normal MLP weights: {e}")
        
        if normal_mlp is not None:
            gaussians.normal_mlp = normal_mlp
        
        # Create projected input data for MLPs (num_train_images values per gaussian)
        num_gaussians = gaussians.get_xyz.shape[0]
        
        # Initialize with reasonable values instead of zeros
        # Base colors: initialize with white (1,1,1) for each training image
        projected_base_colors = torch.ones((num_gaussians, num_train_images, 3), device='cuda', requires_grad=False)  # [num_gaussians, num_train_images, 3]
        
        # Roughness: initialize with medium roughness (0.5) for each training image
        projected_roughness = torch.full((num_gaussians, num_train_images, 1), 0.5, device='cuda', requires_grad=False)  # [num_gaussians, num_train_images, 1]
        
        # Metallic: initialize with non-metallic (0.0) for each training image
        projected_metallic = torch.zeros((num_gaussians, num_train_images, 1), device='cuda', requires_grad=False)  # [num_gaussians, num_train_images, 1]
        
        # Normals: initialize with up direction (0,1,0) for each training image
        projected_normals = torch.tensor([0.0, 1.0, 0.0], device='cuda', requires_grad=False).expand(num_gaussians, num_train_images, 3)  # [num_gaussians, num_train_images, 3]
        
        # Store projected data in gaussians for access during training
        gaussians.projected_base_colors = projected_base_colors
        gaussians.projected_roughness = projected_roughness
        gaussians.projected_metallic = projected_metallic
        if normal_mlp is not None:
            gaussians.projected_normals = projected_normals
        
        # Store relevant flags for model metadata
        gaussians.perform_intersection_tracing = args.perform_intersection_tracing
        gaussians.load_intersection_data = args.load_intersection_data
        
        print(f"✓ MLPs initialized for {num_gaussians} Gaussians")
        if use_combined_mlp:
            print(f"  - Using combined MLP")
        else:
            print(f"  - Using separate MLPs")
        
        # Debug MLP outputs to check initialization
        gaussians.debug_mlp_outputs()
    elif is_pbr:
        print("MLPs disabled - using direct parameter optimization")
        
        # Store relevant flags for model metadata
        gaussians.perform_intersection_tracing = args.perform_intersection_tracing
        gaussians.load_intersection_data = args.load_intersection_data

    gaussians.training_setup(opt)

    """
    Setup PBR components
    """
    pbr_kwargs = dict()
    if is_pbr:
        
        # first update visibility
        gaussians.update_visibility(pipe.sample_num)
        
        pbr_kwargs['sample_num'] = pipe.sample_num
        print("Using global incident light for regularization.")
        direct_env_light = DirectLightMap(dataset.env_resolution, opt.light_init)
        
        if args.checkpoint:
            env_checkpoint = os.path.dirname(args.checkpoint) + "/env_light_" + os.path.basename(args.checkpoint)
            print("Trying to load global incident light from ", env_checkpoint)
            if os.path.exists(env_checkpoint):
                direct_env_light.create_from_ckpt(env_checkpoint, restore_optimizer=True)
                print("Successfully loaded!")
            else:
                print("Failed to load!")

            direct_env_light.training_setup(opt)
            pbr_kwargs["env_light"] = direct_env_light
            
        # Add MLPs to PBR kwargs for access during training (only if MLPs exist)
        if is_pbr and args.use_mlp and hasattr(gaussians, 'base_color_mlp'):
            pbr_kwargs["base_color_mlp"] = gaussians.base_color_mlp
            pbr_kwargs["roughness_mlp"] = gaussians.roughness_mlp
            pbr_kwargs["metallic_mlp"] = gaussians.metallic_mlp
            pbr_kwargs["projected_base_colors"] = gaussians.projected_base_colors
            pbr_kwargs["projected_roughness"] = gaussians.projected_roughness
            pbr_kwargs["projected_metallic"] = gaussians.projected_metallic

    """ Prepare render function and bg"""
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Optional: dump all per-view normal GTs at the start using the exact training transformation
    if getattr(args, 'dump_normals_at_start', False) and getattr(args, 'normal_gt_folder', None) is not None:
        try:
            dump_dir = args.dump_normals_dir or os.path.join(args.model_path, "normal_gt_dump")
            os.makedirs(dump_dir, exist_ok=True)
            training_cameras = scene.getTrainCameras()
            print(f"\n=== Dumping {len(training_cameras)} normal GT images to: {dump_dir} ===")
            for camera in training_cameras:
                try:
                    # Resolve per-camera normal path (same logic as main loop)
                    original_basename = camera.image_name
                    number_part = os.path.splitext(original_basename)[0]
                    number_parts = [number_part]
                    if number_part.startswith("r_"):
                        number_parts.append("r_r_" + number_part[2:])
                    elif number_part.startswith("r_r_"):
                        number_parts.append("r_" + number_part[4:])

                    normal_filename_options = []
                    for num_part in number_parts:
                        normal_filename_options.extend([
                            f"{num_part}_normalcamera0001.exr",
                            f"{num_part}_normal0001.exr",
                            f"{num_part}_normal.exr",
                            f"{num_part}_normal.png"
                        ])

                    normal_gt_path = None
                    normal_folder = args.normal_gt_folder
                    if normal_folder.startswith('/'):
                        normal_folder = normal_folder[1:]
                    for fname in normal_filename_options:
                        p = os.path.join(dataset.source_path, normal_folder, fname)
                        if os.path.exists(p):
                            normal_gt_path = p
                            break

                    if normal_gt_path is None:
                        print(f"[DumpNormals] Missing normal GT for {camera.image_name}")
                        continue

                    device = background.device
                    exr_alpha_mask = None
                    # Load and transform normals to world space if needed
                    if getattr(args, 'normal_gt_is_camera_space', False):
                        normal_gt = load_normal_image(normal_gt_path).to(device)  # [3,H,W] in [-1,1]
                        Hn, Wn = normal_gt.shape[1], normal_gt.shape[2]
                        # EXR alpha read if available
                        if normal_gt_path.endswith('.exr'):
                            try:
                                file = OpenEXR.InputFile(normal_gt_path)
                                dw = file.header()['dataWindow']
                                size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                                if 'A' in file.header()['channels']:
                                    a = np.array(array.array('f', file.channel('A', FLOAT))).reshape(size[1], size[0])
                                    exr_alpha_mask = torch.from_numpy(a).float().view(1, size[1], size[0]).to(device)
                                file.close()
                            except Exception:
                                exr_alpha_mask = None
                        camera_rotation_matrix = get_camera_to_world_rotation_matrix(camera)
                        camera_rotation_tensor = torch.from_numpy(camera_rotation_matrix).float().to(device)
                        normal_flat = normal_gt.permute(1, 2, 0).reshape(-1, 3)
                        world_flat = torch.matmul(normal_flat, camera_rotation_tensor)
                        world_flat = F.normalize(world_flat, dim=-1, eps=1e-6)
                        normal_world = world_flat.reshape(Hn, Wn, 3).permute(2, 0, 1)
                    else:
                        if normal_gt_path.endswith('.exr'):
                            file = OpenEXR.InputFile(normal_gt_path)
                            dw = file.header()['dataWindow']
                            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                            r = np.array(array.array('f', file.channel('R', FLOAT))).reshape(size[1], size[0])
                            g = np.array(array.array('f', file.channel('G', FLOAT))).reshape(size[1], size[0])
                            b = np.array(array.array('f', file.channel('B', FLOAT))).reshape(size[1], size[0])
                            try:
                                if 'A' in file.header()['channels']:
                                    a = np.array(array.array('f', file.channel('A', FLOAT))).reshape(size[1], size[0])
                                    exr_alpha_mask = torch.from_numpy(a).float().view(1, size[1], size[0]).to(device)
                            except Exception:
                                exr_alpha_mask = None
                            file.close()
                            normal_world = torch.from_numpy(np.stack([r, g, b], axis=0)).float().to(device)
                        else:
                            normal_world = torchvision.io.read_image(normal_gt_path).float().to(device)
                            if normal_world.shape[0] >= 4:
                                exr_alpha_mask = normal_world[3:4].clone()
                                normal_world = normal_world[:3]
                            if normal_world.max() > 1.0:
                                normal_world = normal_world / 255.0
                        if getattr(args, 'normal_gt_range', '-1_1') == '0_1':
                            normal_world = normal_world * 2.0 - 1.0
                        normal_world = F.normalize(normal_world, dim=0, eps=1e-6)

                    # Map to [0,1]
                    target_rgb = torch.clamp(normal_world * 0.5 + 0.5, 0.0, 1.0)

                    # Load mask directly from the original image file to ensure we get the correct alpha channel
                    # This is more reliable than relying on camera.image_mask which might not be set correctly
                    img_mask = None
                    
                    # Determine the correct path to load mask from
                    # If images.__backup__ exists, the original images with masks are there
                    # (this happens when the script has already swapped images folder with normals)
                    if hasattr(camera, 'image_path') and camera.image_path is not None:
                        image_path = camera.image_path
                        
                        # Check if there's a backup folder with original images
                        # The backup folder is typically images.__backup__ in the same directory
                        image_dir = os.path.dirname(image_path)
                        image_filename = os.path.basename(image_path)
                        backup_dir = image_dir + ".__backup__"
                        backup_path = os.path.join(backup_dir, image_filename)
                        
                        # Prefer backup path if it exists (original images with masks)
                        if os.path.exists(backup_path):
                            mask_source_path = backup_path
                            print(f"[DumpNormals] Loading mask from backup: {backup_path}")
                        elif os.path.exists(image_path):
                            mask_source_path = image_path
                        else:
                            mask_source_path = None
                        
                        if mask_source_path is not None:
                            try:
                                # Use torchvision to read image (returns [C, H, W] tensor with values 0-255)
                                orig_img = torchvision.io.read_image(mask_source_path)
                                if orig_img.shape[0] == 4:
                                    # Extract alpha channel as mask
                                    alpha = orig_img[3:4, :, :].float() / 255.0
                                    img_mask = alpha.to(device)
                                else:
                                    print(f"[DumpNormals] Image at {mask_source_path} has {orig_img.shape[0]} channels, no alpha")
                            except Exception as e:
                                print(f"[DumpNormals] Could not load mask from {mask_source_path}: {e}")
                    
                    # Second try: use camera.image_mask if available
                    if img_mask is None and hasattr(camera, 'image_mask') and camera.image_mask is not None:
                        img_mask = camera.image_mask.to(device)
                        # Ensure mask has correct shape [1, H, W]
                        if img_mask.dim() == 2:
                            img_mask = img_mask.unsqueeze(0)
                        elif img_mask.dim() == 3 and img_mask.shape[0] != 1:
                            img_mask = img_mask[:1, ...]
                    
                    # Fallback: all ones mask
                    if img_mask is None:
                        img_mask = torch.ones((1, target_rgb.shape[1], target_rgb.shape[2]), device=device, dtype=target_rgb.dtype)
                        print(f"[DumpNormals] Warning: No mask found for {camera.image_name}, using all-ones mask")
                    
                    # Resize mask to match target_rgb if needed
                    if img_mask.shape[1:] != target_rgb.shape[1:]:
                        img_mask = F.interpolate(img_mask[None], size=target_rgb.shape[1:], mode='nearest')[0]
                    
                    # Ensure mask is in [0, 1] range
                    img_mask = img_mask.clamp(0.0, 1.0)
                    
                    # Apply mask to target_rgb (multiply RGB channels with mask)
                    mask_3c = img_mask.repeat(3, 1, 1) if img_mask.shape[0] == 1 else img_mask
                    target_rgb_masked = target_rgb * mask_3c

                    # Save only RGBA (RGB masked, A = mask)
                    rgba = torch.cat([target_rgb_masked.clamp(0,1), img_mask.clamp(0,1)], dim=0)  # 4,H,W
                    save_image(rgba, os.path.join(dump_dir, f"{camera.image_name}.png"))
                except Exception as e:
                    print(f"[DumpNormals] Error for {camera.image_name}: {e}")
            print(f"=== Normal GT dump complete ===\n")
            if getattr(args, 'dump_and_exit', False):
                print("Exiting after normal GT dump as requested (--dump_and_exit).")
                return
        except Exception as e:
            print(f"[DumpNormals] Failed to dump normals at start: {e}")

    """
    Perform intersection tracing for all training images
    """
    if args.perform_intersection_tracing:
        print("\n=== Performing Intersection Tracing ===")
        try:
            gaussians.intersection_tracing = True
            # Get image dimensions from first camera
            first_camera = scene.getTrainCameras()[0]
            width = first_camera.image_width
            height = first_camera.image_height
            
            # Perform intersection tracing
            all_gaussian_base_colors, all_gaussian_roughness, all_gaussian_metallic, all_gaussian_normals, gaussian_hit_mask = perform_intersection_tracing_for_all_training_images(
                gaussians, scene, dataset.model_path, width, height, args, dataset)
            
            print("Shape of the gaussian base colors tensor: ", all_gaussian_base_colors.shape)
            print("Shape of the gaussian roughness tensor: ", all_gaussian_roughness.shape)
            print("Shape of the gaussian metallic tensor: ", all_gaussian_metallic.shape)

            # Compute mean across all camera views, but only for Gaussians that were actually hit
            # For each Gaussian, only average over cameras where it was hit
            
            # Compute mean across all camera views, but only for Gaussians that were actually hit
            print("\n=== Computing Mean Values ===")
            
            # Count how many cameras each Gaussian was hit by
            hit_counts_per_gaussian = gaussian_hit_mask.sum(dim=1)  # [num_gaussians]
            
            # Get number of Gaussians from tensor shape
            num_gaussians = all_gaussian_base_colors.shape[0]
            
            # Compute mean only over valid hits (avoid division by zero)
            mean_base_colors = torch.zeros((num_gaussians, 3), device='cuda', dtype=torch.float32)
            mean_roughness = torch.zeros((num_gaussians, 1), device='cuda', dtype=torch.float32)
            mean_metallic = torch.zeros((num_gaussians, 1), device='cuda', dtype=torch.float32)
            mean_normals = torch.zeros((num_gaussians, 3), device='cuda', dtype=torch.float32)
            
            # Only compute mean for Gaussians that were hit at least once
            valid_gaussians = hit_counts_per_gaussian > 0
            if valid_gaussians.any():
                mean_base_colors[valid_gaussians] = (
                    all_gaussian_base_colors[valid_gaussians].sum(dim=1) / 
                    hit_counts_per_gaussian[valid_gaussians].unsqueeze(-1).float()
                )
                mean_roughness[valid_gaussians] = (
                    all_gaussian_roughness[valid_gaussians].sum(dim=1) / 
                    hit_counts_per_gaussian[valid_gaussians].unsqueeze(-1).float()
                )
                mean_metallic[valid_gaussians] = (
                    all_gaussian_metallic[valid_gaussians].sum(dim=1) / 
                    hit_counts_per_gaussian[valid_gaussians].unsqueeze(-1).float()
                )
                mean_normals[valid_gaussians] = (
                    all_gaussian_normals[valid_gaussians].sum(dim=1) / 
                    hit_counts_per_gaussian[valid_gaussians].unsqueeze(-1).float()
                )
            
            print(f"Mean values computed for {valid_gaussians.sum().item()}/{num_gaussians} Gaussians")
            print(f"Average hits per Gaussian: {hit_counts_per_gaussian.float().mean().item():.2f}")
            
            print(f"Mean base colors shape: {mean_base_colors.shape}")
            print(f"Mean roughness shape: {mean_roughness.shape}")
            print(f"Mean metallic shape: {mean_metallic.shape}")
            print(f"Mean normals shape: {mean_normals.shape}")
            
            # Delete non-hit Gaussians completely
            invalid_gaussians = hit_counts_per_gaussian == 0
            if invalid_gaussians.any():
                num_invalid = invalid_gaussians.sum().item()
                valid_gaussians = ~invalid_gaussians
                
                print(f"Deleting {num_invalid} non-hit Gaussians completely (keeping {valid_gaussians.sum().item()})")
                
                # Filter mean colors/roughness/metallic to only include valid gaussians
                mean_base_colors = mean_base_colors[valid_gaussians]
                mean_roughness = mean_roughness[valid_gaussians]
                mean_metallic = mean_metallic[valid_gaussians]
                mean_normals = mean_normals[valid_gaussians]
                
                # Delete invalid gaussians from the model
                gaussians.prune_points(invalid_gaussians)
                
                # Update projected data to match the new number of gaussians (only if MLPs are being used)
                if hasattr(gaussians, 'base_color_mlp') and hasattr(gaussians, 'projected_base_colors'):
                    gaussians.projected_base_colors = gaussians.projected_base_colors[valid_gaussians]
                    gaussians.projected_roughness = gaussians.projected_roughness[valid_gaussians]
                    gaussians.projected_metallic = gaussians.projected_metallic[valid_gaussians]
                    if hasattr(gaussians, 'projected_normals'):
                        gaussians.projected_normals = gaussians.projected_normals[valid_gaussians]
                    print("✓ Projected data updated after pruning")
                
                # Recompute visibility after pruning since cached incident directions and areas have wrong dimensions
                print("Recomputing visibility after pruning...")
                gaussians.update_visibility(pipe.sample_num)
                print("✓ Visibility recomputed with correct dimensions")
            
            # Apply intersection-traced values to the remaining gaussians
            with torch.no_grad():
                if hasattr(gaussians, 'base_color_mlp'):
                    # Update projected data with intersection-traced values
                    # Vectorized update of projected data with intersection-traced values
                    need_filter = all_gaussian_base_colors.shape[0] != mean_base_colors.shape[0]
                    if need_filter:
                        agb = all_gaussian_base_colors[valid_gaussians]
                        agr = all_gaussian_roughness[valid_gaussians]
                        agm = all_gaussian_metallic[valid_gaussians]
                        agn = all_gaussian_normals[valid_gaussians]
                        ghm = gaussian_hit_mask[valid_gaussians]
                    else:
                        agb = all_gaussian_base_colors
                        agr = all_gaussian_roughness
                        agm = all_gaussian_metallic
                        agn = all_gaussian_normals
                        ghm = gaussian_hit_mask

                    inv_mask = ~ghm  # [G, C]
                    gaussians.projected_base_colors = torch.where(inv_mask[..., None], mean_base_colors[:, None, :], agb)
                    gaussians.projected_roughness = torch.where(inv_mask[..., None], mean_roughness[:, None, :], agr)
                    gaussians.projected_metallic = torch.where(inv_mask[..., None], mean_metallic[:, None, :], agm)
                    if args is not None and getattr(args, 'normals_folder', None) is not None and hasattr(gaussians, 'projected_normals'):
                        filled_normals = torch.where(inv_mask[..., None], mean_normals[:, None, :], agn)
                        gaussians.projected_normals = filled_normals
                    
                    print("✓ Projected data updated with intersection-traced values (normals kept if no normals_folder)")
                else:
                    # Store average values directly in gaussian parameters (no projected data needed)
                    gaussians._base_color.data = mean_base_colors
                    gaussians._roughness.data = mean_roughness
                    gaussians._metallic.data = mean_metallic
                    # Ensure normals are normalized before storing
                    gaussians._normal.data = F.normalize(mean_normals, dim=-1, eps=1e-6)
                    
                    # Set flag to indicate intersection tracing has been used (colors are in linear space)
                    gaussians.intersection_tracing = True
                    
                    print("✓ Average intersection-traced values stored directly in gaussian parameters (normals normalized)")
                
                print("✓ Base colors, roughness, metallic and normal values assigned from intersection tracing")
                print(f"✓ Final model has {gaussians.get_xyz.shape[0]} gaussians after pruning")
            print("✓ Intersection tracing completed successfully")
            
        except Exception as e:
            print(f"✗ Error during intersection tracing: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with training without intersection data...")
    
    elif args.load_intersection_data:
        print("\n=== Loading Existing Intersection Data ===")
        try:
            # Load existing intersection data
            intersection_data = load_intersection_data(dataset.model_path)
            
            if intersection_data is not None:
                # Store intersection data in gaussians for potential use during training
                gaussians.intersection_data = intersection_data
                print("✓ Intersection data loaded successfully")
            else:
                print("✗ No intersection data found to load")
                
        except Exception as e:
            print(f"✗ Error loading intersection data: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with training without intersection data...")

    """ Training """
    viewpoint_stack = None
    ema_dict_for_log = defaultdict(int)
    
    # Choose a single camera for timelapse at the beginning
    timelapse_camera = None
    if pipe.enable_timelapse:
        training_cameras = scene.getTrainCameras()
        if len(training_cameras) > 0:
            # Use specified camera index, or first camera if index is out of range
            camera_index = min(pipe.timelapse_camera_index, len(training_cameras) - 1)
            timelapse_camera = training_cameras[camera_index]
            print(f"✓ Selected camera '{timelapse_camera.image_name}' (index {camera_index}) for timelapse")
            
            # Print timelapse configuration
            if pipe.use_dynamic_timelapse:
                print(f"📹 Dynamic timelapse intervals:")
                print(f"   Iterations 1-{pipe.timelapse_breakpoint_1}: every {pipe.timelapse_interval_early} iterations")
                print(f"   Iterations {pipe.timelapse_breakpoint_1+1}-{pipe.timelapse_breakpoint_2}: every {pipe.timelapse_interval_mid} iterations")
                print(f"   Iterations {pipe.timelapse_breakpoint_2+1}-{pipe.timelapse_breakpoint_3}: every {pipe.timelapse_interval_late} iterations")
                print(f"   Iterations {pipe.timelapse_breakpoint_3+1}+: every {pipe.timelapse_interval_final} iterations")
            else:
                print(f"📹 Fixed timelapse interval: every {pipe.timelapse_interval} iterations")
        else:
            print("Warning: No training cameras found for timelapse!")
    elif not pipe.enable_timelapse:
        print("ℹ️  Timelapse functionality disabled")
    
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress",
                        initial=first_iter, total=opt.iterations)
    
    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)
        
        # Clear combined MLP cache at the start of each iteration
        if hasattr(gaussians, 'clear_combined_mlp_cache'):
            gaussians.clear_combined_mlp_cache()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        # Initialize loss as a Tensor to avoid .backward() errors when skipping photo loss
        # Use a known device (background tensor) since viewpoint_cam isn't set yet
        init_device = background.device if isinstance(background, torch.Tensor) else ("cuda" if torch.cuda.is_available() else "cpu")
        loss = torch.tensor(0.0, device=init_device)
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        pbr_kwargs["iteration"] = iteration - first_iter
        # Provide deferred renderer with conservative pixel and sample chunking
        total_samples = int(pbr_kwargs.get("sample_num", getattr(pipe, "sample_num", 32)))
        # Stream samples across batches to keep peak memory low
        samples_per_batch = max(1, min(8, total_samples))
        num_batches = max(1, (total_samples + samples_per_batch - 1) // samples_per_batch)
        # Pixel chunking (ensure we don't process all pixels at once)
        deferred_options = {
            "max_pixels_per_pass": 32768,
            "samples_per_batch": samples_per_batch,
            "num_batches": num_batches,
        }
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                               opt=opt, is_training=True, dict_params=pbr_kwargs,
                               iteration=iteration, deferred_options=deferred_options)

        viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        tb_dict = render_pkg["tb_dict"]
        if not getattr(args, 'supervise_with_normals_only', False) and not getattr(args, 'use_normals_as_rgb_gt', False):
            loss += render_pkg["loss"]

        # If using normals as RGB GT, compute and add replacement image loss here (works for both PBR and base 3DGS)
        if getattr(args, 'use_normals_as_rgb_gt', False) and getattr(args, 'normal_gt_folder', None) is not None:
            try:
                # Build candidate normal filenames
                original_basename = viewpoint_cam.image_name
                number_part = os.path.splitext(original_basename)[0]
                number_parts = [number_part]
                if number_part.startswith("r_"):
                    number_parts.append("r_r_" + number_part[2:])
                elif number_part.startswith("r_r_"):
                    number_parts.append("r_" + number_part[4:])

                normal_filename_options = []
                for num_part in number_parts:
                    normal_filename_options.extend([
                        f"{num_part}_normalcamera0001.exr",
                        f"{num_part}_normal0001.exr",
                        f"{num_part}_normal.exr",
                        f"{num_part}_normal.png"
                    ])

                normal_gt_path = None
                normal_folder = args.normal_gt_folder
                if normal_folder.startswith('/'):
                    normal_folder = normal_folder[1:]
                for fname in normal_filename_options:
                    p = os.path.join(dataset.source_path, normal_folder, fname)
                    if os.path.exists(p):
                        normal_gt_path = p
                        break

                if normal_gt_path is not None:
                    device = viewpoint_cam.original_image.device
                    exr_alpha_mask = None  # optional alpha from EXR/PNG
                    # Load and transform normals to world space if needed
                    if getattr(args, 'normal_gt_is_camera_space', False):
                        normal_gt = load_normal_image(normal_gt_path).to(device)  # [3,H,W] in [-1,1], normalized
                        Hn, Wn = normal_gt.shape[1], normal_gt.shape[2]
                        # If EXR, try to read alpha channel as mask
                        if normal_gt_path.endswith('.exr'):
                            try:
                                file = OpenEXR.InputFile(normal_gt_path)
                                dw = file.header()['dataWindow']
                                size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                                if 'A' in file.header()['channels']:
                                    a = np.array(array.array('f', file.channel('A', FLOAT))).reshape(size[1], size[0])
                                    exr_alpha_mask = torch.from_numpy(a).float().view(1, size[1], size[0]).to(device)
                                file.close()
                            except Exception:
                                exr_alpha_mask = None
                        camera_rotation_matrix = get_camera_to_world_rotation_matrix(viewpoint_cam)
                        camera_rotation_tensor = torch.from_numpy(camera_rotation_matrix).float().to(device)
                        normal_flat = normal_gt.permute(1, 2, 0).reshape(-1, 3)
                        world_flat = torch.matmul(normal_flat, camera_rotation_tensor)
                        world_flat = F.normalize(world_flat, dim=-1, eps=1e-6)
                        normal_world = world_flat.reshape(Hn, Wn, 3).permute(2, 0, 1)
                    else:
                        # Load as raw tensor and map range
                        if normal_gt_path.endswith('.exr'):
                            file = OpenEXR.InputFile(normal_gt_path)
                            dw = file.header()['dataWindow']
                            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                            r = np.array(array.array('f', file.channel('R', FLOAT))).reshape(size[1], size[0])
                            g = np.array(array.array('f', file.channel('G', FLOAT))).reshape(size[1], size[0])
                            b = np.array(array.array('f', file.channel('B', FLOAT))).reshape(size[1], size[0])
                            # optional alpha
                            try:
                                if 'A' in file.header()['channels']:
                                    a = np.array(array.array('f', file.channel('A', FLOAT))).reshape(size[1], size[0])
                                    exr_alpha_mask = torch.from_numpy(a).float().view(1, size[1], size[0]).to(device)
                            except Exception:
                                exr_alpha_mask = None
                            file.close()
                            normal_world = torch.from_numpy(np.stack([r, g, b], axis=0)).float().to(device)
                        else:
                            normal_world = torchvision.io.read_image(normal_gt_path).float().to(device)
                            if normal_world.shape[0] >= 4:
                                exr_alpha_mask = normal_world[3:4].clone()
                                normal_world = normal_world[:3]
                            if normal_world.max() > 1.0:
                                normal_world = normal_world / 255.0
                        if getattr(args, 'normal_gt_range', '-1_1') == '0_1':
                            normal_world = normal_world * 2.0 - 1.0
                        normal_world = F.normalize(normal_world, dim=0, eps=1e-6)

                    # Map to [0,1] RGB
                    target_rgb = torch.clamp(normal_world * 0.5 + 0.5, 0.0, 1.0)

                    # Predicted image
                    pred_image = render_pkg["pbr"] if is_pbr else render_pkg["render"]
                    if pred_image.shape[1:] != target_rgb.shape[1:]:
                        target_rgb = F.interpolate(target_rgb[None], size=pred_image.shape[1:], mode='bilinear', align_corners=True)[0]

                    # Extract mask from alpha channel of ground truth image
                    gt_image_full = viewpoint_cam.original_image.to(device)
                    
                    # If GT image has 4 channels (RGBA), extract alpha as mask
                    if gt_image_full.dim() == 3 and gt_image_full.shape[0] == 4:
                        # Extract alpha channel as mask
                        img_mask = gt_image_full[3:4, ...].clamp(0.0, 1.0)
                    else:
                        # If no alpha channel, use all ones as mask
                        img_mask = torch.ones((1, gt_image_full.shape[1], gt_image_full.shape[2]), device=device, dtype=gt_image_full.dtype)
                    
                    # Resize mask to match pred_image if needed
                    if img_mask.shape[1:] != pred_image.shape[1:]:
                        img_mask = F.interpolate(img_mask[None], size=pred_image.shape[1:], mode='nearest')[0]
                    mask3 = img_mask.repeat(3, 1, 1)
                    
                    # Apply mask to target_rgb (multiply RGB channels with mask)
                    target_rgb = target_rgb * mask3

                    # Apply mask to BOTH pred and target; normalize by mask coverage
                    diff = (pred_image - target_rgb).abs() * mask3
                    denom = torch.clamp(mask3.sum(), min=1e-6)
                    loss += diff.sum() / denom

                    # Debug print/save
                    if iteration % 50 == 0 or iteration == first_iter + 1:
                        try:
                            tmin = [float(target_rgb[c].min().item()) for c in range(3)]
                            tmax = [float(target_rgb[c].max().item()) for c in range(3)]
                            tmean = [float(target_rgb[c].mean().item()) for c in range(3)]
                            cov = float(img_mask.mean().item())
                            print(f"[Normals-as-RGB GT] Using: {normal_gt_path}")
                            print(f"[Normals-as-RGB GT] stats min={tmin} max={tmax} mean={tmean} mask_coverage={cov:.4f}")
                            if getattr(args, 'save_training_vis', False):
                                vis_dir_rgb = os.path.join(args.model_path, "visualize", "gt_normals_rgb")
                                vis_dir_rgba = os.path.join(args.model_path, "visualize", "gt_normals_rgba")
                                os.makedirs(vis_dir_rgb, exist_ok=True)
                                os.makedirs(vis_dir_rgba, exist_ok=True)
                                # Save unclamped RGB target
                                save_image(target_rgb.clamp(0,1), os.path.join(vis_dir_rgb, f"{iteration:06d}_{viewpoint_cam.image_name}.png"))
                                # Save the exact tensor used for loss with alpha channel
                                alpha = img_mask[0:1].clamp(0,1)  # 1,H,W
                                normal_gt_rgba = torch.cat([target_rgb.clamp(0,1), alpha], dim=0)  # 4,H,W
                                save_image(normal_gt_rgba, os.path.join(vis_dir_rgba, f"{iteration:06d}_{viewpoint_cam.image_name}.png"))
                        except Exception:
                            pass
            except Exception as e:
                print(f"[ERROR] Normals-as-RGB GT (main loss) - Error processing {viewpoint_cam.image_name}: {e}")
        
        
        
        # Extract mask once from the ground truth render for use with all buffers
        gt_image_full = viewpoint_cam.original_image.cuda()
        # Check if ground truth has alpha channel (4 channels)
        if gt_image_full.shape[0] == 4:
            # Extract RGB channels
            gt_image = gt_image_full[:3, ...]
            # Extract alpha channel as mask
            gt_mask = gt_image_full[3:4, ...].clamp(0.0, 1.0)
        else:
            # If no alpha, create a mask of all ones
            gt_image = gt_image_full
            if gt_image.dim() == 3 and gt_image.shape[0] > 3:
                gt_image = gt_image[:3, ...]
            gt_mask = torch.ones((1, gt_image.shape[1], gt_image.shape[2])).to("cuda")
        
        # Apply mask to RGB channels of ground truth
        gt_mask_3c = gt_mask.repeat(3, 1, 1) if gt_mask.shape[0] == 1 else gt_mask
        gt_image = gt_image * gt_mask_3c

        # Optional normal GT supervision comparing predicted normals (skip when using normals as RGB GT)
        if getattr(args, 'normal_gt_folder', None) is not None and not getattr(args, 'use_normals_as_rgb_gt', False):
            try:
                # Build candidate filenames from camera name
                original_basename = viewpoint_cam.image_name
                number_part = os.path.splitext(original_basename)[0]
                number_parts = [number_part]
                if number_part.startswith("r_"):
                    number_parts.append("r_r_" + number_part[2:])
                elif number_part.startswith("r_r_"):
                    number_parts.append("r_" + number_part[4:])

                normal_filename_options = []
                for num_part in number_parts:
                    # Common patterns
                    normal_filename_options.extend([
                        f"{num_part}_normalcamera0001.exr",
                        f"{num_part}_normal0001.exr",
                        f"{num_part}_normal.exr",
                        f"{num_part}_normal.png"
                    ])

                normal_gt_path = None
                normal_folder = args.normal_gt_folder
                if normal_folder.startswith('/'):
                    normal_folder = normal_folder[1:]
                for fname in normal_filename_options:
                    p = os.path.join(dataset.source_path, normal_folder, fname)
                    if os.path.exists(p):
                        normal_gt_path = p
                        break

                if normal_gt_path is not None:
                    # Load GT normals (EXR/PNG)
                    if normal_gt_path.endswith('.exr'):
                        file = OpenEXR.InputFile(normal_gt_path)
                        dw = file.header()['dataWindow']
                        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                        r_str = file.channel('R', FLOAT)
                        g_str = file.channel('G', FLOAT)
                        b_str = file.channel('B', FLOAT)
                        r = np.array(array.array('f', r_str)).reshape(size[1], size[0])
                        g = np.array(array.array('f', g_str)).reshape(size[1], size[0])
                        b = np.array(array.array('f', b_str)).reshape(size[1], size[0])
                        file.close()
                        normal_gt = torch.from_numpy(np.stack([r, g, b], axis=0)).float()
                    else:
                        normal_gt = torchvision.io.read_image(normal_gt_path).float()
                        if normal_gt.shape[0] >= 4:
                            normal_gt = normal_gt[:3]
                        if normal_gt.max() > 1.0:
                            normal_gt = normal_gt / 255.0

                    # Ensure GT normals are on the same device as images/predictions
                    device = gt_image.device
                    normal_gt = normal_gt.to(device)

                    # Convert range to [-1,1] as training target
                    if getattr(args, 'normal_gt_range', '-1_1') == '0_1':
                        normal_gt = normal_gt * 2.0 - 1.0

                    # If GT is camera-space, rotate to world on the fly
                    if getattr(args, 'normal_gt_is_camera_space', False):
                        camera_rotation_matrix = get_camera_to_world_rotation_matrix(viewpoint_cam)
                        camera_rotation_tensor = torch.from_numpy(camera_rotation_matrix).float().to(device)
                        Hn, Wn = normal_gt.shape[1], normal_gt.shape[2]
                        normal_flat = normal_gt.permute(1, 2, 0).reshape(-1, 3)
                        world_flat = torch.matmul(normal_flat, camera_rotation_tensor)
                        world_flat = F.normalize(world_flat, dim=-1, eps=1e-6)
                        normal_gt = world_flat.reshape(Hn, Wn, 3).permute(2, 0, 1)
                    else:
                        normal_gt = F.normalize(normal_gt, dim=0, eps=1e-6)

                    # Predicted normals from renderer
                    pred_normal = render_pkg.get('normal', None)
                    if pred_normal is None:
                        pred_normal = render_pkg.get('pseudo_normal', None)

                    if pred_normal is not None:
                        # Match sizes if needed
                        if pred_normal.shape[1:] != normal_gt.shape[1:]:
                            normal_gt = F.interpolate(normal_gt[None], size=pred_normal.shape[1:], mode='bilinear', align_corners=True)[0]
                        # Apply mask and compute loss
                        normal_mask = gt_mask
                        if normal_mask.shape[0] == 1:
                            normal_mask = normal_mask.repeat(3, 1, 1)
                        normal_loss = F.l1_loss(pred_normal * normal_mask, normal_gt.to(pred_normal.device) * normal_mask)
                        lambda_normal = getattr(opt, 'lambda_normal_buffer', 1.0)
                        loss += lambda_normal * normal_loss
                # else: silently skip if not found
            except Exception as e:
                print(f"[ERROR] Normal GT - Error loading/processing normal for {viewpoint_cam.image_name}: {e}")
        
        # Consolidate all PBR supervision in a single block
        if is_pbr:
            # Base color supervision
            if args.base_color_folder is not None:
                original_basename = viewpoint_cam.image_name
                number_part = os.path.splitext(original_basename)[0]  # Get the <number> part
                
                # Create alternative number parts to support both "r_" and "r_r_" patterns
                number_parts = [number_part]
                if number_part.startswith("r_"):
                    # Add "r_r_" version
                    alt_number_part = "r_r_" + number_part[2:]  # Replace "r_" with "r_r_"
                    number_parts.append(alt_number_part)
                elif number_part.startswith("r_r_"):
                    # Add "r_" version  
                    alt_number_part = "r_" + number_part[4:]  # Replace "r_r_" with "r_"
                    number_parts.append(alt_number_part)
                
                # Try both naming patterns for each number part
                base_color_filename_options = []
                for num_part in number_parts:
                    if num_part.startswith("r_r_"):
                        # Pattern: r_r_<number>_albedo.png
                        base_color_filename_options.append(f"{num_part}_albedo.png")
                    else:
                        # Patterns: r_<number>_basecolor0001.png and r_<number>_albedo.png
                        base_color_filename_options.extend([
                            f"{num_part}_basecolor0001.png",
                            f"{num_part}_basecolor.png",
                            f"{num_part}_albedo.png"
                        ])

                base_color_gt_path = None
                base_color_folder = args.base_color_folder
                if base_color_folder.startswith('/'):
                    base_color_folder = base_color_folder[1:]  # Remove leading slash
                
                # Store attempted paths for debugging
                attempted_base_color_paths = []
                for base_color_filename in base_color_filename_options:
                    path = os.path.join(dataset.source_path, base_color_folder, base_color_filename)
                    attempted_base_color_paths.append(path)
                    if os.path.exists(path):
                        base_color_gt_path = path
                        break

                if base_color_gt_path is not None:
                    try:
                        # Load and transform DiffusionRenderer base color image (with gamma 1.8)
                        base_color_gt = load_and_transform_diffusion_renderer_image(
                            base_color_gt_path, modality='basecolor', device=viewpoint_cam.original_image.device
                        )
                        
                        # Convert ground truth base color to sRGB for supervision
                        base_color_gt = rgb_to_srgb(base_color_gt)
                        
                        base_color_pred = render_pkg['base_color']
                        # Choose loss function based on flag
                        if args.base_color_l1_loss:
                            # Standard L1 loss
                            base_color_loss = l1_loss(base_color_pred, base_color_gt * gt_mask)
                        else:
                            # Scale-invariant loss (default) - allows model to learn correct
                            # base color distribution without being penalized for global scale differences
                            base_color_loss = scale_invariant_l1_loss(base_color_pred, base_color_gt, mask=gt_mask)
                        
                        # Save the masked base color prediction at intervals
                        if iteration % 100 == 0:
                            os.makedirs(os.path.join(args.model_path, "visualize", "base_color_masked"), exist_ok=True)
                            save_image(base_color_pred, 
                                    os.path.join(args.model_path, "visualize", "base_color_masked", 
                                                f"{iteration:06d}_{viewpoint_cam.image_name}.png"))
                        
                        loss += opt.lambda_base_color_buffer * base_color_loss
                    except Exception as e:
                        print(f"[ERROR] Base Color - Error loading image {base_color_gt_path}: {e}")
                        print(f"[ERROR] Base Color - Exception type: {type(e).__name__}")
                        print(f"[ERROR] Base Color - Shape: {base_color_gt.shape if 'base_color_gt' in locals() else 'N/A'}, dtype: {base_color_gt.dtype if 'base_color_gt' in locals() else 'N/A'}")
                else:
                    print(f"[WARNING] Base Color - Image not found. Searched in these paths:")
                    for path in attempted_base_color_paths:
                        print(f"[WARNING] Base Color -   {path}")
            
            # Roughness supervision
            if args.roughness_folder is not None:
                original_basename = viewpoint_cam.image_name
                number_part = os.path.splitext(original_basename)[0]  # Get the <number> part
                
                # Create alternative number parts to support both "r_" and "r_r_" patterns
                number_parts = [number_part]
                if number_part.startswith("r_"):
                    # Add "r_r_" version
                    alt_number_part = "r_r_" + number_part[2:]  # Replace "r_" with "r_r_"
                    number_parts.append(alt_number_part)
                elif number_part.startswith("r_r_"):
                    # Add "r_" version  
                    alt_number_part = "r_" + number_part[4:]  # Replace "r_r_" with "r_"
                    number_parts.append(alt_number_part)
                
                # Try both naming patterns for each number part
                roughness_filename_options = []
                for num_part in number_parts:
                    if num_part.startswith("r_r_"):
                        # Pattern: r_r_<number>_roughness.png or r_r_<number>_metallicroughness.png
                        roughness_filename_options.extend([
                            f"{num_part}_roughness.png",
                            f"{num_part}_metallicroughness.png"
                        ])
                    else:
                        # Patterns: r_<number>_roughness0001.png, r_<number>_metallicroughness0001.png, r_<number>_roughness.png, r_<number>_metallicroughness.png
                        roughness_filename_options.extend([
                            f"{num_part}_roughness0001.png",
                            f"{num_part}_metallicroughness0001.png",
                            f"{num_part}_roughness.png",
                            f"{num_part}_metallicroughness.png"
                        ])

                roughness_gt_path = None
                roughness_folder = args.roughness_folder
                if roughness_folder.startswith('/'):
                    roughness_folder = roughness_folder[1:]  # Remove leading slash
                
                # Store attempted paths for debugging
                attempted_roughness_paths = []
                for roughness_filename in roughness_filename_options:
                    path = os.path.join(dataset.source_path, roughness_folder, roughness_filename)
                    attempted_roughness_paths.append(path)
                    if os.path.exists(path):
                        roughness_gt_path = path
                        break

                if roughness_gt_path is not None:
                    try:
                        # Load and transform DiffusionRenderer roughness image (linear, no gamma)
                        roughness_gt = load_and_transform_diffusion_renderer_image(
                            roughness_gt_path, modality='roughness', device=viewpoint_cam.original_image.device
                        )
                        
                        # Extract green channel (index 1) for roughness
                        if roughness_gt.shape[0] >= 3:  # RGB or RGBA
                            roughness_gt_green = roughness_gt[1:2]  # Get just the green channel
                        else:
                            # Single channel image
                            roughness_gt_green = roughness_gt
                        
                        roughness_pred = render_pkg['roughness']
                        
                        # Make sure prediction has compatible shape with ground truth
                        if len(roughness_pred.shape) == 3 and roughness_pred.shape[0] == 3:
                            # If prediction has 3 channels, take first channel
                            roughness_pred = roughness_pred[0:1]
                        elif len(roughness_pred.shape) == 2:
                            # If prediction has no channel dimension, add one
                            roughness_pred = roughness_pred.unsqueeze(0)
                        
                        # Apply the ground truth mask
                        roughness_loss = l1_loss(roughness_pred, roughness_gt_green * gt_mask)
                        
                        # Save the roughness prediction and ground truth at intervals
                        if iteration % 100 == 0:
                            os.makedirs(os.path.join(args.model_path, "visualize", "roughness_debug"), exist_ok=True)
                            # Create a grid with prediction and ground truth side by side
                            roughness_grid = make_grid([
                                roughness_pred.repeat(3, 1, 1) if roughness_pred.shape[0] == 1 else roughness_pred,
                                roughness_gt_green.repeat(3, 1, 1),
                                gt_mask.repeat(3, 1, 1)
                            ], nrow=3, padding=5, normalize=False)
                            save_image(
                                roughness_grid,
                                os.path.join(args.model_path, "visualize", "roughness_debug", 
                                            f"{iteration:06d}_{viewpoint_cam.image_name}.png")
                            )
                        
                        loss += opt.lambda_roughness_buffer * roughness_loss
                    except Exception as e:
                        print(f"[ERROR] Roughness - Error processing image {roughness_gt_path}: {e}")
                        print(f"[ERROR] Roughness - Exception type: {type(e).__name__}")
                        print(f"[ERROR] Roughness - Shape: {roughness_gt.shape if 'roughness_gt' in locals() else 'N/A'}, dtype: {roughness_gt.dtype if 'roughness_gt' in locals() else 'N/A'}")
                else:
                    print(f"[WARNING] Roughness - Image not found. Searched in these paths:")
                    for path in attempted_roughness_paths:
                        print(f"[WARNING] Roughness -   {path}")

            # Diffuse supervision
            if args.diffuse_folder is not None:
                original_basename = viewpoint_cam.image_name
                number_part = os.path.splitext(original_basename)[0]  # Get the <number> part
                
                # Create alternative number parts to support both "r_" and "r_r_" patterns
                number_parts = [number_part]
                if number_part.startswith("r_"):
                    # Add "r_r_" version
                    alt_number_part = "r_r_" + number_part[2:]  # Replace "r_" with "r_r_"
                    number_parts.append(alt_number_part)
                elif number_part.startswith("r_r_"):
                    # Add "r_" version  
                    alt_number_part = "r_" + number_part[4:]  # Replace "r_r_" with "r_"
                    number_parts.append(alt_number_part)
                
                # Try both file patterns for each number part
                diffuse_filename_options = []
                for num_part in number_parts:
                    if num_part.startswith("r_r_"):
                        # Pattern: r_r_<number>_irradiance.png or r_r_<number>_irradiance_total.exr
                        diffuse_filename_options.extend([
                            f"{num_part}_irradiance.png",
                            f"{num_part}_irradiance_total.exr"
                        ])
                    else:
                        # Patterns: r_<number>_irradiance0001.png, r_<number>_irradiance_total0001.exr, r_<number>_irradiance.png, r_<number>_irradiance_total.exr
                        diffuse_filename_options.extend([
                            f"{num_part}_irradiance0001.png",
                            f"{num_part}_irradiance_total0001.exr",
                            f"{num_part}_irradiance.png",
                            f"{num_part}_irradiance_total.exr"
                        ])

                diffuse_gt_path = None
                diffuse_folder = args.diffuse_folder
                if diffuse_folder.startswith('/'):
                    diffuse_folder = diffuse_folder[1:]  # Remove leading slash
                
                # Store attempted paths for debugging
                attempted_diffuse_paths = []
                for diffuse_filename in diffuse_filename_options:
                    path = os.path.join(dataset.source_path, diffuse_folder, diffuse_filename)
                    attempted_diffuse_paths.append(path)
                    if os.path.exists(path):
                        diffuse_gt_path = path
                        break

                if diffuse_gt_path is not None:
                    try:
                        # Check file extension to determine how to load it
                        if diffuse_gt_path.endswith('.exr'):
                            # Read EXR file
                            exr_file = OpenEXR.InputFile(diffuse_gt_path)
                            dw = exr_file.header()['dataWindow']
                            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                            
                            # Read RGB channels
                            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                            r_str = exr_file.channel('R', FLOAT)
                            g_str = exr_file.channel('G', FLOAT)
                            b_str = exr_file.channel('B', FLOAT)
                            
                            # Convert to numpy arrays
                            r = np.array(array.array('f', r_str)).reshape(size[1], size[0])
                            g = np.array(array.array('f', g_str)).reshape(size[1], size[0])
                            b = np.array(array.array('f', b_str)).reshape(size[1], size[0])
                            
                            # Create tensor with shape [3, H, W]
                            diffuse_gt = torch.from_numpy(np.stack([r, g, b], axis=0)).float()
                            diffuse_gt = diffuse_gt.to(viewpoint_cam.original_image.device)
                        else:
                            # Load PNG file
                            diffuse_gt = torchvision.io.read_image(diffuse_gt_path).float() / 255.0
                            diffuse_gt = diffuse_gt.to(viewpoint_cam.original_image.device)
                            
                            # If the image has 4 channels (RGBA), take only RGB channels
                            if diffuse_gt.shape[0] == 4:
                                diffuse_gt = diffuse_gt[:3]
                        
                        diffuse_pred = render_pkg['diffuse']
                        
                        # Apply ground truth mask to both prediction and ground truth
                        diffuse_loss = l1_loss(diffuse_pred, diffuse_gt * gt_mask)
                        
                        # Save the masked diffuse prediction at intervals
                        if iteration % 100 == 0:
                            os.makedirs(os.path.join(args.model_path, "visualize", "diffuse_masked"), exist_ok=True)
                            save_image(diffuse_pred, 
                                      os.path.join(args.model_path, "visualize", "diffuse_masked", 
                                                  f"{iteration:06d}_{viewpoint_cam.image_name}.png"))
                        
                        loss += opt.lambda_diffuse_buffer * diffuse_loss
                    except Exception as e:
                        print(f"[ERROR] Diffuse - Error loading image {diffuse_gt_path}: {e}")
                        print(f"[ERROR] Diffuse - Exception type: {type(e).__name__}")
                        print(f"[ERROR] Diffuse - Shape: {diffuse_gt.shape if 'diffuse_gt' in locals() else 'N/A'}, dtype: {diffuse_gt.dtype if 'diffuse_gt' in locals() else 'N/A'}")
                        import traceback
                        print(f"[ERROR] Diffuse - Full traceback:")
                        traceback.print_exc()
                else:
                    print(f"[WARNING] Diffuse - Image not found. Searched in these paths:")
                    for path in attempted_diffuse_paths:
                        print(f"[WARNING] Diffuse -   {path}")
            
            # Metallic supervision
            if args.metallic_folder is not None:
                original_basename = viewpoint_cam.image_name
                number_part = os.path.splitext(original_basename)[0]  # Get the <number> part
                
                # Create alternative number parts to support both "r_" and "r_r_" patterns
                number_parts = [number_part]
                if number_part.startswith("r_"):
                    # Add "r_r_" version
                    alt_number_part = "r_r_" + number_part[2:]  # Replace "r_" with "r_r_"
                    number_parts.append(alt_number_part)
                elif number_part.startswith("r_r_"):
                    # Add "r_" version  
                    alt_number_part = "r_" + number_part[4:]  # Replace "r_r_" with "r_"
                    number_parts.append(alt_number_part)
                
                # Try both naming patterns for each number part
                metallic_filename_options = []
                for num_part in number_parts:
                    if num_part.startswith("r_r_"):
                        # Pattern: r_r_<number>_metallic.png or r_r_<number>_metallicroughness.png
                        metallic_filename_options.extend([
                            f"{num_part}_metallic.png",
                            f"{num_part}_metallicroughness.png"
                        ])
                    else:
                        # Patterns: r_<number>_metallic0001.png, r_<number>_metallicroughness0001.png, r_<number>_metallic.png, r_<number>_metallicroughness.png
                        metallic_filename_options.extend([
                            f"{num_part}_metallic0001.png",
                            f"{num_part}_metallicroughness0001.png",
                            f"{num_part}_metallic.png",
                            f"{num_part}_metallicroughness.png"
                        ])

                metallic_gt_path = None
                metallic_folder = args.metallic_folder
                if metallic_folder.startswith('/'):
                    metallic_folder = metallic_folder[1:]  # Remove leading slash
                
                # Store attempted paths for debugging
                attempted_metallic_paths = []
                for metallic_filename in metallic_filename_options:
                    path = os.path.join(dataset.source_path, metallic_folder, metallic_filename)
                    attempted_metallic_paths.append(path)
                    if os.path.exists(path):
                        metallic_gt_path = path
                        break

                if metallic_gt_path is not None:
                    try:
                        # Load and transform DiffusionRenderer metallic image (with gamma 1.8)
                        metallic_gt = load_and_transform_diffusion_renderer_image(
                            metallic_gt_path, modality='metallic', device=viewpoint_cam.original_image.device
                        )
                        
                        # Extract blue channel (index 2) for metallic from metallicroughness or use single channel
                        if metallic_gt.shape[0] >= 3:  # RGB or RGBA
                            metallic_gt_channel = metallic_gt[0:1]  # Get just the red channel for metallic
                        else:
                            # Single channel image
                            metallic_gt_channel = metallic_gt
                        
                        metallic_pred = render_pkg['metallic']
                        
                        # Make sure prediction has compatible shape with ground truth
                        if len(metallic_pred.shape) == 3 and metallic_pred.shape[0] == 3:
                            # If prediction has 3 channels, take first channel
                            metallic_pred = metallic_pred[0:1]
                        elif len(metallic_pred.shape) == 2:
                            # If prediction has no channel dimension, add one
                            metallic_pred = metallic_pred.unsqueeze(0)
                        
                        # Apply the ground truth mask
                        metallic_loss = l1_loss(metallic_pred, metallic_gt_channel * gt_mask)
                        
                        # Save the metallic prediction and ground truth at intervals
                        if iteration % 100 == 0:
                            os.makedirs(os.path.join(args.model_path, "visualize", "metallic_debug"), exist_ok=True)
                            # Create a grid with prediction and ground truth side by side
                            metallic_grid = make_grid([
                                metallic_pred.repeat(3, 1, 1) if metallic_pred.shape[0] == 1 else metallic_pred,
                                metallic_gt_channel.repeat(3, 1, 1),
                                gt_mask.repeat(3, 1, 1)
                            ], nrow=3, padding=5, normalize=False)
                            save_image(
                                metallic_grid,
                                os.path.join(args.model_path, "visualize", "metallic_debug", 
                                            f"{iteration:06d}_{viewpoint_cam.image_name}.png")
                            )
                        
                        loss += opt.lambda_metallic_buffer * metallic_loss
                    except Exception as e:
                        print(f"[ERROR] Metallic - Error processing image {metallic_gt_path}: {e}")
                        print(f"[ERROR] Metallic - Exception type: {type(e).__name__}")
                        print(f"[ERROR] Metallic - Shape: {metallic_gt.shape if 'metallic_gt' in locals() else 'N/A'}, dtype: {metallic_gt.dtype if 'metallic_gt' in locals() else 'N/A'}")
                else:
                    print(f"[WARNING] Metallic - Image not found. Searched in these paths:")
                    for path in attempted_metallic_paths:
                        print(f"[WARNING] Metallic -   {path}")

        # Check loss for NaN/Inf before backward pass
        skip_optimization = False
        if not torch.isfinite(loss):
            print(f"[ERROR] Loss is NaN/Inf at iteration {iteration} - skipping backward pass and optimizer step")
            print(f"[ERROR] Loss value: {loss.item() if hasattr(loss, 'item') else loss}")
            # Try to identify which component caused the NaN
            try:
                if hasattr(loss, 'item'):
                    loss_val = loss.item()
                    if not np.isfinite(loss_val):
                        print(f"[ERROR] Loss.item() = {loss_val}")
            except Exception as e:
                print(f"[ERROR] Could not extract loss value: {e}")
            skip_optimization = True
        else:
            loss.backward()
            
            # Sanitize gradients before optimizer step to prevent NaN propagation
            for param_group in gaussians.optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        if not torch.isfinite(param.grad).all():
                            print(f"[WARNING] NaN/Inf gradients detected in {param_group.get('name', 'unknown')} - sanitizing")
                            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Also sanitize gradients for PBR components (e.g., env_light)
            for key, component in pbr_kwargs.items():
                if hasattr(component, 'optimizer') and component.optimizer is not None:
                    try:
                        for param_group in component.optimizer.param_groups:
                            for param in param_group['params']:
                                if param.grad is not None:
                                    if not torch.isfinite(param.grad).all():
                                        print(f"[WARNING] NaN/Inf gradients detected in {key} ({param_group.get('name', 'unknown')}) - sanitizing")
                                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception as e:
                        # Some components might not have standard optimizer structure
                        pass

        if is_pbr and args.use_mlp and args.enable_gradient_logging:
            has_mlp = (hasattr(gaussians, 'combined_mlp') and gaussians.combined_mlp is not None) or \
                      (hasattr(gaussians, 'base_color_mlp') and gaussians.base_color_mlp is not None)
            if has_mlp:
                log_mlp_gradients(gaussians, iteration, loss)

        with torch.no_grad():
            if pipe.save_training_vis:
                save_training_vis(
                    viewpoint_cam,
                    gaussians,
                    background,
                    render_fn,
                    pipe,
                    opt,
                    first_iter,
                    iteration,
                    pbr_kwargs,
                    args,
                )
            
            # Render timelapse frame with dynamic intervals based on training progress
            if pipe.enable_timelapse and timelapse_camera is not None:
                # Dynamic timelapse intervals based on training progress
                if pipe.use_dynamic_timelapse:
                    if iteration <= pipe.timelapse_breakpoint_1:
                        timelapse_interval = pipe.timelapse_interval_early
                    elif iteration <= pipe.timelapse_breakpoint_2:
                        timelapse_interval = pipe.timelapse_interval_mid
                    elif iteration <= pipe.timelapse_breakpoint_3:
                        timelapse_interval = pipe.timelapse_interval_late
                    else:
                        timelapse_interval = pipe.timelapse_interval_final
                else:
                    # Use fixed interval
                    timelapse_interval = pipe.timelapse_interval
                
                if iteration % timelapse_interval == 0:
                    timelapse_dir = os.path.join(args.model_path, "timelapse")
                    render_timelapse_frame(
                        timelapse_camera,
                        gaussians,
                        render_fn,
                        pipe,
                        background,
                        opt,
                        pbr_kwargs,
                        iteration,
                        timelapse_dir,
                        is_pbr,
                    )
                    # Print interval info every 1000 iterations
                    if iteration % 1000 == 0:
                        print(f"📊 Timelapse interval: {timelapse_interval} (iteration {iteration})")
            
            # Progress bar
            pbar_dict = {"num": gaussians.get_xyz.shape[0]}
            if is_pbr:
                # Check for NaN/Inf before computing light_mean
                env_mean = direct_env_light.get_env.mean()
                if torch.isfinite(env_mean):
                    pbar_dict["light_mean"] = env_mean.item()
                else:
                    pbar_dict["light_mean"] = float('nan')
                    print(f"[WARNING] light_mean is NaN/Inf at iteration {iteration}")
                pbar_dict["env"] = direct_env_light.H
            for k in tb_dict:
                if k in ["psnr", "psnr_pbr"]:
                    ema_dict_for_log[k] = 0.4 * tb_dict[k] + 0.6 * ema_dict_for_log[k]
                    pbar_dict[k] = f"{ema_dict_for_log[k]:.{7}f}"
            progress_bar.set_postfix(pbar_dict)

            # Log and save
            training_report(
                tb_writer,
                iteration,
                tb_dict,
                scene,
                render_fn,
                pipe=pipe,
                bg_color=background,
                args=args,
                dict_params=pbr_kwargs,
            )

            # densification (skip when geometry is frozen)
            if not getattr(opt, 'freeze_geometry', False):
                if iteration < opt.densify_until_iter:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, 
                                                        render_pkg['weights'])
                    # Keep track of max radii in image-space for pruning
                    if gaussians.max_radii2D.numel() > 0:
                        vf_device = visibility_filter.to(gaussians.max_radii2D.device)
                        gaussians.max_radii2D[vf_device] = torch.max(gaussians.max_radii2D[vf_device],
                                                              radii[vf_device])
                    
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        densify_grad_normal_threshold = opt.densify_grad_normal_threshold if iteration > opt.normal_densify_from_iter else 99999
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold,
                                                    densify_grad_normal_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (
                            dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            
            # Optimizer step (skip if loss was NaN/Inf)
            if not skip_optimization:
                gaussians.step()
                for component in pbr_kwargs.values():
                    try:
                        component.step()
                    except:
                        pass
            else:
                # Zero gradients even if skipping optimization to prevent accumulation
                gaussians.optimizer.zero_grad()
                for key, component in pbr_kwargs.items():
                    try:
                        if hasattr(component, 'optimizer') and component.optimizer is not None:
                            component.optimizer.zero_grad()
                    except:
                        pass

            # Periodically release cached blocks to mitigate fragmentation
            if iteration % 50 == 0:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            try:
                del render_pkg
            except Exception:
                pass

            # save checkpoints
            if iteration % args.save_interval == 0 or iteration == args.iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, final_iteration=opt.iterations)

            if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
                
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, "chkpnt" + str(iteration) + ".pth"))

                for com_name, component in pbr_kwargs.items():
                    try:
                        torch.save((component.capture(), iteration),
                                   os.path.join(scene.model_path, f"{com_name}_chkpnt" + str(iteration) + ".pth"))
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    except:
                        pass

                    print("[ITER {}] Saving {} Checkpoint".format(iteration, com_name))



    if dataset.eval:
        eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, args)
    
    # Create timelapse video at the end of training
    if pipe.enable_timelapse:
        timelapse_dir = os.path.join(args.model_path, "timelapse")
        # Create output/misc directory if it doesn't exist
        misc_dir = os.path.join("output", "misc")
        os.makedirs(misc_dir, exist_ok=True)
        output_video_path = os.path.join(misc_dir, "training_timelapse.mp4")
        create_timelapse_video(timelapse_dir, output_video_path, fps=pipe.timelapse_fps)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'neilf', 'neilf_deferred'], default='render')
    parser.add_argument("--test_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--base_color_folder", type=str, default=None, 
                        help="Folder name containing base color ground truth images")
    parser.add_argument("--base_color_l1_loss", action="store_true", default=False,
                        help="If set, use L1 loss for base color supervision instead of scale-invariant loss")
    parser.add_argument("--roughness_folder", type=str, default=None, 
                        help="Folder name containing roughness ground truth images")
    parser.add_argument("--normal_folder", type=str, default=None, 
                        help="Folder name containing normal ground truth images")
    parser.add_argument("--diffuse_folder", type=str, default=None, 
                        help="Folder name containing diffuse ground truth images (EXR format)")
    parser.add_argument("--metallic_folder", type=str, default=None, 
                        help="Folder name containing metallic ground truth images")
    parser.add_argument("--normals_folder", type=str, default=None, 
                        help="Folder name containing normal ground truth images")
    parser.add_argument("--normals_are_world_space", action="store_true", default=False,
                        help="If set, normals_folder contains world-space normals; skip camera rotation and Blender reversal.")
    parser.add_argument("--normals_range", type=str, choices=['-1_1','0_1'], default='0_1',
                        help="Range of normals in normals_folder: '-1_1' for [-1,1], '0_1' for [0,1].")
    # Training-time normal supervision (on-the-fly transform)
    parser.add_argument("--normal_gt_folder", type=str, default=None,
                        help="Folder (relative to source_path) containing normal GT images (per-frame)")
    parser.add_argument("--normal_gt_is_camera_space", action="store_true", default=False,
                        help="If true, normal GT is in camera space and will be rotated to world on the fly")
    parser.add_argument("--normal_gt_range", type=str, choices=['-1_1','0_1'], default='-1_1',
                        help="Range of normal GT images: '-1_1' for [-1,1], '0_1' for [0,1].")
    parser.add_argument("--supervise_with_normals_only", action="store_true", default=False,
                        help="If set, disable image reconstruction loss and supervise with normals only (plus any enabled regularizers).")
    parser.add_argument("--use_normals_as_rgb_gt", action="store_true", default=False,
                        help="If set, compare rendered image to normal GT (as RGB) instead of photo GT.")
    parser.add_argument("--dump_normals_at_start", action="store_true", default=False,
                        help="If set, dump all per-view normal GTs (as transformed RGB[A]) at the start of training.")
    parser.add_argument("--dump_normals_dir", type=str, default=None,
                        help="Output directory for dumped normal GTs. Defaults to <model_path>/normal_gt_dump")
    parser.add_argument("--dump_and_exit", action="store_true", default=False,
                        help="If set together with --dump_normals_at_start, stop training immediately after dumping.")
    parser.add_argument("--perform_intersection_tracing", action="store_true", default=False,
                        help="Perform intersection tracing for all training images.")
    parser.add_argument("--load_intersection_data", action="store_true", default=False,
                        help="Load existing intersection data from the model directory.")
    parser.add_argument("--use_mlp", action="store_true", default=False,
                        help="Use MLPs for base_color, roughness, and metallic prediction.")
    parser.add_argument("--combined_mlp", action="store_true", default=False,
                        help="Use a single combined MLP instead of three separate MLPs for base_color, roughness, and metallic. Only effective when --use_mlp is set.")
    parser.add_argument("--enable_gradient_logging", action="store_true", default=False,
                        help="Enable gradient logging for MLPs to diagnose training issues.")
    args = parser.parse_args(sys.argv[1:])

    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    is_pbr = args.type in ['neilf', 'neilf_deferred']
    training(lp.extract(args), op.extract(args), pp.extract(args), is_pbr=is_pbr)

    # All done
    print("\nTraining complete.")
