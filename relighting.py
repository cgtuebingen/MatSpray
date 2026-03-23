import json
import os
import cv2
from gaussian_renderer import render_fn_dict
import numpy as np
import torch
from scene import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene.cameras import Camera
from scene.envmap import EnvLight
from utils.graphics_utils import focal2fov, fov2focal
from torchvision.utils import save_image
from tqdm import tqdm
from utils.graphics_utils import rgb_to_srgb
from plyfile import PlyData
from utils.camera_utils import JSON_to_camera
import shutil

import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


def scene_composition(scene_dict: dict, dataset: ModelParams, render_type: str = "neilf"):
    gaussians_list = []
    for scene in scene_dict:
        ply_path = scene_dict[scene]["path"]
        # Infer SH degree from PLY to avoid mismatch assertions
        sh_degree_in_ply = dataset.sh_degree
        try:
            ply = PlyData.read(ply_path)
            names = [p.name for p in ply.elements[0].properties]
            extra_f = [n for n in names if n.startswith("f_rest_")]
            num_extra = len(extra_f)
            if num_extra <= 0:
                sh_degree_in_ply = 0
            else:
                val = num_extra / 3.0 + 1.0
                d = int(round(np.sqrt(val) - 1.0))
                # Validate expected count
                if 3 * ((d + 1) ** 2) - 3 == num_extra:
                    sh_degree_in_ply = d
        except Exception:
            # Fallback to provided dataset sh_degree
            pass

        gaussians = GaussianModel(sh_degree_in_ply, render_type=render_type)
        gaussians.load_ply(ply_path)
        
        # Always attach MLPs and projected data if they exist in the model directory
        ply_dir = os.path.dirname(scene_dict[scene]["path"])
        mlp_data_dir = os.path.join(ply_dir, "mlp_data")
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
                    
                    # Load normal MLP data if it exists
                    if mlp_metadata.get('has_normal_mlp', False):
                        try:
                            projected_normals = np.load(os.path.join(mlp_data_dir, "projected_normals.npy"))
                            gaussians.projected_normals = torch.tensor(projected_normals, dtype=torch.float, device="cuda", requires_grad=False)
                            print(f"  - Normal MLP data loaded, shape: {gaussians.projected_normals.shape}")
                        except Exception as e:
                            print(f"Warning: Failed to load normal MLP data: {e}")
                    
                    # Initialize MLPs with the same pattern as in train.py
                    num_train_images = mlp_metadata.get('num_train_images', projected_base_colors.shape[1])
                    net_width = mlp_metadata.get('net_width', 64)
                    
                    # Create MLPs
                    from scene.mlp import BaseColorMLP, RoughnessMLP, MetallicMLP, NormalMLP
                    gaussians.base_color_mlp = BaseColorMLP(
                        num_train_images=num_train_images, net_width=net_width
                    ).cuda()
                    gaussians.roughness_mlp = RoughnessMLP(
                        num_train_images=num_train_images, net_width=net_width
                    ).cuda()
                    gaussians.metallic_mlp = MetallicMLP(
                        num_train_images=num_train_images, net_width=net_width
                    ).cuda()
                    
                    # Initialize normal MLP if data exists
                    if mlp_metadata.get('has_normal_mlp', False):
                        gaussians.normal_mlp = NormalMLP(
                            num_train_images=num_train_images, net_width=net_width
                        ).cuda()
                    
                    # Load MLP weights
                    try:
                        gaussians.base_color_mlp.load_state_dict(torch.load(os.path.join(mlp_data_dir, "base_color_mlp_weights.pth")), strict=False)
                        gaussians.roughness_mlp.load_state_dict(torch.load(os.path.join(mlp_data_dir, "roughness_mlp_weights.pth")), strict=False)
                        gaussians.metallic_mlp.load_state_dict(torch.load(os.path.join(mlp_data_dir, "metallic_mlp_weights.pth")), strict=False)
                        if mlp_metadata.get('has_normal_mlp', False):
                            gaussians.normal_mlp.load_state_dict(torch.load(os.path.join(mlp_data_dir, "normal_mlp_weights.pth")), strict=False)
                    except Exception as e:
                        print(f"Warning: Failed to load some MLP weights strictly: {e}")
                    
                    
                    # Set to evaluation mode
                    gaussians.base_color_mlp.eval()
                    gaussians.roughness_mlp.eval()
                    gaussians.metallic_mlp.eval()
                    if mlp_metadata.get('has_normal_mlp', False):
                        gaussians.normal_mlp.eval()
                    
                    print(f"✓ MLPs and projected data attached from {mlp_data_dir}")
                    gaussians.quick_debug_mlp_status()
                    
            except Exception as e:
                print(f"Warning: Failed to load MLP data from {mlp_data_dir}: {e}")
                print("Model will be used without MLPs")

        torch_transform = torch.tensor(scene_dict[scene]["transform"], device="cuda", dtype=torch.float32).reshape(4, 4)
        gaussians.set_transform(transform=torch_transform)

        gaussians_list.append(gaussians)

    gaussians_composite = GaussianModel.create_from_gaussians(gaussians_list, dataset)
    n = gaussians_composite.get_xyz.shape[0]
    print(f"Totally {n} points loaded.")

    gaussians_composite._visibility_rest = (
        torch.nn.Parameter(torch.cat(
            [gaussians_composite._visibility_rest.data,
             torch.zeros(n, 5 ** 2 - 4 ** 2, 1, device="cuda", dtype=torch.float32)],
            dim=1).requires_grad_(True)))

    gaussians_composite._incidents_dc.data[:] = 0
    gaussians_composite._incidents_rest.data[:] = 0

    return gaussians_composite



def render_points(camera, gaussians):
    intrinsic = camera.get_intrinsics()
    w2c = camera.world_view_transform.transpose(0, 1)

    xyz = gaussians.get_xyz
    color = gaussians.get_base_color
    xyz_homo = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
    xyz_cam = (xyz_homo @ w2c.T)[:, :3]
    z = xyz_cam[:, 2]
    uv_homo = xyz_cam @ intrinsic.T
    uv = uv_homo[:, :2] / uv_homo[:, 2:]
    uv = uv.long()

    valid_point = torch.logical_and(torch.logical_and(uv[:, 0] >= 0, uv[:, 0] < W),
                                    torch.logical_and(uv[:, 1] >= 0, uv[:, 1] < H))
    uv = uv[valid_point]
    z = z[valid_point]
    color = color[valid_point]

    depth_buffer = torch.full_like(render_pkg['render'][0], 10000)
    rgb_buffer = torch.full_like(render_pkg['render'], bg)
    while True:
        mask = depth_buffer[uv[:, 1], uv[:, 0]] > z
        if mask.sum() == 0:
            break
        uv_mask = uv[mask]
        depth_buffer[uv_mask[:, 1], uv_mask[:, 0]] = z[mask]
        rgb_buffer[:, uv_mask[:, 1], uv_mask[:, 0]] = color[mask].transpose(-1, -2)

    return rgb_buffer


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('-co', '--config', default=None, required=True, help="Path to transforms_*.json or a dataset directory")
    parser.add_argument('--split', choices=['train', 'test', 'val', 'validation'], default='test', help="Which split to use when --config is a directory")
    parser.add_argument('--ply_path', default=None, required=True, help="Path to the model PLY file to relight")
    parser.add_argument('-e', '--envmap_path', default=None, help="Env map path")
    parser.add_argument('-bg', "--background_color", type=float, default=None,
                        help="If set, use it as background color")
    parser.add_argument('--bake', action='store_true', default=False, help="Bake the visibility and refine.")
    parser.add_argument('--video', action='store_true', default=False, help="If True, output video as well.")
    parser.add_argument('--output', default="./capture_trace", help="Output dir.")
    parser.add_argument('--capture_list', default="pbr_env,diffuse,specular", help="what should be rendered for output.")
    parser.add_argument('--gt_path', default=None, help="Path to ground truth images for metrics calculation")
    parser.add_argument('-t', '--type', choices=['neilf', 'neilf_deferred', 'neilf_deferred_importance'], default='neilf_deferred',
                        help="Renderer type for relighting")
    parser.add_argument('--no_precompute_visibility', action='store_true', default=False,
                        help="Skip precomputing visibility even for renderers that support it")
    args = parser.parse_args()
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    # Load camera definitions:
    # - Prefer cameras.json if present (Gaussian Splatting format)
    # - Otherwise, use transforms_<split>.json (Blender/NeRF format)
    use_cameras_json = False
    cameras_json_path = None
    camera_config_path = args.config
    if os.path.isdir(camera_config_path):
        # Directory: prefer transforms_<split>.json (e.g., transforms_test.json) if present
        split = args.split
        if split == 'validation':
            split = 'val'
        transforms_path = os.path.join(camera_config_path, f"transforms_{split}.json")
        if os.path.exists(transforms_path):
            camera_config_path = transforms_path
        else:
            # Else fallback to cameras.json (optionally seed from parent)
            possible_cameras = os.path.join(camera_config_path, "cameras.json")
            if not os.path.exists(possible_cameras):
                try:
                    parent_dir = os.path.dirname(os.path.normpath(camera_config_path))
                    parent_cameras = os.path.join(parent_dir, "cameras.json")
                    if os.path.exists(parent_cameras):
                        os.makedirs(camera_config_path, exist_ok=True)
                        shutil.copy2(parent_cameras, possible_cameras)
                        print(f"Copied cameras.json from {parent_cameras} to {possible_cameras}")
                except Exception as e:
                    print(f"Warning: Failed to auto-copy cameras.json: {e}")
            if os.path.exists(possible_cameras):
                use_cameras_json = True
                cameras_json_path = possible_cameras
            else:
                # Neither transforms nor cameras present; let error be raised below
                pass
    else:
        # File path: honor explicit cameras.json if given, else treat as transforms JSON
        if camera_config_path.endswith("cameras.json"):
            if os.path.exists(camera_config_path):
                use_cameras_json = True
                cameras_json_path = camera_config_path
    
    if use_cameras_json:
        with open(cameras_json_path, "r") as f:
            cameras_list = json.load(f)
        if not isinstance(cameras_list, list) or len(cameras_list) == 0:
            raise FileNotFoundError(f"Invalid or empty cameras.json at: {cameras_json_path}")
        # Derive a default H/W from the first camera
        H = cameras_list[0].get("height", 512)
        W = cameras_list[0].get("width", 512)
        num_frames = len(cameras_list)
    else:
        camera_data = load_json_config(camera_config_path)
        if camera_data is None:
            raise FileNotFoundError(f"Camera configuration file not found at: {camera_config_path}")
        # Extract global settings for Blender transforms format
        H = camera_data.get("h", 512)
        W = camera_data.get("w", 512)
        num_frames = len(camera_data.get("frames", []))
    
    # Create a simple scene dict with the model path
    scene_dict = {
        "model": {
            "path": args.ply_path,
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  # Identity transform
        }
    }
    
    # Extract default camera FOVs for transforms.json format
    if not use_cameras_json:
        # Get camera field of view (or use default if not provided)
        fovx_default = camera_data.get("camera_angle_x", 0.6911112070083618)
        fovy_default = focal2fov(fov2focal(fovx_default, W), H)
    
    # load gaussians
    light = EnvLight(path=args.envmap_path, scale=1)
    gaussians_composite = scene_composition(scene_dict, dataset, args.type)

    # Update visibility only for renderers that require precomputed visibility,
    # unless explicitly disabled.
  
    gaussians_composite.update_visibility(args.sample_num)

    # rendering
    capture_dir = args.output
    os.makedirs(capture_dir, exist_ok=True)
    capture_list = [str.strip() for str in args.capture_list.split(",")]
    for capture_type in capture_list:
        capture_type_dir = os.path.join(capture_dir, capture_type)
        os.makedirs(capture_type_dir, exist_ok=True)

    bg = args.background_color
    if bg is None:
        bg = 1 if dataset.white_background else 0
    background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")
    render_fn = render_fn_dict[args.type]

    render_kwargs = {
        "pc": gaussians_composite,
        "pipe": pipe,
        "bg_color": background,
        "is_training": False,
        "dict_params": {
            "env_light": light,
            "sample_num": args.sample_num,
        },
        "bake": args.bake
    }

    # Map sample_num to deferred renderer sampling options
    if args.type == 'neilf_deferred_importance':
        render_kwargs["deferred_options"] = {
            "samples_per_batch": int(args.sample_num),
            "num_batches": 1,
        }

    # Initialize LPIPS model
    loss_fn_alex = None
    if args.gt_path is not None:
        loss_fn_alex = lpips.LPIPS(net='alex').to('cuda')

    # Initialize metrics variables
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    count = 0
    
    # Create metrics file at the beginning to test write permissions
    metrics_file_path = os.path.join(args.output, "metrics.txt")
    try:
        with open(metrics_file_path, 'w') as f:
            f.write(f"Environment Map: {os.path.basename(args.envmap_path)}\n")
            f.write("Processing frames...\n")
        print(f"Initialized metrics file at {metrics_file_path}")
    except Exception as e:
        print(f"Warning: Could not initialize metrics file: {e}")
        metrics_file_path = None

    # Process frames based on camera data
    progress_bar = tqdm(range(num_frames), desc="Rendering", total=num_frames)

    for idx in progress_bar:
        try:
            if use_cameras_json:
                cam_data = cameras_list[idx]
                custom_cam = JSON_to_camera(cam_data)
            else:
                frames = camera_data.get("frames", [])
                frame = frames[idx]
                # Extract camera matrices (Blender-style)
                c2w = np.array(frame.get("transform_matrix", []), dtype=np.float32)
                
                # Blender to OpenGL/computer vision conversion:
                blender_to_cv = np.array([
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],  # Flip Y
                    [0, 0, -1, 0],  # Flip Z
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                # Apply conversion to camera-to-world matrix
                c2w = c2w @ blender_to_cv
                w2c = np.linalg.inv(c2w)
                
                R = w2c[:3, :3].T
                T = w2c[:3, 3]
                
                # Use default FOVs derived from transforms.json
                custom_cam = Camera(colmap_id=0, R=R, T=T,
                                    FoVx=fovx_default, FoVy=fovy_default, fx=None, fy=None, cx=None, cy=None,
                                    image=torch.zeros(3, H, W), image_name=None, uid=0)

            with torch.no_grad():
                render_pkg = render_fn(viewpoint_camera=custom_cam, **render_kwargs)
            
            # Calculate metrics if ground truth path is provided
            if args.gt_path is not None and loss_fn_alex is not None:
                # Try different naming conventions for ground truth images
                possible_gt_paths = [
                    os.path.join(args.gt_path, f"r_{idx}.png"),           # Original pattern r_0.png
                    os.path.join(args.gt_path, f"{idx:03d}.png"),         # 000.png format
                    os.path.join(args.gt_path, f"frame_{idx}.png"),       # frame_0.png
                    os.path.join(args.gt_path, f"image{idx:04d}.png")     # image0000.png
                ]
                
                # Find the first existing GT image from our possible naming patterns
                gt_img_path = None
                for path in possible_gt_paths:
                    if os.path.exists(path):
                        gt_img_path = path
                        break
                
                if gt_img_path is not None:
                    count += 1
                    # Load ground truth image
                    gt_img = cv2.imread(gt_img_path, cv2.IMREAD_UNCHANGED)
                    if gt_img.shape[2] == 4:  # If image has alpha channel
                        alpha_mask = gt_img[:, :, 3] > 0  # Create boolean mask where alpha > 0
                        gt_img_rgb = cv2.cvtColor(gt_img[:, :, :3], cv2.COLOR_BGR2RGB) / 255.0
                    else:
                        # Fallback if no alpha channel
                        alpha_mask = np.ones((gt_img.shape[0], gt_img.shape[1]), dtype=bool)
                        gt_img_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB) / 255.0
                    
                    gt_tensor = torch.from_numpy(gt_img_rgb).permute(2, 0, 1).to('cuda').float()
                    alpha_mask_tensor = torch.from_numpy(alpha_mask).to('cuda')
                    
                    # Select the render type to use for metrics (e.g., "pbr_env")
                    render_type = "pbr_env" if "pbr_env" in capture_list else capture_list[0]
                    rendered_img = render_pkg[render_type]
                    
                    # Convert rendered tensor to numpy
                    rendered_np = rendered_img.detach().cpu().permute(1, 2, 0).numpy()
                    
                    # Create masked comparison with L1 loss visualization
                    # Create directory for side-by-side images if it doesn't exist
                    side_by_side_dir = os.path.join(capture_dir, "side_by_side")
                    os.makedirs(side_by_side_dir, exist_ok=True)
                    
                    # Create masked versions for visualization (showing what's actually used in metrics)
                    masked_gt_vis = gt_img_rgb.copy()
                    masked_rendered_vis = rendered_np.copy()
                    
                    # Completely remove areas outside the mask
                    for c in range(3):
                        masked_gt_vis[~alpha_mask, c] = 0.0  # Black out areas outside mask
                        masked_rendered_vis[~alpha_mask, c] = 0.0  # Black out areas outside mask
                    
                    # Calculate L1 loss (absolute difference)
                    l1_loss_vis = np.abs(masked_rendered_vis - masked_gt_vis)
                    
                    # Take the mean across color channels to get total loss per pixel
                    l1_loss_total = np.mean(l1_loss_vis, axis=2)
                    
                    # Normalize for better visualization
                    if l1_loss_total.max() > 0:
                        l1_loss_total = l1_loss_total / l1_loss_total.max()
                    
                    # Create a 3-channel image where all channels have the same value (grayscale)
                    l1_loss_vis = np.zeros_like(masked_rendered_vis)
                    for c in range(3):
                        l1_loss_vis[:, :, c] = l1_loss_total
                    
                    # Convert to BGR for OpenCV
                    masked_gt_bgr = (masked_gt_vis * 255).astype(np.uint8)
                    masked_gt_bgr = cv2.cvtColor(masked_gt_bgr, cv2.COLOR_RGB2BGR)
                    
                    masked_rendered_bgr = (masked_rendered_vis * 255).astype(np.uint8)
                    masked_rendered_bgr = cv2.cvtColor(masked_rendered_bgr, cv2.COLOR_RGB2BGR)
                    
                    l1_loss_bgr = (l1_loss_vis * 255).astype(np.uint8)
                    l1_loss_bgr = cv2.cvtColor(l1_loss_bgr, cv2.COLOR_RGB2BGR)
                    
                    # Create side-by-side-by-side comparison (three panels)
                    h, w = masked_rendered_bgr.shape[:2]
                    masked_combined = np.zeros((h, w*3, 3), dtype=np.uint8)
                    masked_combined[:, :w] = masked_rendered_bgr
                    masked_combined[:, w:2*w] = masked_gt_bgr
                    masked_combined[:, 2*w:] = l1_loss_bgr
                    
                    # Add labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(masked_combined, "Masked Render", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(masked_combined, "Masked GT", (w + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(masked_combined, "L1 Loss", (2*w + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Save the comparison to side_by_side folder
                    comparison_path = os.path.join(side_by_side_dir, f"comparison_{idx}.png")
                    cv2.imwrite(comparison_path, masked_combined)
                    
                    # PSNR calculation - apply mask
                    rendered_np = rendered_img.detach().cpu().permute(1, 2, 0).numpy()
                    masked_gt = gt_img_rgb.copy()
                    masked_rendered = rendered_np.copy()
                    
                    # Apply mask by only considering pixels where alpha > 0
                    if not np.all(alpha_mask):
                        
                        # SSIM calculation - apply mask
                        # For SSIM, we need to create masked arrays that maintain the original image dimensions
                        masked_gt = gt_img_rgb.copy()
                        masked_rendered= rendered_np.copy()
                        invalid_mask = ~alpha_mask
                        for c in range(3):  # For each color channel
                            masked_gt[invalid_mask, c] = 0
                            masked_rendered[invalid_mask, c] = 0

                        current_psnr = psnr(masked_gt, masked_rendered, data_range=1.0)
                        
                        current_ssim = ssim(masked_gt, masked_rendered, data_range=1.0, channel_axis=2)
                        
                        # LPIPS calculation with masking
                        # Create a 2D mask and expand to match image dimensions
                        mask_3d = alpha_mask_tensor.unsqueeze(0).repeat(3, 1, 1)
                        masked_rendered_tensor = rendered_img * mask_3d
                        masked_gt_tensor = gt_tensor * mask_3d
                        
                        current_lpips = loss_fn_alex(masked_rendered_tensor.unsqueeze(0), masked_gt_tensor.unsqueeze(0)).item()
                    else:
                        # If mask is all True, calculate metrics normally
                        current_psnr = psnr(gt_img_rgb, rendered_np, data_range=1.0)
                        current_ssim = ssim(gt_img_rgb, rendered_np, data_range=1.0, channel_axis=2)
                        current_lpips = loss_fn_alex(rendered_img.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
                    
                    psnr_test += current_psnr
                    ssim_test += current_ssim
                    lpips_test += current_lpips
                    
                    # Update progress bar with current metrics
                    progress_bar.set_postfix(PSNR=current_psnr, SSIM=current_ssim, LPIPS=current_lpips)

                    # Save incremental metrics every 20 frames or when count is a multiple of 20
                    if count > 0 and count % 20 == 0 and metrics_file_path:
                        try:
                            with open(metrics_file_path, 'w') as f:
                                f.write(f"Environment Map: {os.path.basename(args.envmap_path)}\n")
                                f.write(f"Frames processed: {count}\n")
                                f.write(f"Current PSNR: {psnr_test/count:.4f}\n")
                                f.write(f"Current SSIM: {ssim_test/count:.4f}\n")
                                f.write(f"Current LPIPS: {lpips_test/count:.4f}\n")
                        except Exception as e:
                            print(f"Warning: Could not update metrics file: {e}")

            for capture_type in capture_list:
                # Check if the capture_type exists in render_pkg before processing
                if capture_type not in render_pkg and capture_type != "points":
                    print(f"Warning: Capture type '{capture_type}' not found in render output. Skipping.")
                    continue
                    
                if capture_type == "points":
                    render_pkg[capture_type] = render_points(custom_cam, gaussians_composite)
                elif capture_type == "normal":
                    render_pkg[capture_type] = render_pkg[capture_type] * 0.5 + 0.5
                    render_pkg[capture_type] = render_pkg[capture_type] + (1 - render_pkg['opacity']) * bg
                elif capture_type in ["base_color", "roughness", "visibility", "diffuse", "specular"]:
                    render_pkg[capture_type] = render_pkg[capture_type] + (1 - render_pkg['opacity']) * bg
                elif capture_type in ["pbr", "pbr_env", "render"]:
                    render_pkg[capture_type] = render_pkg[capture_type]
                save_image(render_pkg[capture_type], f"{capture_dir}/{capture_type}/frame_{idx}.png")

                if capture_type == "base_color" and args.gt_path is not None:
                    # Get the directory structure from gt_path
                    gt_parent_dir = os.path.dirname(os.path.dirname(args.gt_path) if args.gt_path.endswith('/') else os.path.dirname(args.gt_path))
                    base_color_gt_dir = os.path.join(gt_parent_dir, "base_basecolor")
                    
                    # Use correct naming convention: "r_<number>_basecolor.png"
                    base_color_gt_path = os.path.join(base_color_gt_dir, f"r_{idx}_basecolor.png")
                    
                    if os.path.exists(base_color_gt_path):
                        # Load ground truth base color
                        base_color_gt = cv2.imread(base_color_gt_path, cv2.IMREAD_UNCHANGED)
                        if base_color_gt.shape[2] == 4:  # If image has alpha channel
                            base_alpha_mask = base_color_gt[:, :, 3] > 0
                            base_color_gt_rgb = cv2.cvtColor(base_color_gt[:, :, :3], cv2.COLOR_BGR2RGB) / 255.0
                        else:
                            base_alpha_mask = np.ones((base_color_gt.shape[0], base_color_gt.shape[1]), dtype=bool)
                            base_color_gt_rgb = cv2.cvtColor(base_color_gt, cv2.COLOR_BGR2RGB) / 255.0
                        
                        # Get rendered base color
                        base_color_rendered = render_pkg["base_color"].detach().cpu().permute(1, 2, 0).numpy()
                        
                        # Create masked versions
                        masked_base_gt = base_color_gt_rgb.copy()
                        masked_base_rendered = base_color_rendered.copy()
                        
                        # Apply mask
                        for c in range(3):
                            masked_base_gt[~base_alpha_mask, c] = 0.0
                            masked_base_rendered[~base_alpha_mask, c] = 0.0
                        
                        # Calculate L1 loss for base color
                        base_l1_loss = np.abs(masked_base_rendered - masked_base_gt)
                        
                        # Take the mean across color channels
                        base_l1_total = np.mean(base_l1_loss, axis=2)
                        
                        # Normalize
                        if base_l1_total.max() > 0:
                            base_l1_total = base_l1_total / base_l1_total.max()
                        
                        # Create a 3-channel L1 loss image
                        base_l1_vis = np.zeros_like(masked_base_rendered)
                        for c in range(3):
                            base_l1_vis[:, :, c] = base_l1_total
                        
                        # Convert to BGR for OpenCV
                        base_gt_bgr = (masked_base_gt * 255).astype(np.uint8)
                        base_gt_bgr = cv2.cvtColor(base_gt_bgr, cv2.COLOR_RGB2BGR)
                        
                        base_rendered_bgr = (masked_base_rendered * 255).astype(np.uint8)
                        base_rendered_bgr = cv2.cvtColor(base_rendered_bgr, cv2.COLOR_RGB2BGR)
                        
                        base_l1_bgr = (base_l1_vis * 255).astype(np.uint8)
                        base_l1_bgr = cv2.cvtColor(base_l1_bgr, cv2.COLOR_RGB2BGR)
                        
                        # Create side-by-side-by-side comparison for base color
                        h, w = base_rendered_bgr.shape[:2]
                        base_combined = np.zeros((h, w*3, 3), dtype=np.uint8)
                        base_combined[:, :w] = base_rendered_bgr
                        base_combined[:, w:2*w] = base_gt_bgr
                        base_combined[:, 2*w:] = base_l1_bgr
                        
                        # Add labels
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(base_combined, "Base Color", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(base_combined, "GT Base Color", (w + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(base_combined, "L1 Loss", (2*w + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Save the comparison directly to the base_color folder
                        # Override the standard base_color image with the comparison
                        comparison_path = os.path.join(capture_dir, "base_color", f"frame_{idx}.png")
                        cv2.imwrite(comparison_path, base_combined)
                        
                        # Skip the normal save_image call since we've already saved the comparison
                        continue

                # Add roughness comparison with ground truth
                elif capture_type == "roughness" and args.gt_path is not None:
                    gt_parent_dir = os.path.dirname(os.path.dirname(args.gt_path) if args.gt_path.endswith('/') else os.path.dirname(args.gt_path))
                    roughness_gt_dir = os.path.join(gt_parent_dir, "base_roughness")
                    
                    # Use correct naming convention: "r_<number>_roughness.png"
                    roughness_gt_path = os.path.join(roughness_gt_dir, f"r_{idx}_roughness.png")
                    
                    if os.path.exists(roughness_gt_path):
                        # Load ground truth roughness
                        roughness_gt = cv2.imread(roughness_gt_path, cv2.IMREAD_UNCHANGED)
                        if roughness_gt.shape[2] == 4:  # If image has alpha channel
                            roughness_alpha_mask = roughness_gt[:, :, 3] > 0
                            # For roughness, just use the first channel as it's grayscale
                            roughness_gt_val = cv2.cvtColor(roughness_gt[:, :, :3], cv2.COLOR_BGR2GRAY) / 255.0
                            # Expand to 3 channels for visualization
                            roughness_gt_rgb = np.stack([roughness_gt_val] * 3, axis=2)
                        else:
                            roughness_alpha_mask = np.ones((roughness_gt.shape[0], roughness_gt.shape[1]), dtype=bool)
                            roughness_gt_val = cv2.cvtColor(roughness_gt, cv2.COLOR_BGR2GRAY) / 255.0
                            roughness_gt_rgb = np.stack([roughness_gt_val] * 3, axis=2)
                        
                        # Get rendered roughness and expand to 3 channels for visualization
                        roughness_rendered = render_pkg["roughness"].detach().cpu().numpy()
                        if roughness_rendered.shape[0] == 1:  # If single channel
                            roughness_rendered = roughness_rendered[0]
                            roughness_rendered_rgb = np.stack([roughness_rendered] * 3, axis=2)
                        else:  # If already multi-channel
                            roughness_rendered_rgb = roughness_rendered.transpose(1, 2, 0)
                            if roughness_rendered_rgb.shape[2] == 1:
                                roughness_rendered_rgb = np.concatenate([roughness_rendered_rgb] * 3, axis=2)
                        
                        # Create masked versions
                        masked_roughness_gt = roughness_gt_rgb.copy()
                        masked_roughness_rendered = roughness_rendered_rgb.copy()
                        
                        # Apply mask
                        for c in range(3):
                            masked_roughness_gt[~roughness_alpha_mask, c] = 0.0
                            masked_roughness_rendered[~roughness_alpha_mask, c] = 0.0
                        
                        # Calculate L1 loss for roughness
                        roughness_l1_loss = np.abs(masked_roughness_rendered - masked_roughness_gt)
                        
                        # Take the mean across channels
                        roughness_l1_total = np.mean(roughness_l1_loss, axis=2)
                        
                        # Normalize
                        if roughness_l1_total.max() > 0:
                            roughness_l1_total = roughness_l1_total / roughness_l1_total.max()
                        
                        # Create a 3-channel L1 loss image
                        roughness_l1_vis = np.zeros_like(masked_roughness_rendered)
                        for c in range(3):
                            roughness_l1_vis[:, :, c] = roughness_l1_total
                        
                        # Convert to BGR for OpenCV
                        roughness_gt_bgr = (masked_roughness_gt * 255).astype(np.uint8)
                        roughness_gt_bgr = cv2.cvtColor(roughness_gt_bgr, cv2.COLOR_RGB2BGR)
                        
                        roughness_rendered_bgr = (masked_roughness_rendered * 255).astype(np.uint8)
                        roughness_rendered_bgr = cv2.cvtColor(roughness_rendered_bgr, cv2.COLOR_RGB2BGR)
                        
                        roughness_l1_bgr = (roughness_l1_vis * 255).astype(np.uint8)
                        roughness_l1_bgr = cv2.cvtColor(roughness_l1_bgr, cv2.COLOR_RGB2BGR)
                        
                        # Create side-by-side-by-side comparison for roughness
                        h, w = roughness_rendered_bgr.shape[:2]
                        roughness_combined = np.zeros((h, w*3, 3), dtype=np.uint8)
                        roughness_combined[:, :w] = roughness_rendered_bgr
                        roughness_combined[:, w:2*w] = roughness_gt_bgr
                        roughness_combined[:, 2*w:] = roughness_l1_bgr
                        
                        # Add labels
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(roughness_combined, "Roughness", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(roughness_combined, "GT Roughness", (w + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(roughness_combined, "L1 Loss", (2*w + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Save the comparison
                        comparison_path = os.path.join(capture_dir, "roughness", f"frame_{idx}.png")
                        cv2.imwrite(comparison_path, roughness_combined)
                        continue

                # Add irradiance comparison with ground truth 
                elif capture_type == "irradiance" and args.gt_path is not None:
                    gt_parent_dir = os.path.dirname(os.path.dirname(args.gt_path) if args.gt_path.endswith('/') else os.path.dirname(args.gt_path))
                    irradiance_gt_dir = os.path.join(gt_parent_dir, "base_irradiance")
                    
                    # Use correct naming convention: "r_<number>_irradiance.png"
                    irradiance_gt_path = os.path.join(irradiance_gt_dir, f"r_{idx}_irradiance.png")
                    
                    if os.path.exists(irradiance_gt_path):
                        # Load ground truth irradiance
                        irradiance_gt = cv2.imread(irradiance_gt_path, cv2.IMREAD_UNCHANGED)
                        if irradiance_gt.shape[2] == 4:  # If image has alpha channel
                            irradiance_alpha_mask = irradiance_gt[:, :, 3] > 0
                            irradiance_gt_rgb = cv2.cvtColor(irradiance_gt[:, :, :3], cv2.COLOR_BGR2RGB) / 255.0
                        else:
                            irradiance_alpha_mask = np.ones((irradiance_gt.shape[0], irradiance_gt.shape[1]), dtype=bool)
                            irradiance_gt_rgb = cv2.cvtColor(irradiance_gt, cv2.COLOR_BGR2RGB) / 255.0
                        
                        # Get rendered irradiance
                        irradiance_rendered = render_pkg["irradiance"].detach().cpu().permute(1, 2, 0).numpy()
                        
                        # Create masked versions
                        masked_irradiance_gt = irradiance_gt_rgb.copy()
                        masked_irradiance_rendered = irradiance_rendered.copy()
                        
                        # Apply mask
                        for c in range(3):
                            masked_irradiance_gt[~irradiance_alpha_mask, c] = 0.0
                            masked_irradiance_rendered[~irradiance_alpha_mask, c] = 0.0
                        
                        # Calculate L1 loss for irradiance
                        irradiance_l1_loss = np.abs(masked_irradiance_rendered - masked_irradiance_gt)
                        
                        # Take the mean across color channels
                        irradiance_l1_total = np.mean(irradiance_l1_loss, axis=2)
                        
                        # Normalize
                        if irradiance_l1_total.max() > 0:
                            irradiance_l1_total = irradiance_l1_total / irradiance_l1_total.max()
                        
                        # Create a 3-channel L1 loss image
                        irradiance_l1_vis = np.zeros_like(masked_irradiance_rendered)
                        for c in range(3):
                            irradiance_l1_vis[:, :, c] = irradiance_l1_total
                        
                        # Convert to BGR for OpenCV
                        irradiance_gt_bgr = (masked_irradiance_gt * 255).astype(np.uint8)
                        irradiance_gt_bgr = cv2.cvtColor(irradiance_gt_bgr, cv2.COLOR_RGB2BGR)
                        
                        irradiance_rendered_bgr = (masked_irradiance_rendered * 255).astype(np.uint8)
                        irradiance_rendered_bgr = cv2.cvtColor(irradiance_rendered_bgr, cv2.COLOR_RGB2BGR)
                        
                        irradiance_l1_bgr = (irradiance_l1_vis * 255).astype(np.uint8)
                        irradiance_l1_bgr = cv2.cvtColor(irradiance_l1_bgr, cv2.COLOR_RGB2BGR)
                        
                        # Create side-by-side-by-side comparison for irradiance
                        h, w = irradiance_rendered_bgr.shape[:2]
                        irradiance_combined = np.zeros((h, w*3, 3), dtype=np.uint8)
                        irradiance_combined[:, :w] = irradiance_rendered_bgr
                        irradiance_combined[:, w:2*w] = irradiance_gt_bgr
                        irradiance_combined[:, 2*w:] = irradiance_l1_bgr
                        
                        # Add labels
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(irradiance_combined, "Irradiance", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(irradiance_combined, "GT Irradiance", (w + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(irradiance_combined, "L1 Loss", (2*w + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Save the comparison
                        comparison_path = os.path.join(capture_dir, "irradiance", f"frame_{idx}.png")
                        cv2.imwrite(comparison_path, irradiance_combined)
                        continue

                elif capture_type == "metallic":
                    # Blend with background if needed (single channel, expand to 3 for saving)
                    metallic_map = render_pkg["metallic"]
                    if metallic_map.shape[0] == 1:
                        metallic_map = metallic_map.repeat(3, 1, 1)
                    elif metallic_map.shape[0] != 3:
                        # Handle case where metallic might have unexpected dimensions
                        print(f"Warning: Unexpected metallic map shape: {metallic_map.shape}")
                        # Try to reshape or skip this frame
                        continue
                    metallic_map = metallic_map + (1 - render_pkg['opacity']) * bg
                    save_image(metallic_map, f"{capture_dir}/metallic/frame_{idx}.png")
                    continue

        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
            import traceback
            print(f"Full traceback for frame {idx}:")
            traceback.print_exc()
            # Continue with the next frame instead of crashing

    # Final metrics output - keep this simple
    print("count: ", count, " args.gt_path: ", args.gt_path)
    if count > 0 and args.gt_path is not None:
        avg_psnr = psnr_test / count
        avg_ssim = ssim_test / count
        avg_lpips = lpips_test / count
        
        print(f"Average PSNR: {avg_psnr:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average LPIPS: {avg_lpips:.4f}")
        
        # Write final metrics to file
        if metrics_file_path:
            try:
                with open(metrics_file_path, 'w') as f:
                    f.write(f"Environment Map: {os.path.basename(args.envmap_path)}\n")
                    f.write(f"Total frames processed: {count}\n")
                    f.write(f"Average PSNR: {avg_psnr:.4f}\n")
                    f.write(f"Average SSIM: {avg_ssim:.4f}\n")
                    f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
                print(f"Final metrics saved to {metrics_file_path}")
            except Exception as e:
                print(f"Failed to write final metrics: {e}")

    # output as video
    if args.video:
        # Add side_by_side to the list of capture types for video creation
        video_capture_list = capture_list.copy()
        video_capture_list.append("side_by_side")
        # No need to add base_color_comparison anymore
        
        progress_bar = tqdm(video_capture_list, desc="Outputting video")
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        for capture_type in progress_bar:
            video_path = f"{capture_dir}/{capture_type}.mp4"
            
            # Handle different file naming patterns
            if capture_type == "side_by_side":
                image_names = [os.path.join(capture_dir, capture_type, f"comparison_{j}.png") for j in
                              range(num_frames)]
            else:
                image_names = [os.path.join(capture_dir, capture_type, f"frame_{j}.png") for j in
                              range(num_frames)]
            
            # Find first existing image to get dimensions
            first_img = None
            for img_path in image_names:
                if os.path.exists(img_path):
                    first_img = cv2.imread(img_path)
                    if first_img is not None:
                        break
            
            if first_img is None:
                print(f"Warning: No valid images found for {capture_type}, skipping video creation.")
                continue
            
            # Get dimensions from the first image
            height, width = first_img.shape[:2]
            print(f"Creating video for {capture_type} with dimensions {width}x{height}")
            
            # Create video with dimensions from actual image
            media_writer = cv2.VideoWriter(video_path, fourcc, 24, (width, height))
            
            frame_count = 0
            for image_name in image_names:
                if os.path.exists(image_name):
                    img = cv2.imread(image_name)
                    if img is not None:
                        media_writer.write(img)
                        frame_count += 1
            
            media_writer.release()
            print(f"Video created for {capture_type} with {frame_count} frames")