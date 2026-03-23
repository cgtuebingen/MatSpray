import math
import torch
import numpy as np
import torch.nn.functional as F
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim, bilateral_smooth_loss, second_order_edge_aware_loss, tv_loss, first_order_edge_aware_loss, first_order_loss, first_order_edge_aware_norm_loss
from utils.image_utils import psnr
from utils.graphics_utils import fibonacci_sphere_sampling, rgb_to_srgb, srgb_to_rgb
from gs_sss_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render_view(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None, opt_params=None):
    direct_light_env_light = dict_params.get("env_light")
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    intrinsic = viewpoint_camera.intrinsics

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.compute_SHs_python:
            dir_pp_normalized = F.normalize(viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D,
                                            dim=-1)
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_shs
    else:
        colors_precomp = override_color

    base_color = pc.get_base_color
    roughness = pc.get_roughness
    metallic = pc.get_metallic
    
    # Get camera direction for normal MLP
    camera_direction = -viewpoint_camera.world_view_transform[:3, 2]  # Forward direction
    camera_direction = F.normalize(camera_direction, dim=-1)
    
    normal = pc.get_normal(camera_direction)
    incidents = pc.get_incidents  # incident shs
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_shs.shape[0], 1))
    dir_pp_normalized = F.normalize(dir_pp, dim=-1)
    
    if is_training:
        brdf_color, extra_results = rendering_equation(
            base_color, roughness, metallic, normal, viewdirs, incidents,
            direct_light_env_light, visibility_precompute=pc._visibility_tracing, 
            incident_dirs_precompute=pc._incident_dirs, incident_areas_precompute=pc._incident_areas,
            opt_params=opt_params)
    else:
        chunk_size = 100000
        brdf_color = []
        extra_results = []
        for i in range(0, means3D.shape[0], chunk_size):
            _brdf_color, _extra_results = rendering_equation(
                base_color[i:i + chunk_size], roughness[i:i + chunk_size], metallic[i:i + chunk_size], 
                normal[i:i + chunk_size].detach(), viewdirs[i:i + chunk_size], incidents[i:i + chunk_size],
                direct_light_env_light, 
                visibility_precompute=pc._visibility_tracing[i:i + chunk_size], 
                incident_dirs_precompute=pc._incident_dirs[i:i + chunk_size], 
                incident_areas_precompute=pc._incident_areas[i:i + chunk_size],
                opt_params=opt_params)
            brdf_color.append(_brdf_color)
            extra_results.append(_extra_results)
        brdf_color = torch.cat(brdf_color, dim=0)
        extra_results = {k: torch.cat([_extra_results[k] for _extra_results in extra_results], dim=0) for k in extra_results[0]}
        torch.cuda.empty_cache()

    xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
    depths = (xyz_homo @ viewpoint_camera.world_view_transform)[:, 2:3]
    depths2 = depths.square()
    
    if is_training:
        features = torch.cat([depths, depths2, brdf_color, normal, base_color, roughness, metallic, 
                              extra_results["diffuse_light"], 
                              extra_results["incident_visibility"].mean(-2)], dim=-1)
    else:
        features = torch.cat([depths, depths2, brdf_color, normal, base_color, roughness, metallic,
                              extra_results["diffuse_light"], 
                              extra_results["specular"], 
                              extra_results["incident_lights"].mean(-2),
                              extra_results["local_incident_lights"].mean(-2),
                              extra_results["global_incident_lights"].mean(-2),
                              extra_results["incident_visibility"].mean(-2)], dim=-1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
    )

    mask = num_contrib > 0
    rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5) * mask
    feature_dict = {}
    if is_training:
        rendered_depth, rendered_depth2, rendered_pbr, rendered_normal, rendered_base_color, \
            rendered_roughness, rendered_metallic, rendered_diffuse, rendered_visibility \
            = rendered_feature.split([1, 1, 3, 3, 3, 1, 1, 3, 1], dim=0)
        feature_dict.update({"base_color": rgb_to_srgb(rendered_base_color),
                             "roughness": rendered_roughness,
                             "metallic": rendered_metallic,
                             "diffuse": rgb_to_srgb(rendered_diffuse),
                             "visibility": rendered_visibility
                             })
    else:
        rendered_depth, rendered_depth2, rendered_pbr, rendered_normal, rendered_base_color, rendered_roughness, \
            rendered_metallic, rendered_diffuse, rendered_specular, rendered_light, rendered_local_light, rendered_global_light, rendered_visibility \
            = rendered_feature.split([1, 1, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 1], dim=0)
        feature_dict.update({
                             "base_color": rgb_to_srgb(rendered_base_color),
                             "roughness": rendered_roughness,
                             "metallic": rendered_metallic,
                             "diffuse": rgb_to_srgb(rendered_diffuse),
                             "specular": rgb_to_srgb(rendered_specular),
                             "lights": rgb_to_srgb(rendered_light),
                             "local_lights": rgb_to_srgb(rendered_local_light),
                             "global_lights": rgb_to_srgb(rendered_global_light),
                             "visibility": rendered_visibility,
                             })
    rendered_var = rendered_depth2 - rendered_depth.square()

    pbr = rendered_pbr
    rendered_pbr = pbr * rendered_opacity + (1 - rendered_opacity) * bg_color[:, None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results = {"render": rendered_image,
               "depth": rendered_depth,
               "depth_var": rendered_var,
               "pbr": rgb_to_srgb(rendered_pbr),
               "normal": rendered_normal,
               "pseudo_normal": rendered_pseudo_normal,
               "surface_xyz": rendered_surface_xyz,
               "opacity": rendered_opacity,
               "depth": rendered_depth,
               "viewspace_points": screenspace_points,
               "visibility_filter": radii > 0,
               "radii": radii,
               "num_rendered": num_rendered,
               "num_contrib": num_contrib
               }

    results.update(feature_dict)
    results["diffuse_light"] = extra_results["diffuse_light"]
    try:
        results["env"] = direct_light_env_light.get_env
    except:
        pass
    
    if not is_training:
        directions = viewpoint_camera.get_world_directions()
        direct_env = direct_light_env_light.direct_light(directions.permute(1, 2, 0)).permute(2, 0, 1)
        results["render_env"] = rendered_image + (1 - rendered_opacity) * rgb_to_srgb(direct_env)
        results["pbr_env"] = rgb_to_srgb(pbr * rendered_opacity + (1 - rendered_opacity) * direct_env)
        results["env_only"] = rgb_to_srgb(direct_env)
        
    return results

def render_view_custom(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, custom_base_color: torch.Tensor,
                       custom_roughness: torch.Tensor, custom_metallic: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None, opt_params=None):
    direct_light_env_light = dict_params.get("env_light")
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    intrinsic = viewpoint_camera.intrinsics

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.compute_SHs_python:
            dir_pp_normalized = F.normalize(viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D,
                                            dim=-1)
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_shs
    else:
        colors_precomp = override_color

    base_color = custom_base_color
    roughness = custom_roughness
    metallic = custom_metallic
    
    # Get camera direction for normal MLP
    camera_direction = -viewpoint_camera.world_view_transform[:3, 2]  # Forward direction
    camera_direction = F.normalize(camera_direction, dim=-1)
    
    normal = pc.get_normal(camera_direction)
    incidents = pc.get_incidents  # incident shs
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_shs.shape[0], 1))
    dir_pp_normalized = F.normalize(dir_pp, dim=-1)
    
    if is_training:
        brdf_color, extra_results = rendering_equation(
            base_color, roughness, metallic, normal, viewdirs, incidents,
            direct_light_env_light, visibility_precompute=pc._visibility_tracing, 
            incident_dirs_precompute=pc._incident_dirs, incident_areas_precompute=pc._incident_areas,
            opt_params=opt_params)
    else:
        chunk_size = 100000
        brdf_color = []
        extra_results = []
        for i in range(0, means3D.shape[0], chunk_size):
            _brdf_color, _extra_results = rendering_equation(
                base_color[i:i + chunk_size], roughness[i:i + chunk_size], metallic[i:i + chunk_size], 
                normal[i:i + chunk_size].detach(), viewdirs[i:i + chunk_size], incidents[i:i + chunk_size],
                direct_light_env_light, 
                visibility_precompute=pc._visibility_tracing[i:i + chunk_size], 
                incident_dirs_precompute=pc._incident_dirs[i:i + chunk_size], 
                incident_areas_precompute=pc._incident_areas[i:i + chunk_size],
                opt_params=opt_params)
            brdf_color.append(_brdf_color)
            extra_results.append(_extra_results)
        brdf_color = torch.cat(brdf_color, dim=0)
        extra_results = {k: torch.cat([_extra_results[k] for _extra_results in extra_results], dim=0) for k in extra_results[0]}
        torch.cuda.empty_cache()

    xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
    depths = (xyz_homo @ viewpoint_camera.world_view_transform)[:, 2:3]
    depths2 = depths.square()
    
    if is_training:
        features = torch.cat([depths, depths2, brdf_color, normal, base_color, roughness, metallic, 
                              extra_results["diffuse_light"], 
                              extra_results["incident_visibility"].mean(-2)], dim=-1)
    else:
        features = torch.cat([depths, depths2, brdf_color, normal, base_color, roughness, metallic,
                              extra_results["diffuse_light"], 
                              extra_results["specular"], 
                              extra_results["incident_lights"].mean(-2),
                              extra_results["local_incident_lights"].mean(-2),
                              extra_results["global_incident_lights"].mean(-2),
                              extra_results["incident_visibility"].mean(-2)], dim=-1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
    )

    mask = num_contrib > 0
    rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5) * mask
    feature_dict = {}
    if is_training:
        rendered_depth, rendered_depth2, rendered_pbr, rendered_normal, rendered_base_color, \
            rendered_roughness, rendered_metallic, rendered_diffuse, rendered_visibility \
            = rendered_feature.split([1, 1, 3, 3, 3, 1, 1, 3, 1], dim=0)
        feature_dict.update({"base_color": rgb_to_srgb(rendered_base_color),
                             "roughness": rendered_roughness,
                             "metallic": rendered_metallic,
                             "diffuse": rgb_to_srgb(rendered_diffuse),
                             "visibility": rendered_visibility
                             })
    else:
        rendered_depth, rendered_depth2, rendered_pbr, rendered_normal, rendered_base_color, rendered_roughness, \
            rendered_metallic, rendered_diffuse, rendered_specular, rendered_light, rendered_local_light, rendered_global_light, rendered_visibility \
            = rendered_feature.split([1, 1, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 1], dim=0)
        feature_dict.update({
                             "base_color": rgb_to_srgb(rendered_base_color),
                             "roughness": rendered_roughness,
                             "metallic": rendered_metallic,
                             "diffuse": rgb_to_srgb(rendered_diffuse),
                             "specular": rgb_to_srgb(rendered_specular),
                             "lights": rgb_to_srgb(rendered_light),
                             "local_lights": rgb_to_srgb(rendered_local_light),
                             "global_lights": rgb_to_srgb(rendered_global_light),
                             "visibility": rendered_visibility,
                             })
    rendered_var = rendered_depth2 - rendered_depth.square()

    pbr = rendered_pbr
    rendered_pbr = pbr * rendered_opacity + (1 - rendered_opacity) * bg_color[:, None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results = {"render": rendered_image,
               "depth": rendered_depth,
               "depth_var": rendered_var,
               "pbr": rgb_to_srgb(rendered_pbr),
               "normal": rendered_normal,
               "pseudo_normal": rendered_pseudo_normal,
               "surface_xyz": rendered_surface_xyz,
               "opacity": rendered_opacity,
               "depth": rendered_depth,
               "viewspace_points": screenspace_points,
               "visibility_filter": radii > 0,
               "radii": radii,
               "num_rendered": num_rendered,
               "num_contrib": num_contrib
               }

    results.update(feature_dict)
    results["diffuse_light"] = extra_results["diffuse_light"]
    try:
        results["env"] = direct_light_env_light.get_env
    except:
        pass
    
    if not is_training:
        directions = viewpoint_camera.get_world_directions()
        direct_env = direct_light_env_light.direct_light(directions.permute(1, 2, 0)).permute(2, 0, 1)
        results["render_env"] = rendered_image + (1 - rendered_opacity) * rgb_to_srgb(direct_env)
        results["pbr_env"] = rgb_to_srgb(pbr * rendered_opacity + (1 - rendered_opacity) * direct_env)
        results["env_only"] = rgb_to_srgb(direct_env)
        
    return results


def calculate_loss(viewpoint_camera, pc, results, opt, direct_light_env_light):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    rendered_image = results["render"]
    rendered_depth = results["depth"]
    rendered_normal = results["normal"]
    rendered_pbr = results["pbr"]
    rendered_opacity = results["opacity"]
    rendered_base_color = results["base_color"]
    rendered_roughness = results["roughness"]
    rendered_diffuse = results["diffuse"]

    # Ensure GT is available for cameras loaded from cameras.json (no embedded original_image)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if (not hasattr(viewpoint_camera, 'original_image')) or (viewpoint_camera.original_image is None):
        try:
            from scene.utils import load_img_rgb
            assert hasattr(viewpoint_camera, 'image_path') and viewpoint_camera.image_path is not None, "Missing image_path; cannot lazy-load GT"
            img_np = load_img_rgb(viewpoint_camera.image_path)
            alpha_t = None
            if img_np.ndim == 3 and img_np.shape[2] >= 4:
                alpha_np = img_np[..., 3]
                img_np = img_np[..., :3]
                alpha_t = torch.from_numpy(alpha_np).float().unsqueeze(0).clamp(0.0, 1.0).to(device)
            gt_img_t = torch.from_numpy(img_np).float().permute(2, 0, 1).clamp(0.0, 1.0).to(device)
            viewpoint_camera.original_image = gt_img_t
            # Always prefer freshly loaded alpha as mask if available; otherwise fallback to ones
            if alpha_t is not None:
                viewpoint_camera.image_mask = alpha_t
            else:
                viewpoint_camera.image_mask = torch.ones((1, gt_img_t.shape[-2], gt_img_t.shape[-1]), device=device, dtype=torch.float32)
        except Exception as e:
            print(f"Warning: Failed to lazy-load image for camera {viewpoint_camera.image_name}: {e}")
            # Create a dummy image if loading fails
            if not hasattr(viewpoint_camera, 'image_width') or not hasattr(viewpoint_camera, 'image_height'):
                raise AttributeError(f"Camera {viewpoint_camera.image_name} has no original_image and cannot load from image_path")
            viewpoint_camera.original_image = torch.zeros((3, viewpoint_camera.image_height, viewpoint_camera.image_width), device=device, dtype=torch.float32)
            viewpoint_camera.image_mask = torch.ones((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device=device, dtype=torch.float32)

    # Prepare ground-truth image and mask
    if hasattr(viewpoint_camera, 'original_image') and viewpoint_camera.original_image is not None:
        gt_image_full = viewpoint_camera.original_image.to(device)
        
        # Extract mask from alpha channel of ground truth image
        # If GT image has 4 channels (RGBA), extract alpha as mask and RGB as image
        if gt_image_full.dim() == 3 and gt_image_full.shape[0] == 4:
            # Extract RGB channels
            gt_image = gt_image_full[:3, ...]
            # Extract alpha channel as mask
            image_mask = gt_image_full[3:4, ...].clamp(0.0, 1.0)
        else:
            # If no alpha channel, check if image_mask was set separately (e.g., in train.py)
            gt_image = gt_image_full
            if gt_image.dim() == 3 and gt_image.shape[0] > 3:
                gt_image = gt_image[:3, ...]
            # Prefer image_mask attribute if available, otherwise use all ones
            if hasattr(viewpoint_camera, 'image_mask') and viewpoint_camera.image_mask is not None:
                image_mask = viewpoint_camera.image_mask.to(device)
                if image_mask.dim() == 2:
                    image_mask = image_mask.unsqueeze(0)
            else:
                image_mask = torch.ones((1, gt_image.shape[1], gt_image.shape[2]), device=device, dtype=gt_image.dtype)
    else:
        # Fallback: black image (should rarely happen)
        H = rendered_image.shape[1]
        W = rendered_image.shape[2]
        gt_image = torch.zeros((3, H, W), device=device, dtype=rendered_image.dtype)
        # Try to get image_mask from camera, otherwise use all ones
        if hasattr(viewpoint_camera, 'image_mask') and viewpoint_camera.image_mask is not None:
            image_mask = viewpoint_camera.image_mask.to(device)
            if image_mask.dim() == 2:
                image_mask = image_mask.unsqueeze(0)
        else:
            image_mask = torch.ones((1, H, W), device=device, dtype=gt_image.dtype)
    
    # Ensure image_mask is in correct format
    if image_mask.dim() == 2:
        image_mask = image_mask.unsqueeze(0)
    
    # Apply mask to RGB channels of ground truth
    image_mask_3c = image_mask.repeat(3, 1, 1) if image_mask.shape[0] == 1 else image_mask
    gt_image = gt_image * image_mask_3c
    
    Ll1 = F.l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    tb_dict["l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

    Ll1_pbr = F.l1_loss(rendered_pbr, gt_image)
    ssim_val_pbr = ssim(rendered_pbr, gt_image)
    tb_dict["l1_pbr"] = Ll1_pbr.item()
    tb_dict["ssim_pbr"] = ssim_val_pbr.item()
    tb_dict["psnr_pbr"] = psnr(rendered_pbr, gt_image).mean().item()
    loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (1.0 - ssim_val_pbr)
    loss = loss + opt.lambda_pbr * loss_pbr

    if opt.lambda_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        image_mask = viewpoint_camera.image_mask.cuda().bool()
        depth_mask = gt_depth > 0
        sur_mask = torch.logical_xor(image_mask, depth_mask)

        loss_depth = F.l1_loss(rendered_depth[~sur_mask], gt_depth[~sur_mask])
        tb_dict["loss_depth"] = loss_depth.item()
        loss = loss + opt.lambda_depth * loss_depth

    if opt.lambda_mask_entropy > 0:
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_mask_entropy = -(image_mask * torch.log(o) + (1 - image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

    if opt.lambda_normal_render_depth > 0:
        normal_pseudo = results['pseudo_normal']
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_normal_render_depth = F.mse_loss(
            rendered_normal * image_mask, normal_pseudo.detach() * image_mask)
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth

    if opt.lambda_normal_mvs_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        depth_mask = (gt_depth > 0).float()
        mvs_normal = viewpoint_camera.normal.cuda()
        loss_normal_mvs_depth = F.mse_loss(
            rendered_normal * depth_mask, mvs_normal * depth_mask)
        tb_dict["loss_normal_mvs_depth"] = loss_normal_mvs_depth.item()
        loss = loss + opt.lambda_normal_mvs_depth * loss_normal_mvs_depth

    if opt.lambda_light > 0:
        diffuse_light = results["diffuse_light"]
        mean_light = diffuse_light.mean(-1, keepdim=True).expand_as(diffuse_light)
        loss_light = F.l1_loss(diffuse_light, mean_light)
        tb_dict["loss_light"] = loss_light.item()
        loss = loss + opt.lambda_light * loss_light

    if opt.lambda_base_color_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
        # loss_base_color_smooth = second_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
        tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
        loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth

    if opt.lambda_roughness_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
        # loss_roughness_smooth = second_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
        tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
        loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth
    
    
    if opt.lambda_light_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_light_smooth = first_order_edge_aware_loss(rendered_diffuse * image_mask, rendered_normal)
        # loss_light_smooth = second_order_edge_aware_loss(rendered_diffuse * image_mask, gt_image)
        tb_dict["loss_light_smooth"] = loss_light_smooth.item()
        loss = loss + opt.lambda_light_smooth * loss_light_smooth
        
    if opt.lambda_env_smooth > 0:
        env = direct_light_env_light.get_env
        loss_env_smooth = tv_loss(env[0].permute(2, 0, 1))
        tb_dict["loss_env_smooth"] = loss_env_smooth.item()
        loss = loss + opt.lambda_env_smooth * loss_env_smooth
    
    if opt.lambda_normal_smooth > 0:
        # loss_normal_smooth = second_order_edge_aware_loss(rendered_normal * image_mask, gt_image)
        loss_normal_smooth = tv_loss(rendered_normal * image_mask)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        loss = loss + opt.lambda_normal_smooth * loss_normal_smooth
    
    tb_dict["loss"] = loss.item()

    return loss, tb_dict


def render_neilf(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                 scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
                 is_training=False, dict_params=None, **kwargs):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(viewpoint_camera, pc, pipe, bg_color,
                          scaling_modifier, override_color, is_training, dict_params, opt_params=opt)

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt, direct_light_env_light=dict_params['env_light'])
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    return results


def rendering_equation(base_color, roughness, metallic, normals, viewdirs,
                      incidents, direct_light_env_light=None,
                      visibility_precompute=None, incident_dirs_precompute=None, incident_areas_precompute=None,
                      opt_params=None):
    incident_dirs, incident_areas = incident_dirs_precompute, incident_areas_precompute

    incident_dirs = F.normalize(incident_dirs, dim=-1)
    normals = F.normalize(normals, dim=-1)
    viewdirs = F.normalize(viewdirs, dim=-1)

    deg = int(np.sqrt(incidents.shape[1]) - 1)
    global_incident_lights = direct_light_env_light.direct_light(incident_dirs)
    local_incident_lights = eval_sh(deg, incidents.transpose(1, 2).view(-1, 1, 3, (deg + 1) ** 2), incident_dirs).clamp_min(0)
    
    incident_visibility = visibility_precompute
    global_incident_lights = global_incident_lights * incident_visibility
    incident_lights = local_incident_lights + global_incident_lights

    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    
    # Use the actual metallic parameter from the gaussians
    metalness = metallic
    
    # Process base_color, roughness, and metallic based on detach flags
    detach_base_color = opt_params.detach_base_color if hasattr(opt_params, 'detach_base_color') else False
    detach_roughness = opt_params.detach_roughness if hasattr(opt_params, 'detach_roughness') else False
    detach_metallic = opt_params.detach_metallic if hasattr(opt_params, 'detach_metallic') else False
    
    processed_base_color = base_color.detach() if detach_base_color else base_color
    processed_roughness = roughness.detach() if detach_roughness else roughness
    processed_metallic = metalness.detach() if detach_metallic else metalness
    
    f_d, f_s = compute_pbr_brdf(processed_base_color, processed_metallic, processed_roughness, normals, viewdirs, incident_dirs)

    transport = incident_lights * incident_areas * n_d_i  # (num_pts, num_sample, 3)
    specular = ((f_s) * transport).mean(dim=-2)
    # pbr = ((f_d + f_s) * transport).mean(dim=-2)
    pbr = ((f_d + f_s) * transport).mean(dim=-2)
    pbr = torch.clamp(pbr, 0.0, 1.0)
    diffuse_light = transport.mean(dim=-2)



    extra_results = {
        "incident_dirs": incident_dirs,
        "incident_lights": incident_lights,
        "local_incident_lights": local_incident_lights,
        "global_incident_lights": global_incident_lights,
        "incident_visibility": incident_visibility,
        "diffuse_light": diffuse_light,
        "specular": specular,
    }

    return pbr, extra_results


def compute_pbr_brdf(base_color, metalness, roughness, normals, viewdirs, incident_dirs):
    """
    Compute PBR BRDF terms using corrected formulation
    """
    

    
    # Compute halfway vector and dot products
    half_vector = F.normalize(incident_dirs + viewdirs.unsqueeze(1), dim=-1)
    dot_nv = torch.clamp(torch.sum(normals * viewdirs, dim=-1, keepdim=True), 0.0, 1.0)
    dot_nl = torch.clamp(torch.sum(normals.unsqueeze(1) * incident_dirs, dim=-1, keepdim=True), 0.0, 1.0)
    dot_nh = torch.clamp(torch.sum(normals.unsqueeze(1) * half_vector, dim=-1, keepdim=True), 0.0, 1.0)
    dot_vh = torch.clamp(torch.sum(viewdirs.unsqueeze(1) * half_vector, dim=-1, keepdim=True), 0.0, 1.0)

    # Expand material properties to match the 64 samples dimension
    base_color_expanded = base_color.unsqueeze(1)  # [100000, 1, 3]
    metalness_expanded = metalness.unsqueeze(1)    # [100000, 1, 1]
    roughness_expanded = roughness.unsqueeze(1)    # [100000, 1, 1]

    # Corrected Fresnel equation using standard Schlick's approximation
    def fresnel_schlick(dot_vh, f0):
        exponent = (-5.55473*dot_vh -6.98316) * dot_vh
        result = f0 + (1.0 - f0) * pow(2.0, exponent)
        return result


    def ndf_ggx(alpha, dot_nh):
        alpha2 = alpha * alpha
        div_denominator = 1.0 / ((dot_nh * dot_nh) * (alpha2 - 1.0) + 1.0)
        return alpha2 * div_denominator * div_denominator / torch.pi

    # Geometry function using Smith's method with Schlick-GGX
    def geometry_smith(alpha, dot_nv, dot_nl):
        def geometry_schlick_ggx(dot_nx, k):
            return dot_nx / (dot_nx * (1.0 - k) + k + 1e-6)
        
        k = alpha / 2.0
        ggx2 = geometry_schlick_ggx(dot_nv, k)
        ggx1 = geometry_schlick_ggx(dot_nl, k)
        
        return ggx1 * ggx2
    
    # Fresnel at normal incidence
    f0 = 0.04 * (1.0 - metalness_expanded) + base_color_expanded * metalness_expanded

    alpha = roughness_expanded ** 2
    alpha = torch.clamp(alpha, 0.05, 0.95)

    # Compute BRDF terms
    Fr = fresnel_schlick(dot_vh, f0)
    D = ndf_ggx(alpha, dot_nh)
    G = geometry_smith(alpha, dot_nv.unsqueeze(1), dot_nl)
    
    # Specular BRDF term
    f_s = (D * Fr * G) / (4.0 * dot_nv.unsqueeze(1) * dot_nl + 1e-6)

    # Energy-conserving diffuse BRDF term
    f_d = (1.0 - metalness_expanded) * base_color_expanded / torch.pi



    return f_d, f_s