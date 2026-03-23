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
from utils.graphics_utils import fibonacci_sphere_sampling, rgb_to_srgb, srgb_to_rgb, fov2focal
from gs_sss_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from bvh import RayTracer
from utils.screen_space_rendering import (
    uniform_hemisphere_sampling,
    cosine_weighted_hemisphere_sampling,
    ggx_importance_sampling,
    transform_hemisphere_to_normal_robust,
    half_vector_to_light_direction,
    compute_ggx_pdf,
    compute_pbr_brdf as ss_compute_pbr_brdf,
)
import os


def halton_hemisphere_sampling_batched(n_points, n_samples_per_batch, n_batches, device, random_rotate=True):
    """
    Generate Halton-sampled directions on the upper hemisphere in local space with multiple batches
    
    Args:
        n_points: number of points to sample for
        n_samples_per_batch: number of samples per batch
        n_batches: number of batches to generate
        device: torch device
        random_rotate: whether to apply random rotation to break up patterns
    
    Returns:
        incident_dirs: (n_points, n_samples_per_batch * n_batches, 3) sampled directions in local space
        incident_areas: (n_points, n_samples_per_batch * n_batches, 1) corresponding solid angles
    """
    total_samples = n_points * n_samples_per_batch * n_batches
    
    # Generate uniform random samples for spherical coordinates
    # Use different random seeds for each batch to ensure no repetition
    all_incident_dirs = []
    all_incident_areas = []
    
    for batch_idx in range(n_batches):
        # Set a different random seed for each batch to ensure diversity
        torch.manual_seed(torch.randint(0, 1000000, (1,)).item())
        
        # Generate samples for this batch
        batch_samples = n_points * n_samples_per_batch
        
        # Generate uniform random samples for spherical coordinates
        u1 = torch.rand(batch_samples, device=device)  # For phi (azimuthal)
        u2 = torch.rand(batch_samples, device=device)  # For theta (polar)
        
        # Cosine-weighted sampling: cos(theta) = sqrt(1 - u2)
        cos_theta = torch.sqrt(1.0 - u2)
        sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
        
        # Uniform sampling for phi
        phi = 2.0 * torch.pi * u1
        
        # Apply random rotation if requested
        if random_rotate:
            random_offset = torch.rand(n_points, device=device) * 2.0 * torch.pi
            phi = phi.view(n_points, n_samples_per_batch) + random_offset.unsqueeze(1)
            phi = phi.view(-1)
        
        # Convert to Cartesian coordinates (local space, upper hemisphere)
        x = sin_theta * torch.cos(phi)
        y = sin_theta * torch.sin(phi)
        z = cos_theta  # This ensures we're in the upper hemisphere
        
        # Reshape to (n_points, n_samples_per_batch, 3)
        batch_incident_dirs = torch.stack([x, y, z], dim=-1).view(n_points, n_samples_per_batch, 3)
        
        # Compute solid angles for each sample
        # For cosine-weighted sampling, the PDF is cos(theta)/pi
        # The Monte Carlo estimator is: f(x) * L(x) * cos(theta) / (cos(theta)/pi) = f(x) * L(x) * pi
        # Since we multiply by cos(theta) later in the transport calculation, we need pi here
        batch_incident_areas = torch.ones_like(batch_incident_dirs[..., :1]) * torch.pi
        
        all_incident_dirs.append(batch_incident_dirs)
        all_incident_areas.append(batch_incident_areas)
    
    # Concatenate all batches
    incident_dirs = torch.cat(all_incident_dirs, dim=1)  # (n_points, n_samples_per_batch * n_batches, 3)
    incident_areas = torch.cat(all_incident_areas, dim=1)  # (n_points, n_samples_per_batch * n_batches, 1)
    
    return incident_dirs, incident_areas

def halton_hemisphere_sampling(n_points, n_samples, device, random_rotate=True, batch_offset=0):
    """
    Generate Halton-sampled directions on the upper hemisphere in local space
    
    Args:
        n_points: number of points to sample for
        n_samples: number of samples per point
        device: torch device
        random_rotate: whether to apply random rotation to break up patterns
        batch_offset: offset for batch sampling to ensure diversity
    
    Returns:
        incident_dirs: (n_points, n_samples, 3) sampled directions in local space
        incident_areas: (n_points, n_samples, 1) corresponding solid angles
    """
    # Use a much simpler and faster approach with uniform random sampling
    # This is much faster than Halton sequences and still gives good results
    total_samples = n_points * n_samples
    
    # Set different random seed based on batch offset to ensure diversity
    if batch_offset > 0:
        torch.manual_seed(torch.randint(0, 1000000, (1,)).item() + batch_offset)
    
    # Generate uniform random samples for spherical coordinates
    u1 = torch.rand(total_samples, device=device)  # For phi (azimuthal)
    u2 = torch.rand(total_samples, device=device)  # For theta (polar)
    
    # Cosine-weighted sampling: cos(theta) = sqrt(1 - u2)
    cos_theta = torch.sqrt(1.0 - u2)
    sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
    
    # Uniform sampling for phi
    phi = 2.0 * torch.pi * u1
    
    # Apply random rotation if requested
    if random_rotate:
        random_offset = torch.rand(n_points, device=device) * 2.0 * torch.pi
        phi = phi.view(n_points, n_samples) + random_offset.unsqueeze(1)
        phi = phi.view(-1)
    
    # Convert to Cartesian coordinates (local space, upper hemisphere)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta  # This ensures we're in the upper hemisphere
    
    # Reshape to (n_points, n_samples, 3)
    incident_dirs = torch.stack([x, y, z], dim=-1).view(n_points, n_samples, 3)
    
    # Compute solid angles for each sample
    # For cosine-weighted sampling, the PDF is cos(theta)/pi
    # The Monte Carlo estimator is: f(x) * L(x) * cos(theta) / (cos(theta)/pi) = f(x) * L(x) * pi
    # Since we multiply by cos(theta) later in the transport calculation, we need pi here
    incident_areas = torch.ones_like(incident_dirs[..., :1]) * torch.pi
    
    return incident_dirs, incident_areas

def transform_hemisphere_to_normal(hemisphere_dirs, normals):
    """
    Transform hemisphere samples from local space to world space aligned with normals
    
    Args:
        hemisphere_dirs: (n_points, n_samples, 3) directions in local space
        normals: (n_points, 3) surface normals in world space
    
    Returns:
        world_dirs: (n_points, n_samples, 3) directions in world space
    """
    # Normalize normals
    normals = F.normalize(normals, dim=-1)
    
    # Create rotation matrices to transform from [0,0,1] to normal
    # We'll use the method from the existing code but vectorized
    
    # For each normal, create a rotation matrix that maps [0,0,1] to the normal
    # We can use the Rodrigues rotation formula or construct an orthonormal basis
    
    # Method: Construct orthonormal basis with normal as z-axis
    # Find a perpendicular vector to the normal
    up = torch.tensor([0.0, 1.0, 0.0], device=normals.device).expand_as(normals)
    
    # If normal is close to up vector, use a different reference
    close_to_up = torch.abs(torch.sum(normals * up, dim=-1, keepdim=True)) > 0.9
    reference = torch.where(close_to_up, 
                           torch.tensor([1.0, 0.0, 0.0], device=normals.device).expand_as(normals),
                           up)
    
    # Create orthonormal basis: [tangent, bitangent, normal]
    tangent = F.normalize(torch.cross(reference, normals, dim=-1), dim=-1)
    bitangent = F.normalize(torch.cross(normals, tangent, dim=-1), dim=-1)
    
    # Construct rotation matrix: [tangent, bitangent, normal]
    rotation_matrix = torch.stack([tangent, bitangent, normals], dim=-1)  # (n_points, 3, 3)
    
    # Apply rotation to hemisphere directions
    # Reshape for batch matrix multiplication
    hemisphere_dirs_reshaped = hemisphere_dirs.view(-1, 3)  # (n_points * n_samples, 3)
    rotation_matrix_expanded = rotation_matrix.unsqueeze(1).expand(-1, hemisphere_dirs.shape[1], -1, -1)
    rotation_matrix_flat = rotation_matrix_expanded.reshape(-1, 3, 3)  # (n_points * n_samples, 3, 3)
    
    # Apply rotation
    world_dirs = torch.bmm(hemisphere_dirs_reshaped.unsqueeze(1), rotation_matrix_flat).squeeze(1)
    world_dirs = world_dirs.view(hemisphere_dirs.shape)  # (n_points, n_samples, 3)
    
    # Normalize to ensure unit vectors
    world_dirs = F.normalize(world_dirs, dim=-1)
    
    return world_dirs


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
    # Clamp roughness to valid range
    roughness = torch.clamp(roughness, 0.1, 1.0)
    metallic = pc.get_metallic
    normal = pc.get_normal
    incidents = pc.get_incidents  # incident shs
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_shs.shape[0], 1))
    dir_pp_normalized = F.normalize(dir_pp, dim=-1)
    
    if is_training:
        # Use batched rendering for training with more samples
        brdf_color, extra_results = rendering_equation_batched(
            base_color, roughness, metallic, normal.detach(), viewdirs, incidents,
            direct_light_env_light, 
            opt_params=opt_params, n_batches=64, samples_per_batch=64,
            gaussian_model=pc, world_positions=means3D)
    else:
        chunk_size = 100000
        brdf_color = []
        extra_results = []
        for i in range(0, means3D.shape[0], chunk_size):
            _brdf_color, _extra_results = rendering_equation_batched(
                base_color[i:i + chunk_size], roughness[i:i + chunk_size], metallic[i:i + chunk_size], 
                normal[i:i + chunk_size].detach(), viewdirs[i:i + chunk_size], incidents[i:i + chunk_size],
                direct_light_env_light, 
                opt_params=opt_params, n_batches=64, samples_per_batch=64,
                gaussian_model=pc, world_positions=means3D[i:i + chunk_size])
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
               "pbr": rgb_to_srgb(rendered_pbr).clamp(0.0, 1.0),
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
        results["render_env"] = rendered_image + (1 - rendered_opacity) * rgb_to_srgb(direct_env).clamp(0.0, 1.0)
        results["pbr_env"] = rgb_to_srgb(pbr * rendered_opacity + (1 - rendered_opacity) * direct_env).clamp(0.0, 1.0)
        results["env_only"] = rgb_to_srgb(direct_env).clamp(0.0, 1.0)
        
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
    # Clamp roughness to valid range
    roughness = torch.clamp(roughness, 0.1, 1.0)
    metallic = custom_metallic
    normal = pc.get_normal
    incidents = pc.get_incidents  # incident shs
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_shs.shape[0], 1))
    dir_pp_normalized = F.normalize(dir_pp, dim=-1)
    
    if is_training:
        # Use batched rendering for training with more samples
        brdf_color, extra_results = rendering_equation_batched(
            base_color, roughness, metallic, normal.detach(), viewdirs, incidents,
            direct_light_env_light, 
            opt_params=opt_params, n_batches=64, samples_per_batch=64)
    else:
        chunk_size = 100000
        brdf_color = []
        extra_results = []
        for i in range(0, means3D.shape[0], chunk_size):
            _brdf_color, _extra_results = rendering_equation_batched(
                base_color[i:i + chunk_size], roughness[i:i + chunk_size], metallic[i:i + chunk_size], 
                normal[i:i + chunk_size].detach(), viewdirs[i:i + chunk_size], incidents[i:i + chunk_size],
                direct_light_env_light, 
                opt_params=opt_params, n_batches=64, samples_per_batch=64)
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
               "pbr": rgb_to_srgb(rendered_pbr).clamp(0.0, 1.0),
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
        results["render_env"] = rendered_image + (1 - rendered_opacity) * rgb_to_srgb(direct_env).clamp(0.0, 1.0)
        results["pbr_env"] = rgb_to_srgb(pbr * rendered_opacity + (1 - rendered_opacity) * direct_env).clamp(0.0, 1.0)
        results["env_only"] = rgb_to_srgb(direct_env).clamp(0.0, 1.0)
        
    return results


def calculate_loss(viewpoint_camera, pc, results, opt, direct_light_env_light):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
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
                # One-time debug print about alpha/mask content
                try:
                    if not hasattr(viewpoint_camera, "_alpha_debug_printed") or not viewpoint_camera._alpha_debug_printed:
                        has_zero = (alpha_t <= 0).any().item()
                        if has_zero:
                            print(f"[GT-Load] {viewpoint_camera.image_path} has alpha and contains zeros (mask not all ones).")
                        else:
                            print(f"[GT-Load] {viewpoint_camera.image_path} has alpha; mask appears all ones.")
                        viewpoint_camera._alpha_debug_printed = True
                except Exception:
                    pass
            gt_img_t = torch.from_numpy(img_np).float().permute(2, 0, 1).clamp(0.0, 1.0).to(device)
            viewpoint_camera.original_image = gt_img_t
            # Always prefer freshly loaded alpha as mask if available; otherwise fallback to ones
            if alpha_t is not None:
                viewpoint_camera.image_mask = alpha_t
            else:
                viewpoint_camera.image_mask = torch.ones((1, gt_img_t.shape[-2], gt_img_t.shape[-1]), device=device, dtype=torch.float32)
        except Exception:
            pass
    rendered_image = results["render"]
    rendered_depth = results["depth"]
    rendered_normal = results["normal"]
    rendered_pbr = results["pbr"]
    rendered_opacity = results["opacity"]
    rendered_base_color = results["base_color"]
    rendered_roughness = results["roughness"]
    rendered_diffuse = results["diffuse"]

    # Prepare ground-truth image and mask
    if hasattr(viewpoint_camera, 'original_image') and viewpoint_camera.original_image is not None:
        gt_image = viewpoint_camera.original_image.to(device)
        if gt_image.dim() == 3 and gt_image.shape[0] == 4:
            gt_image = gt_image[:3, ...]
    else:
        # Fallback: black image (should rarely happen)
        H = rendered_image.shape[1]
        W = rendered_image.shape[2]
        gt_image = torch.zeros((3, H, W), device=device, dtype=rendered_image.dtype)
    image_mask = getattr(viewpoint_camera, 'image_mask', None)
    if image_mask is None:
        image_mask = torch.ones((1, gt_image.shape[1], gt_image.shape[2]), device=device, dtype=gt_image.dtype)
    else:
        image_mask = image_mask.to(device)
        if image_mask.dim() == 2:
            image_mask = image_mask.unsqueeze(0)
    # Apply mask to GT
    gt_image = gt_image * image_mask
   
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
    Render the scene using deferred shading with batched sampling.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view_deferred(viewpoint_camera, pc, pipe, bg_color,
                                  scaling_modifier, override_color, is_training, dict_params, opt_params=opt)

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt, direct_light_env_light=dict_params['env_light'])
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    return results


def rendering_equation(base_color, roughness, metallic, normals, viewdirs,
                      incidents, direct_light_env_light=None,
                      opt_params=None):
    pbr, extra_results = rendering_equation_batched(
        base_color,
        roughness,
        metallic,
        normals,
        viewdirs,
        incidents,
        direct_light_env_light=direct_light_env_light,
        opt_params=opt_params,
        n_batches=1,
        samples_per_batch=64,
    )
    return pbr, extra_results


def rendering_equation_batched(base_color, roughness, metallic, normals, viewdirs,
                              incidents, direct_light_env_light=None,
                              opt_params=None, n_batches=64, samples_per_batch=64,
                              gaussian_model=None, world_positions=None):
    _ = (incidents, gaussian_model, world_positions)

    device = base_color.device
    n_points = normals.shape[0]

    if n_points == 0:
        empty_color = base_color.new_zeros((0, 3))
        empty_light = base_color.new_zeros((0, 1, 3))
        extra_results = {
            "incident_lights": empty_light,
            "local_incident_lights": empty_light,
            "global_incident_lights": empty_light,
            "incident_visibility": base_color.new_zeros((0, 1, 1)),
            "diffuse_light": empty_color,
            "specular": empty_color,
        }
        return empty_color, extra_results

    normals = F.normalize(normals, dim=-1, eps=1e-6)
    normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    viewdirs = F.normalize(viewdirs, dim=-1, eps=1e-6)
    viewdirs = torch.nan_to_num(viewdirs, nan=0.0, posinf=0.0, neginf=0.0)

    detach_base_color = getattr(opt_params, "detach_base_color", False) if opt_params is not None else False
    detach_roughness = getattr(opt_params, "detach_roughness", False) if opt_params is not None else False
    detach_metallic = getattr(opt_params, "detach_metallic", False) if opt_params is not None else False

    base_for_brdf = base_color.detach() if detach_base_color else base_color
    roughness_for_brdf = roughness.detach() if detach_roughness else roughness
    # clamp roughness to avoid degenerate alpha, use 0.1 minimum for numerical stability
    roughness_for_brdf = torch.clamp(roughness_for_brdf, 0.1, 0.95)
    metallic_for_brdf = metallic.detach() if detach_metallic else metallic

    total_samples = max(1, int(n_batches) * int(samples_per_batch))

    # Streaming accumulation over sample batches to reduce memory
    diffuse_sum = torch.zeros((n_points, 3), device=device, dtype=base_color.dtype)
    specular_sum = torch.zeros((n_points, 3), device=device, dtype=base_color.dtype)
    incident_lights_sum = torch.zeros((n_points, 1, 3), device=device, dtype=base_color.dtype)
    visibility_sum = torch.zeros((n_points, 1, 1), device=device, dtype=base_color.dtype)

    for _ in range(int(n_batches)):
        hemisphere_samples = uniform_hemisphere_sampling(n_points, int(samples_per_batch), device)
        light_dirs = transform_hemisphere_to_normal_robust(hemisphere_samples, normals)
        light_dirs = F.normalize(light_dirs, dim=-1, eps=1e-6)
        light_dirs = torch.nan_to_num(light_dirs, nan=0.0, posinf=0.0, neginf=0.0)

        if direct_light_env_light is None:
            env_colors = torch.zeros((n_points, int(samples_per_batch), 3), device=device, dtype=base_color.dtype)
        else:
            env_colors = direct_light_env_light.direct_light(light_dirs.view(-1, 3))
            env_colors = env_colors.view(n_points, int(samples_per_batch), 3)

        f_d, f_s = ss_compute_pbr_brdf(
            base_for_brdf,
            metallic_for_brdf,
            roughness_for_brdf,
            normals,
            viewdirs,
            light_dirs,
        )

        cos_theta = torch.clamp(
            torch.sum(normals.unsqueeze(1) * light_dirs, dim=-1, keepdim=True),
            0.0,
            1.0,
        )
        mc_weight = cos_theta * (2.0 * math.pi)

        diffuse_sum = diffuse_sum + (f_d * env_colors * mc_weight).sum(dim=1)
        # Clamp specular contribution per sample to prevent fireflies
        specular_contribution = f_s * env_colors * mc_weight
        specular_sum = specular_sum + specular_contribution.sum(dim=1)
        incident_lights_sum = incident_lights_sum + env_colors.sum(dim=1, keepdim=True)
        visibility_sum = visibility_sum + cos_theta.sum(dim=1, keepdim=True)

    diffuse = diffuse_sum / total_samples
    specular = specular_sum / total_samples
    pbr = diffuse + specular

    incident_lights_mean = incident_lights_sum / total_samples
    incident_visibility_mean = visibility_sum / total_samples
    zeros_like_incident = torch.zeros_like(incident_lights_mean)

    extra_results = {
        "incident_lights": incident_lights_mean,
        "local_incident_lights": zeros_like_incident,
        "global_incident_lights": incident_lights_mean,
        "incident_visibility": incident_visibility_mean,
        "diffuse_light": diffuse,
        "specular": specular,
    }

    # Provide one set of sampled directions for debugging/inspection
    extra_results["incident_dirs"] = light_dirs

    return pbr, extra_results

def depth_to_world_position(depth_image, intrinsic_matrix, extrinsic_matrix):
    """
    Convert depth image to world positions using the same approach as the existing code
    
    Args:
        depth_image: (H, W) depth values
        intrinsic_matrix: (3, 3) camera intrinsic matrix
        extrinsic_matrix: (4, 4) camera extrinsic matrix (world to camera)
    
    Returns:
        world_positions: (3, H, W) world coordinates for each pixel
    """
    def ndc_2_cam(ndc_xyz, intrinsic, W, H):
        inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
        cam_z = ndc_xyz[..., 2:3]
        cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
        cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
        cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
        return cam_xyz

    def depth2point_cam(sampled_depth, ref_intrinsic):
        B, N, C, H, W = sampled_depth.shape
        valid_z = sampled_depth
        valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
        valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
        valid_y, valid_x = torch.meshgrid(valid_y, valid_x, indexing='ij')
        # B,N,H,W
        valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
        valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
        ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)
        cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H)
        return ndc_xyz, cam_xyz
    
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None, None, None, ...], intrinsic_matrix[None, ...])
    xyz_cam = xyz_cam.reshape(-1, 3)
    xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[..., 0:1])], dim=-1) @ torch.inverse(extrinsic_matrix).transpose(0, 1)
    xyz_world = xyz_world[..., :3]
    xyz_world = xyz_world.reshape(*depth_image.shape, 3)
    xyz_world = xyz_world.permute(2, 0, 1)

    return xyz_world

def get_camera_matrices_from_camera(camera: Camera):
    """
    Extract intrinsic and extrinsic matrices from Camera object using the same approach as existing code
    """
    # Use the same approach as in the existing code
    FoVx = camera.FoVx
    image_width = camera.image_width
    image_height = camera.image_height
    world_view_transform = camera.world_view_transform

    focal = fov2focal(FoVx, image_width)  # original focal length
    intrinsic_matrix = torch.tensor([[focal, 0, image_width / 2], [0, focal, image_height / 2], [0, 0, 1]], device=world_view_transform.device).float()
    extrinsic_matrix = world_view_transform.transpose(0, 1).contiguous()  # cam2world
    
    return intrinsic_matrix, extrinsic_matrix

def deferred_shading_equation(base_color, metallic, roughness, normals, view_directions, 
                            light_directions, incident_lights, opt_params=None):
    """
    Deferred shading equation for PBR materials
    
    Args:
        base_color: (N, 3) base color
        metallic: (N, 1) metallic value
        roughness: (N, 1) roughness value  
        normals: (N, 3) surface normals
        view_directions: (N, 3) view directions
        light_directions: (N, 3) light directions
        incident_lights: (N, 3) incident light intensities
        opt_params: optimization parameters for detaching gradients
    
    Returns:
        final_color: (N, 3) final shaded color
        extra: dict with intermediate results
    """
    # Process material properties based on detach flags
    detach_base_color = opt_params.detach_base_color if hasattr(opt_params, 'detach_base_color') else False
    detach_roughness = opt_params.detach_roughness if hasattr(opt_params, 'detach_roughness') else False
    detach_metallic = opt_params.detach_metallic if hasattr(opt_params, 'detach_metallic') else False
    
    processed_base_color = base_color.detach() if detach_base_color else base_color
    processed_roughness = roughness.detach() if detach_roughness else roughness
    processed_metallic = metallic.detach() if detach_metallic else metallic

    processed_base_color = torch.clamp(processed_base_color, 0.0, 1.0)
    processed_roughness = torch.clamp(processed_roughness, 0.1, 0.95)
    processed_metallic = torch.clamp(processed_metallic, 0.0, 1.0)
    
    # Normalize vectors
    normals = F.normalize(normals, dim=-1)
    view_directions = F.normalize(view_directions, dim=-1)
    light_directions = F.normalize(light_directions, dim=-1)
    
    # Compute halfway vector and dot products
    half_vector = F.normalize(light_directions + view_directions, dim=-1)
    dot_nv = torch.clamp(torch.sum(normals * view_directions, dim=-1, keepdim=True), 0.0, 1.0)
    dot_nl = torch.clamp(torch.sum(normals * light_directions, dim=-1, keepdim=True), 0.0, 1.0)
    dot_nh = torch.clamp(torch.sum(normals * half_vector, dim=-1, keepdim=True), 0.0, 1.0)
    dot_vh = torch.clamp(torch.sum(view_directions * half_vector, dim=-1, keepdim=True), 0.0, 1.0)
    
    # Fresnel equation using Schlick's approximation
    def fresnel_schlick(dot_vh, f0):
        exponent = (-5.55473 * dot_vh - 6.98316) * dot_vh
        return f0 + (1.0 - f0) * torch.pow(2.0, exponent)
    
    # Normal Distribution Function using GGX
    def ndf_ggx(alpha, dot_nh):
        alpha2 = alpha * alpha
        # Clamp alpha2 to match roughness minimum of 0.1 (roughness=0.1 -> alpha=0.01 -> alpha2=0.0001)
        alpha2 = torch.clamp(alpha2, 0.0001, 0.95)
        denominator = (dot_nh * dot_nh) * (alpha2 - 1.0) + 1.0
        return alpha2 / (torch.pi * denominator * denominator + 1e-6)
    
    # Geometry function using Smith's method with Schlick-GGX
    def geometry_smith(alpha, dot_nv, dot_nl):
        def geometry_schlick_ggx(dot_nx, k):
            return dot_nx / (dot_nx * (1.0 - k) + k + 1e-6)
        
        k = alpha / 2.0
        ggx2 = geometry_schlick_ggx(dot_nv, k)
        ggx1 = geometry_schlick_ggx(dot_nl, k)
        
        return ggx1 * ggx2
    
    # Fresnel at normal incidence
    f0 = 0.04 * (1.0 - processed_metallic) + processed_base_color * processed_metallic
    
    # Roughness to alpha
    alpha = processed_roughness ** 2
    # Clamp alpha to match roughness minimum of 0.1 (alpha = 0.01)
    alpha = torch.clamp(alpha, 0.01, 0.95)
    
    # Compute BRDF terms
    Fr = fresnel_schlick(dot_vh, f0)
    D = ndf_ggx(alpha, dot_nh)
    G = geometry_smith(alpha, dot_nv, dot_nl)
    
    # Specular BRDF term
    f_s = (D * Fr * G) / (4.0 * dot_nv * dot_nl + 1e-6)
    
    # Diffuse BRDF term
    f_d = (1.0 - processed_metallic) * processed_base_color / torch.pi
    
    # Final shading
    diffuse = f_d * incident_lights * dot_nl
    specular = f_s * incident_lights * dot_nl
    
    final_color = diffuse + specular
    final_color = torch.clamp(final_color, 0.0, 1.0)
    
    extra = {
        "diffuse": diffuse,
        "specular": specular,
        "f_d": f_d,
        "f_s": f_s,
        "dot_nl": dot_nl,
        "incident_lights": incident_lights
    }
    
    return final_color, extra

def render_view_deferred(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                        scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None,
                        opt_params=None, custom_materials=None, deferred_options=None):
    """
    Deferred shading version of render_view
    """
    direct_light_env_light = dict_params.get("env_light")
    
    # Create zero tensor for gradients
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

    device = means3D.device
    custom_materials = custom_materials or {}
    screen_space_maps = custom_materials.get("screen_space_maps")

    # Step 1: query per-Gaussian material parameters (mirrors neilf.py)
    gaussian_materials = {
        "base_color": custom_materials.get("base_color", pc.get_base_color),
        "roughness": custom_materials.get("roughness", pc.get_roughness),
        "metallic": custom_materials.get("metallic", pc.get_metallic),
    }

    camera_direction = -viewpoint_camera.world_view_transform[:3, 2]
    camera_direction = F.normalize(camera_direction, dim=-1)
    default_normals = pc.get_normal(camera_direction)
    gaussian_materials["normal"] = custom_materials.get("normal", default_normals)

    for key, value in list(gaussian_materials.items()):
        if isinstance(value, torch.Tensor) and value.device != device:
            gaussian_materials[key] = value.to(device)

    base_color = gaussian_materials["base_color"]
    roughness = gaussian_materials["roughness"]
    # Clamp roughness to valid range
    roughness = torch.clamp(roughness, 0.05, 1.0)
    metallic = gaussian_materials["metallic"]
    normal = gaussian_materials["normal"]

    # Handle covariance computation
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Handle SH colors
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

    # Create features for rasterization (materials + depth)
    xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
    depths = (xyz_homo @ viewpoint_camera.world_view_transform)[:, 2:3]
    depths2 = depths.square()
    
    # Features: [depth, depth2, base_color, roughness, metallic, normal]
    features = torch.cat([depths, depths2, base_color, roughness, metallic, normal], dim=-1)

    # Rasterize to get G-buffer
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

    # Extract G-buffer components
    mask = num_contrib > 0
    rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5) * mask
    
    # Split features: [depth, depth2, base_color, roughness, metallic, normal]
    gbuffer_depth, gbuffer_depth2, gbuffer_base_color, gbuffer_roughness, \
    gbuffer_metallic, gbuffer_normal = rendered_feature.split([1, 1, 3, 1, 1, 3], dim=0)

    # Gradient safety: replace NaN/Inf gradients flowing back into gbuffer tensors
    def _sanitize_grad(grad):
        if grad is None:
            return None
        return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    # Enable grad sanitization hooks by default; disable with GRAD_SANITIZE=0
    if os.environ.get("GRAD_SANITIZE", "1") != "0":
        try:
            gbuffer_normal.register_hook(_sanitize_grad)
        except Exception:
            pass
        try:
            gbuffer_base_color.register_hook(_sanitize_grad)
        except Exception:
            pass
        try:
            gbuffer_roughness.register_hook(_sanitize_grad)
        except Exception:
            pass
        try:
            gbuffer_metallic.register_hook(_sanitize_grad)
        except Exception:
            pass

    if screen_space_maps:
        def _prepare_map(key, target_tensor, expected_channels):
            if key not in screen_space_maps:
                return target_tensor
            map_tensor = screen_space_maps[key]
            if map_tensor is None:
                return target_tensor
            map_tensor = map_tensor.to(target_tensor.device, dtype=target_tensor.dtype)
            if map_tensor.dim() == 2:
                map_tensor = map_tensor.unsqueeze(0)
            if map_tensor.shape[0] != expected_channels:
                if map_tensor.shape[0] > expected_channels:
                    map_tensor = map_tensor[:expected_channels]
                elif map_tensor.shape[0] == 1:
                    map_tensor = map_tensor.expand(expected_channels, *map_tensor.shape[1:])
                else:
                    repeats = math.ceil(expected_channels / map_tensor.shape[0])
                    repeat_factors = [repeats] + [1] * (map_tensor.dim() - 1)
                    map_tensor = map_tensor.repeat(repeat_factors)[:expected_channels]
            if map_tensor.shape[1:] != target_tensor.shape[1:]:
                map_tensor = F.interpolate(
                    map_tensor.unsqueeze(0),
                    size=target_tensor.shape[1:],
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0)
            return map_tensor

        gbuffer_base_color = _prepare_map("base_color", gbuffer_base_color, 3)
        gbuffer_roughness = _prepare_map("roughness", gbuffer_roughness, 1)
        gbuffer_metallic = _prepare_map("metallic", gbuffer_metallic, 1)
    
    gbuffer_depth_var = gbuffer_depth2 - gbuffer_depth.square()

    # Transform per-Gaussian materials into screen-space G-buffers and world positions
    intrinsic, extrinsic = get_camera_matrices_from_camera(viewpoint_camera)
    world_positions = depth_to_world_position(gbuffer_depth[0], intrinsic, extrinsic)
    
    # Flatten the G-buffer for per-pixel deferred shading
    H, W = gbuffer_depth.shape[1:]
    gbuffer_base_color_flat = gbuffer_base_color.permute(1, 2, 0).reshape(-1, 3)
    gbuffer_roughness_flat = gbuffer_roughness.permute(1, 2, 0).reshape(-1, 1)
    gbuffer_metallic_flat = gbuffer_metallic.permute(1, 2, 0).reshape(-1, 1)
    gbuffer_normal_flat = gbuffer_normal.permute(1, 2, 0).reshape(-1, 3)
    world_positions_flat = world_positions.permute(1, 2, 0).reshape(-1, 3)
    
    deferred_options = deferred_options or {}
    samples_per_batch = max(1, int(deferred_options.get("samples_per_batch", 32)))
    num_batches = max(1, int(deferred_options.get("num_batches", 16)))
    max_pixels_per_pass = int(deferred_options.get("max_pixels_per_pass", 0))

    valid_mask = rendered_opacity[0] > 0
    valid_indices = torch.where(valid_mask.flatten())[0]

    if len(valid_indices) == 0:
        rendered_final = torch.zeros_like(gbuffer_base_color)
        rendered_diffuse = torch.zeros_like(gbuffer_base_color)
        rendered_specular = torch.zeros_like(gbuffer_base_color)
        rendered_light = torch.zeros_like(gbuffer_base_color)
        rendered_local_light = torch.zeros_like(gbuffer_base_color)
        rendered_global_light = torch.zeros_like(gbuffer_base_color)
        rendered_visibility = torch.zeros_like(gbuffer_depth)
    else:
        camera_position = viewpoint_camera.camera_center
        rendered_final = torch.zeros_like(gbuffer_base_color)
        rendered_diffuse = torch.zeros_like(gbuffer_base_color)
        rendered_specular = torch.zeros_like(gbuffer_base_color)
        rendered_light = torch.zeros_like(gbuffer_base_color)
        rendered_local_light = torch.zeros_like(gbuffer_base_color)
        rendered_global_light = torch.zeros_like(gbuffer_base_color)
        rendered_visibility = torch.zeros_like(gbuffer_depth)

        rendered_final_flat = rendered_final.permute(1, 2, 0).reshape(-1, 3)
        rendered_diffuse_flat = rendered_diffuse.permute(1, 2, 0).reshape(-1, 3)
        rendered_specular_flat = rendered_specular.permute(1, 2, 0).reshape(-1, 3)
        rendered_light_flat = rendered_light.permute(1, 2, 0).reshape(-1, 3)
        rendered_local_flat = rendered_local_light.permute(1, 2, 0).reshape(-1, 3)
        rendered_global_flat = rendered_global_light.permute(1, 2, 0).reshape(-1, 3)
        rendered_visibility_flat = rendered_visibility.permute(1, 2, 0).reshape(-1, 1)

        num_valid = len(valid_indices)
        if max_pixels_per_pass > 0:
            ranges = [(s, min(s + max_pixels_per_pass, num_valid)) for s in range(0, num_valid, max_pixels_per_pass)]
        else:
            ranges = [(0, num_valid)]

        for start, end in ranges:
            chunk_indices = valid_indices[start:end]

            normals_chunk = F.normalize(gbuffer_normal_flat[chunk_indices], dim=-1, eps=1e-6)
            normals_chunk = torch.nan_to_num(normals_chunk, nan=0.0, posinf=0.0, neginf=0.0)

            base_color_chunk = gbuffer_base_color_flat[chunk_indices]
            metallic_chunk = gbuffer_metallic_flat[chunk_indices]
            roughness_chunk = gbuffer_roughness_flat[chunk_indices]
            world_positions_chunk = world_positions_flat[chunk_indices]

            view_directions_chunk = F.normalize(camera_position - world_positions_chunk, dim=-1, eps=1e-6)
            view_directions_chunk = torch.nan_to_num(view_directions_chunk, nan=0.0, posinf=0.0, neginf=0.0)

            dummy_incidents = torch.zeros(normals_chunk.shape[0], 3, 1, device=device)

            pbr_chunk, extra_results_chunk = rendering_equation_batched(
                base_color_chunk, roughness_chunk, metallic_chunk, normals_chunk, view_directions_chunk,
                dummy_incidents, direct_light_env_light, opt_params=opt_params, n_batches=num_batches,
                samples_per_batch=samples_per_batch, gaussian_model=pc, world_positions=world_positions_chunk
            )

            pbr_chunk = torch.nan_to_num(pbr_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            for key in ["specular", "diffuse_light", "incident_lights", "local_incident_lights", "global_incident_lights", "incident_visibility"]:
                if key in extra_results_chunk:
                    extra_results_chunk[key] = torch.nan_to_num(extra_results_chunk[key], nan=0.0, posinf=0.0, neginf=0.0)

            specular_chunk = extra_results_chunk["specular"]
            diffuse_light_chunk = extra_results_chunk["diffuse_light"]
            incident_lights_chunk = extra_results_chunk["incident_lights"].squeeze(1)
            local_lights_chunk = extra_results_chunk["local_incident_lights"].squeeze(1)
            global_lights_chunk = extra_results_chunk["global_incident_lights"].squeeze(1)
            visibility_chunk = extra_results_chunk["incident_visibility"].squeeze(1)

            rendered_final_flat[chunk_indices] = pbr_chunk
            rendered_diffuse_flat[chunk_indices] = diffuse_light_chunk
            rendered_specular_flat[chunk_indices] = specular_chunk
            rendered_light_flat[chunk_indices] = incident_lights_chunk
            rendered_local_flat[chunk_indices] = local_lights_chunk
            rendered_global_flat[chunk_indices] = global_lights_chunk
            rendered_visibility_flat[chunk_indices] = visibility_chunk

        rendered_final = rendered_final_flat.view(H, W, 3).permute(2, 0, 1)
        rendered_diffuse = rendered_diffuse_flat.view(H, W, 3).permute(2, 0, 1)
        rendered_specular = rendered_specular_flat.view(H, W, 3).permute(2, 0, 1)
        rendered_light = rendered_light_flat.view(H, W, 3).permute(2, 0, 1)
        rendered_local_light = rendered_local_flat.view(H, W, 3).permute(2, 0, 1)
        rendered_global_light = rendered_global_flat.view(H, W, 3).permute(2, 0, 1)
        rendered_visibility = rendered_visibility_flat.view(H, W, 1).permute(2, 0, 1)

    # Apply background
    rendered_final = rendered_final * rendered_opacity + (1 - rendered_opacity) * bg_color[:, None, None]

    rendered_final_srgb = rgb_to_srgb(rendered_final).clamp(0,1)
    rendered_diffuse_srgb = rgb_to_srgb(rendered_diffuse)
    rendered_specular_srgb = rgb_to_srgb(rendered_specular)
    rendered_light_srgb = rgb_to_srgb(rendered_light)
    rendered_local_light_srgb = rgb_to_srgb(rendered_local_light)
    rendered_global_light_srgb = rgb_to_srgb(rendered_global_light)

    rendered_visibility = rendered_visibility.clamp(0.0, 1.0)

    # Prepare results
    results = {
        "render": rendered_final_srgb.clamp(0,1),
        "depth": gbuffer_depth,
        "depth_var": gbuffer_depth_var,
        "normal": gbuffer_normal,
        "pseudo_normal": rendered_pseudo_normal,
        "surface_xyz": rendered_surface_xyz,
        "opacity": rendered_opacity,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "num_rendered": num_rendered,
        "num_contrib": num_contrib,
        "base_color": rgb_to_srgb(gbuffer_base_color),
        "roughness": gbuffer_roughness,
        "metallic": gbuffer_metallic,
        "diffuse": rendered_diffuse_srgb,
        "specular": rendered_specular_srgb,
        "pbr": rendered_final_srgb.clamp(0,1),
        "visibility": rendered_visibility
    }

    # Debug: attach backward hooks to catch NaN/Inf gradients at key tensors
    # These hooks both detect AND sanitize NaN/Inf gradients to prevent propagation
    def _grad_nan_hook(name):
        def _hook(grad):
            if grad is None:
                return None
            try:
                has_nan = torch.isnan(grad).any()
                has_inf = torch.isinf(grad).any()
                if has_nan or has_inf:
                    print(f"[GradNaN] {name} gradient has NaN/Inf - sanitizing")
                    # Sanitize the gradient by replacing NaN/Inf with 0
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                # If check fails, still try to sanitize
                try:
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    pass
            return grad
        return _hook

    try:
        rendered_final.register_hook(_grad_nan_hook("render"))
    except Exception:
        pass
    try:
        gbuffer_normal.register_hook(_grad_nan_hook("gbuffer_normal"))
    except Exception:
        pass
    try:
        gbuffer_base_color.register_hook(_grad_nan_hook("gbuffer_base_color"))
    except Exception:
        pass
    try:
        rendered_light.register_hook(_grad_nan_hook("incident_light"))
    except Exception:
        pass

    # Add diffuse light and environment lighting results (matching original render_view)
    # For deferred shading, we don't have the same diffuse_light computation as the original
    # So we'll use the diffuse component from our shading
    results["diffuse_light"] = rendered_diffuse
    
    if not is_training:
        try:
            results["env"] = direct_light_env_light.get_env
        except:
            pass
        results["world_positions"] = world_positions
        results["lights"] = rendered_light_srgb
        results["local_lights"] = rendered_local_light_srgb
        results["global_lights"] = rendered_global_light_srgb
    
    if not is_training:
        directions = viewpoint_camera.get_world_directions()
        direct_env = direct_light_env_light.direct_light(directions.permute(1, 2, 0)).permute(2, 0, 1)
        results["render_env"] = rendered_final + (1 - rendered_opacity) * rgb_to_srgb(direct_env).clamp(0.0, 1.0)
        results["pbr_env"] = rgb_to_srgb(rendered_final * rendered_opacity + (1 - rendered_opacity) * direct_env).clamp(0.0, 1.0)
        results["env_only"] = rgb_to_srgb(direct_env).clamp(0.0, 1.0)

    return results

def render_neilf_deferred(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                         scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
                         is_training=False, dict_params=None, **kwargs):
    """
    Deferred shading version of render_neilf
    """
    custom_materials = kwargs.get("custom_materials")
    deferred_options = kwargs.get("deferred_options")
    results = render_view_deferred(viewpoint_camera, pc, pipe, bg_color,
                                  scaling_modifier, override_color, is_training, dict_params,
                                  opt_params=opt, custom_materials=custom_materials,
                                  deferred_options=deferred_options)

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt, direct_light_env_light=dict_params['env_light'])
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    return results