#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from gs_sss_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from bvh import RayTracer
from utils.graphics_utils import fov2focal
from torch.nn import functional as F

def render_equation_new(base_color, metalness, roughness, subsurfaceness, normal, residual, light_direction, view_direction, incident_light, debug=False):
    """
    Render equation for the PBR shading model in pytorch

    base_color: tensor of shape (N, 3)
    metalness: tensor of shape (N, 1)
    roughness: tensor of shape (N, 1)
    subsurfaceness: tensor of shape (N, 1)
    normal: tensor of shape (N, 3)
    incident_light: tensor of shape (N, 3)
    residual: tensor of shape (N, 3) # subsurface irradiance
    light_direction: tensor of shape (N, 3) away from the incident point
    view_direction: tensor of shape (N, 3) away from the incident point
    """

    # Compute halfway vector and dot products
    half_vector = F.normalize(light_direction + view_direction, dim=-1)
    dot_nv = torch.clamp(torch.sum(normal * view_direction, dim=-1, keepdim=True), 0.0, 1.0)
    dot_nl = torch.clamp(torch.sum(normal * light_direction, dim=-1, keepdim=True), 0.0, 1.0)
    dot_nh = torch.clamp(torch.sum(normal * half_vector, dim=-1, keepdim=True), 0.0, 1.0)
    dot_vh = torch.clamp(torch.sum(view_direction * half_vector, dim=-1, keepdim=True), 0.0, 1.0)

    # Fresnel equation using Schlick's approximation
    def fresnel_schlick(dot_vh, f0):
        return f0 + (1.0 - f0) * torch.pow(2, -5.55473 * dot_vh - 6.98316 * dot_vh)

    # Calculate NDF (Normal Distribution Function) using GGX/Trowbridge-Reitz
    def ndf_ggx(roughness, dot_nh):
        alpha = roughness ** 2
        alpha2 = alpha ** 2
        denominator = (dot_nh ** 2) * (alpha2 - 1.0) + 1.0
        return alpha2 / (torch.pi * (denominator ** 2) + 1e-6)

    # Geometry function using Smith's method with Schlick-GGX
    def geometry_smith(roughness, dot_nv, dot_nl):
        def geometry_schlick_ggx(dot_nx, roughness):
            k = ((roughness + 1) ** 2) / 8 # Not for IBL see https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
            return dot_nx / (dot_nx * (1.0 - k) + k + 1e-6)
        
        ggx2 = geometry_schlick_ggx(dot_nv, roughness)
        ggx1 = geometry_schlick_ggx(dot_nl, roughness)
        
        return ggx1 * ggx2

    # Fresnel at normal incidence
    f0 = 0.04 * (1.0 - metalness) + base_color * metalness # Fresnel at normal incidence

    # Specular term
    Fr = fresnel_schlick(dot_vh, f0) # Fresnel
    D = ndf_ggx(roughness, dot_nh) #  # Normal Distribution Function
    G = geometry_smith(roughness, dot_nv, dot_nl) # Geometry Function
    specular = (D * Fr * G) / (4.0 * dot_nv * dot_nl + 1e-5)

    # Diffuse term (Lambertian reflection)
    diffuse = (1.0 - metalness) * base_color / torch.pi

    # Combine the components using the subsurface weight as blending factor
    surface_reflection = (diffuse + specular) * incident_light * dot_nl
    pbr = surface_reflection
    pbr = torch.clamp(pbr, 0.0, 1.0)
    pbr_combined = (1.0 - subsurfaceness) * pbr + subsurfaceness * residual
    pbr_combined = pbr
    pbr_combined = torch.clamp(pbr_combined, 0.0, 1.0)
    
    # pbr_combined = pbr

    extra = {
        'diffuse': diffuse * incident_light * dot_nl,
        'specular': specular * incident_light * dot_nl,
        'pbr': pbr
    }

    return pbr_combined, extra

def render_equation(base_color, metalness, roughness, subsurfaceness, normal, visibilities, residual, light_direction, view_direction, incident_light, debug=False):
    """
    Render equation for the PBR shading model

    base_color: tensor of shape (N, 3)
    metalness: tensor of shape (N, 1)
    roughness: tensor of shape (N, 1)
    normal: tensor of shape (N, 3)
    visibilty: tensor of shape (N, 3)
    incident_light: tensor of shape (N, 3)
    residual: tensor of shape (N, 3)
    light_direction: tensor of shape (N, 3)
    view_direction: tensor of shape (N, 3)
    light_position: tensor of shape (N, 3)
    light_intensity: tensor of shape (N, 3)
    positions: tensor of shape (N, 3)
    """

    # From: https://google.github.io/filament/Filament.html#materialsystem/specularbrdf/normaldistributionfunction(speculard)


    # Distribution
    def D_GGX(ndh, alpha):
        alpha2 = alpha * alpha
        f = (ndh * alpha2 - ndh) * ndh + 1.0
        return alpha2 / (np.pi * f * f + 1e-5)

    # Geometry
    def V_SmithGGXCorrelate(ndv, ndl, alpha):
        alpha2 = alpha * alpha
        ggxl = ndv * torch.sqrt((-ndl * alpha2 + ndl) * ndl + alpha2 + 1e-5)
        ggxv = ndl * torch.sqrt((-ndv * alpha2 + ndv) * ndv + alpha2 + 1e-5)
        return 0.5 / (ggxl + ggxv + 1e-5)

    # Fresnel
    def F_Schlick(u, f0):
        return f0 + (1.0 - f0) * ((1.0 - u) ** 5)
   

    n = normal
    v = view_direction
    l = light_direction

    h = torch.nn.functional.normalize(l + v, dim=-1)
    ndl = torch.clamp(torch.sum(n * l, dim=-1, keepdim=True), 0, 1)
    ndv = torch.clamp(torch.sum(n * v, dim=-1, keepdim=True), 0, 1)
    ndh = torch.clamp(torch.sum(n * h, dim=-1, keepdim=True), 0, 1)
    ldh = torch.clamp(torch.sum(l * h, dim=-1, keepdim=True), 0, 1)

    # Diffuse 
    diffuse = base_color 
    diffuse_color = (1 - metalness) * base_color # Don't know if I can reparetermise like this
    diffuse = diffuse_color / torch.pi

    # Specular
    # Distribution ( )
    D = D_GGX(ndh, roughness)

    # Geometry
    G = V_SmithGGXCorrelate(ndv, ndl, roughness)

    # Fresnel
    F0 = 0.04 * (1 - metalness) + base_color * metalness # Don't know if I can reparetermise like this
    F = F_Schlick(ldh, F0)


    # Specular shading
    specular = (D * G) * F 

    # Output radiance
    rgb_d = (diffuse * incident_light * ndl)
    rgb_s = (specular * incident_light * ndl)
    pbr = rgb_d + rgb_s 

    # Linearize
    pbr = torch.clamp(pbr, 0, 1)
    pbr = torch.pow(pbr + 1e-5, 1/2.2).clamp(0, 1)
    
    pbr_combined = (1 - subsurfaceness) * pbr + subsurfaceness * residual

    """ with torch.no_grad():
        if debug: 
            color_specular = torch.clamp(rgb_s, 0, 1)
            color_specular = torch.pow(color_specular + 1e-5, 1/2.2).clamp(0, 1)

            color_diffuse = torch.clamp(rgb_d, 0, 1)
            color_diffuse = torch.pow(color_diffuse + 1e-5, 1/2.2).clamp(0, 1)
        else: 
            color_specular = torch.zeros_like(pbr)
            color_diffuse = torch.zeros_like(pbr)    """ 
    
    color_specular = torch.zeros_like(pbr)
    color_diffuse = torch.zeros_like(pbr) 


    extra = {
        "diffuse": color_diffuse,
        "specular": color_specular,
        "pbr": pbr
    }

    return pbr_combined, extra

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, debug=False, evaluate=False, iteration=-1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera["FoVx"] * 0.5)
    tanfovy = math.tan(viewpoint_camera["FoVy"] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera["image_height"]),
        image_width=int(viewpoint_camera["image_width"]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=viewpoint_camera["cx"],
        cy=viewpoint_camera["cy"],
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera["world_view_transform"],
        projmatrix=viewpoint_camera["full_proj_transform"],
        sh_degree=1,
        campos=viewpoint_camera["camera_center"],
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    normals = pc.get_normal()
    positions = pc.get_xyz

    base_color = pc.get_base_color
    metalness = pc.get_metallic
    roughness = pc.get_roughness
    subsurfaceness = pc.get_subsurfaceness
    visibility = pc.get_visibility

    # Freeze roughness for 10 000 iterations
    if iteration < 10_000 and iteration != -1:
        roughness = torch.ones_like(roughness) * 0.5


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


    # View Direction
    camera_position_sss = viewpoint_camera["camera_center"].repeat(means3D.shape[0], 1)
    view_direction_sss = (camera_position_sss - means3D)
    view_direction_normalized_sss = view_direction_sss / (view_direction_sss.norm(dim=-1, keepdim=True) + 1e-5)

    # Light Direction
    light_position_sss = viewpoint_camera["light_position"].repeat(means3D.shape[0], 1)
    light_direction_sss = (light_position_sss - means3D)
    light_direction_normalized_sss = light_direction_sss / (view_direction_sss.norm(dim=-1, keepdim=True) + 1e-5)
    light_distance_sss = light_direction_sss.norm(dim=-1, keepdim=True)


    # Visibility
    visibility_shs_view = visibility.transpose(1, 2).view(-1, 1, 4 ** 2)
    
    # Flip the light direction YZ 
    light_direction_normalized_switched  = light_direction_normalized_sss.clone()
    # light_direction_normalized_switched[:, 1] = -light_direction_normalized_switched[:, 1]
    # light_direction_normalized_switched[:, 2] = -light_direction_normalized_switched[:, 2]
    visibilities = eval_sh(3, visibility_shs_view, light_direction_normalized_switched)
    visibilities = torch.clamp(visibilities + 0.5, 0.0, 1.0)

    # Visibility
    """ with torch.no_grad():
        cov_inv = pc.get_inverse_covariance()
        raytracer = RayTracer(means3D, pc.get_scaling, pc.get_rotation)
        trace_results = raytracer.trace_visibility(means3D, light_direction_sss, means3D, cov_inv, opacity, normals)
        visibilities = trace_results["visibility"] """

    # Residual
    residual, incident_light = pc._sss(positions, rotations, scales, view_direction_normalized_sss, light_direction_normalized_sss, normals, visibilities, light_distance_sss, iteration=iteration)
    residual = torch.clamp(residual, 0, 1)

    # Add PBR to color
    colors_precomp = torch.zeros_like(base_color)
    # base_color = torch.ones_like(base_color) * torch.tensor([0.3, 0.5, 0.5], device=base_color.device)
    # residual = residual * torch.tensor([0.3, 0.5, 0.5], device=base_color.device)
    features = torch.cat([normals, base_color, metalness, roughness, subsurfaceness, visibilities, incident_light, residual], dim=1)

    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth, 
        rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii) = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        features=features)
    
    rendered_color_specular = None
    rendered_color_diffuse = None

    rendered_normal = rendered_feature[:3, :, :]
    rendered_base_color = rendered_feature[3:6, :, :]
    rendered_metalness = rendered_feature[6:7, :, :]
    rendered_roughness = rendered_feature[7:8, :, :]
    rendered_subsurfaceness = rendered_feature[8:9, :, :]
    rendered_visibility = rendered_feature[9:10, :, :]
    rendered_incident_light = rendered_feature[10:13, :, :]
    rendered_residual = rendered_feature[13:16, :, :]

    with torch.no_grad():
        # with torch.no_grad():
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
            ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
            cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
            return ndc_xyz, cam_xyz
        
        def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
            # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
            _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
            xyz_cam = xyz_cam.reshape(-1,3)
            xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
            xyz_world = xyz_world[...,:3]
            xyz_world = xyz_world.reshape(*depth_image.shape, 3)
            xyz_world = xyz_world.permute(2, 0, 1)

            return xyz_world

        def get_calib_matrix_nerf(viewpoint_camera):
            FoVx = viewpoint_camera["FoVx"]
            image_width = viewpoint_camera["image_width"]
            image_height = viewpoint_camera["image_height"]
            world_view_transform = viewpoint_camera["world_view_transform"]

            focal = fov2focal(FoVx, image_width)  # original focal length
            intrinsic_matrix = torch.tensor([[focal, 0, image_width / 2], [0, focal, image_height / 2], [0, 0, 1]], device=world_view_transform.device).float()
            extrinsic_matrix = world_view_transform.transpose(0,1).contiguous() # cam2world
            return intrinsic_matrix, extrinsic_matrix

        # rendered_depth = rendered_depth 
        intrinsic_matrix, extrinsic_matrix = get_calib_matrix_nerf(viewpoint_camera)
        rendered_surface_xyz = depth2point_world(rendered_depth[0], intrinsic_matrix, extrinsic_matrix) 

        # Flatten image space
        rendered_surface_xyz_flat = rendered_surface_xyz.permute(1, 2, 0).reshape(-1, 3)

        # Light Direction
        light_position = viewpoint_camera["light_position"].repeat(rendered_surface_xyz_flat.shape[0], 1)
        light_direction = (light_position - rendered_surface_xyz_flat)
        light_direction_normalized = light_direction / (light_direction.norm(dim=-1, keepdim=True) + 1e-5)

        # View Direction
        camera_position = viewpoint_camera["camera_center"].repeat(rendered_surface_xyz_flat.shape[0], 1)
        view_direction = (camera_position - rendered_surface_xyz_flat)
        view_direction_normalized = view_direction / (view_direction.norm(dim=-1, keepdim=True) + 1e-5)
    

    # Flatten all values 
    rendered_base_color_flat = rendered_base_color.permute(1, 2, 0).reshape(-1, 3)
    rendered_metalness_flat = rendered_metalness.permute(1, 2, 0).reshape(-1, 1)
    rendered_roughness_flat = rendered_roughness.permute(1, 2, 0).reshape(-1, 1)
    rendered_subsurfaceness_flat = rendered_subsurfaceness.permute(1, 2, 0).reshape(-1, 1)
    rendered_normal_flat = rendered_normal.permute(1, 2, 0).reshape(-1, 3)
    rendered_residual_flat = rendered_residual.permute(1, 2, 0).reshape(-1, 3)
    rendered_subsurfaceness_flat = rendered_subsurfaceness.permute(1, 2, 0).reshape(-1, 1)
    rendered_incident_light_flat = rendered_incident_light.permute(1, 2, 0).reshape(-1, 3)

    # Set metallic to 1
    rendered_metalness_flat = torch.ones_like(rendered_metalness_flat) * 0
    rendered_roughness_flat = torch.ones_like(rendered_metalness_flat) * 0
    rendered_subsurfaceness_flat = torch.ones_like(rendered_metalness_flat) * 0
    rrendered_incident_light_flat = torch.ones_like(rendered_metalness_flat) * 0
    rendered_base_color_flat = torch.ones_like(rendered_base_color_flat) *0 

    pbr_combined, extra = render_equation_new(
        rendered_base_color_flat,
        rendered_metalness_flat,
        rendered_roughness_flat,
        rendered_subsurfaceness_flat, 
        rendered_normal_flat,
        rendered_residual_flat,
        light_direction_normalized,
        view_direction_normalized,
        rendered_incident_light_flat,
        debug=debug
    )

    rendered_image = pbr_combined.view(rendered_surface_xyz.shape[1], rendered_surface_xyz.shape[2], 3).permute(2, 0, 1)
    rendered_color_specular = extra["specular"].view(rendered_surface_xyz.shape[1], rendered_surface_xyz.shape[2], 3).permute(2, 0, 1)
    rendered_color_diffuse = extra["diffuse"].view(rendered_surface_xyz.shape[1], rendered_surface_xyz.shape[2], 3).permute(2, 0, 1)
    rendered_pbr = extra["pbr"].view(rendered_surface_xyz.shape[1], rendered_surface_xyz.shape[2], 3).permute(2, 0, 1)

  
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "normals": rendered_normal,
            "pseudo_normals": rendered_pseudo_normal,
            "surface_xyz": rendered_surface_xyz,
            "opacity": rendered_opacity,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "num_rendered": num_rendered,
            "num_contrib": num_contrib,
            "base_color": rendered_base_color,
            "metalness": rendered_metalness,
            "roughness": rendered_roughness,
            "subsurfaceness": rendered_subsurfaceness,
            "pbr": rendered_pbr,
            "residual": rendered_residual,
            "visibility": rendered_visibility,
            "incident_light": rendered_incident_light,
            "color_specular": rendered_color_specular,
            "color_diffuse": rendered_color_diffuse,
            }
