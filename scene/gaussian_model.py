import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.general_utils import rotation_to_quaternion, quaternion_multiply
from utils.sh_utils import RGB2SH, eval_sh
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from arguments import OptimizationParams
from tqdm import tqdm
from bvh import RayTracer
from utils.graphics_utils import fibonacci_sphere_sampling


def sample_incident_rays(normals, is_training=False, sample_num=24):
    if is_training:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=False)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        if self.use_pbr:
            self.base_color_activation = lambda x: torch.sigmoid(x) * 0.77 + 0.03
            self.roughness_activation = lambda x: torch.sigmoid(x) * 0.9 + 0.09
            self.inverse_roughness_activation = lambda y: inverse_sigmoid((y-0.09) / 0.9)
            self.metallic_activation = lambda x: torch.sigmoid(x) * 0.9 + 0.09

    def __init__(self, sh_degree: int, render_type='render'):
        self.render_type = render_type
        self.use_pbr = render_type in ['neilf', 'neilf_deferred', 'neilf_deferred_importance']
        self.active_sh_degree = 3
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._normal = torch.empty(0)  # normal
        self._shs_dc = torch.empty(0)  # output radiance
        self._shs_rest = torch.empty(0)  # output radiance
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._visibility_tracing = None
        self.max_radii2D = torch.empty(0)
        self.weights_accum = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.normal_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.intersection_tracing = False
        # When True, load_ply will derive normals from RGB (normal = normalize(rgb*2-1))
        # instead of using existing nx,ny,nz. This is meant for special conversion cases
        # (e.g. treating 3DGS RGB as normals).
        self.convert_rgb_to_normals = False

        self.setup_functions()
        self.transform = {}
        if self.use_pbr:
            self._base_color = torch.empty(0)
            self._roughness = torch.empty(0)
            self._metallic = torch.empty(0)
            self._incidents_dc = torch.empty(0)
            self._incidents_rest = torch.empty(0)
            self._visibility_dc = torch.empty(0)
            self._visibility_rest = torch.empty(0)
        self.base_color_scale = torch.ones(3, dtype=torch.float, device="cuda")

    @torch.no_grad()
    def set_transform(self, rotation=None, center=None, scale=None, offset=None, transform=None):
        if transform is not None:
            scale = transform[:3, :3].norm(dim=-1)
            self._scaling.data = self.scaling_inverse_activation(self.get_scaling * scale)
            xyz_homo = torch.cat([self._xyz.data, torch.ones_like(self._xyz[:, :1])], dim=-1)
            self._xyz.data = (xyz_homo @ transform.T)[:, :3]
            rotation = transform[:3, :3] / scale[:, None]
            self._normal.data = self._normal.data @ rotation.T
            rotation_q = rotation_to_quaternion(rotation[None])
            self._rotation.data = quaternion_multiply(rotation_q, self._rotation.data)
            return

        if center is not None:
            self._xyz.data = self._xyz.data - center
        if rotation is not None:
            self._xyz.data = (self._xyz.data @ rotation.T)
            self._normal.data = self._normal.data @ rotation.T
            rotation_q = rotation_to_quaternion(rotation[None])
            self._rotation.data = quaternion_multiply(rotation_q, self._rotation.data)
        if scale is not None:
            self._xyz.data = self._xyz.data * scale
            self._scaling.data = self.scaling_inverse_activation(self.get_scaling * scale)
        if offset is not None:
            self._xyz.data = self._xyz.data + offset

    def capture(self):
        captured_list = [
            self.active_sh_degree,
            self._xyz,
            self._normal,
            self._shs_dc,
            self._shs_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.weights_accum,
            self.xyz_gradient_accum,
            self.normal_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        ]
        if self.use_pbr:
            captured_list.extend([
                self._base_color,
                self._roughness,
                self._metallic,
                self._incidents_dc,
                self._incidents_rest,
                self._visibility_dc,
                self._visibility_rest,
            ])

        return captured_list

    def restore(self, model_args, training_args,
                is_training=False, restore_optimizer=True):
        (self.active_sh_degree,
         self._xyz,
         self._normal,
         self._shs_dc,
         self._shs_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         weights_accum,
         xyz_gradient_accum,
         normal_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args[:15]
        if len(model_args) > 15 and self.use_pbr:
            (self._base_color,
             self._roughness,
             self._metallic,
             self._incidents_dc,
             self._incidents_rest,
             self._visibility_dc,
             self._visibility_rest) = model_args[15:]

        if is_training:
            self.training_setup(training_args)
            self.weights_accum = weights_accum
            self.xyz_gradient_accum = xyz_gradient_accum
            self.normal_gradient_accum = normal_gradient_accum
            self.denom = denom
            if restore_optimizer:
                # TODO automatically match the opt_dict
                try:
                    self.optimizer.load_state_dict(opt_dict)
                except:
                    pass
            
            # Try to load MLP data if it exists (this will be handled by load_ply or create_from_ckpt)
            # The MLP data loading is already implemented in those methods

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    def get_normal(self, camera_direction=None):
        if hasattr(self, 'normal_mlp'):
            return self.compute_normal_with_mlp(camera_direction)
        else:
            return self.normal_activation(self._normal)

    @property
    def get_shs(self):
        """SH"""
        shs_dc = self._shs_dc
        shs_rest = self._shs_rest
        return torch.cat((shs_dc, shs_rest), dim=1)

    @property
    def get_incidents(self):
        """SH"""
        incidents_dc = self._incidents_dc
        incidents_rest = self._incidents_rest
        return torch.cat((incidents_dc, incidents_rest), dim=1)

    @property
    def get_visibility(self):
        """SH"""
        visibility_dc = self._visibility_dc
        visibility_rest = self._visibility_rest
        return torch.cat((visibility_dc, visibility_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_base_color(self):
        # Check for combined MLP first
        if hasattr(self, 'use_combined_mlp') and self.use_combined_mlp and hasattr(self, 'combined_mlp'):
            return self.compute_properties_with_combined_mlp()[0]  # Returns base_color
        elif hasattr(self, 'base_color_mlp') and self.base_color_mlp is not None:
            # Use separate MLP to compute base color
            return self.compute_base_color_with_mlp()
        elif self.intersection_tracing:
            # For intersection-traced values, don't apply base_color_scale since values are already in correct color space
            return self._base_color
        else:
            return self.base_color_activation(self._base_color) * self.base_color_scale[None, :]

    @property
    def get_roughness(self):
        # Check for combined MLP first
        if hasattr(self, 'use_combined_mlp') and self.use_combined_mlp and hasattr(self, 'combined_mlp'):
            return self.compute_properties_with_combined_mlp()[1]  # Returns roughness
        elif hasattr(self, 'roughness_mlp') and self.roughness_mlp is not None:
            # Use separate MLP to compute roughness
            return self.compute_roughness_with_mlp()
        elif self.intersection_tracing:
            return self._roughness
        else:
            # Use direct parameter (no MLP)
            return self.roughness_activation(self._roughness)

    @property
    def get_metallic(self):
        # Check for combined MLP first
        if hasattr(self, 'use_combined_mlp') and self.use_combined_mlp and hasattr(self, 'combined_mlp'):
            return self.compute_properties_with_combined_mlp()[2]  # Returns metallic
        elif hasattr(self, 'metallic_mlp') and self.metallic_mlp is not None:
            # Use separate MLP to compute metallic
            return self.compute_metallic_with_mlp()
        elif self.intersection_tracing:
            return self._metallic
        else:
            return self.metallic_activation(self._metallic)
    
    def compute_base_color_with_mlp(self):
        """Compute base color using the MLP"""
        if not hasattr(self, 'base_color_mlp') or not hasattr(self, 'projected_base_colors'):
            # Fallback to direct parameter
            return self.base_color_activation(self._base_color) * self.base_color_scale[None, :]
        
        # Prepare inputs
        positions = self.get_xyz  # [num_gaussians, 3]
        # Use MLP to compute weights (softmax outputs)
        weights = self.base_color_mlp(self.projected_base_colors, positions)  # [num_gaussians, num_train_images, 1]
        
        # Multiply weights with projected base color values and sum
        # self.projected_base_colors: [num_gaussians, num_train_images, 3]
        # weights: [num_gaussians, num_train_images, 1]
        # Result: [num_gaussians, 3]
        base_colors = (weights * self.projected_base_colors).sum(dim=1)
        
        # Apply the base color scale to the MLP output
        base_colors = base_colors * self.base_color_scale[None, :]
        
        return base_colors
    
    def compute_roughness_with_mlp(self):
        """Compute roughness using the MLP"""
        if not hasattr(self, 'roughness_mlp') or not hasattr(self, 'projected_roughness'):
            # Fallback to direct parameter
            return self.roughness_activation(self._roughness)
        
        # Prepare inputs
        positions = self.get_xyz  # [num_gaussians, 3]
        # Use MLP to compute weights (softmax outputs)
        weights = self.roughness_mlp(self.projected_roughness, positions)  # [num_gaussians, num_train_images, 1]
        
        # Multiply weights with projected roughness values and sum
        # self.projected_roughness: [num_gaussians, num_train_images, 1]
        # weights: [num_gaussians, num_train_images, 1]
        # Result: [num_gaussians, 1]
        roughness = (weights * self.projected_roughness).sum(dim=1)
        
        return roughness
    
    def compute_metallic_with_mlp(self):
        """Compute metallic using the MLP"""
        if not hasattr(self, 'metallic_mlp') or not hasattr(self, 'projected_metallic'):
            # Fallback to direct parameter
            return self.metallic_activation(self._metallic)
        
        # Prepare inputs
        positions = self.get_xyz  # [num_gaussians, 3]
        # Use MLP to compute weights (softmax outputs)
        weights = self.metallic_mlp(self.projected_metallic, positions)  # [num_gaussians, num_train_images, 1]
        
        # Multiply weights with projected metallic values and sum
        # self.projected_metallic: [num_gaussians, num_train_images, 1]
        # weights: [num_gaussians, num_train_images, 1]
        # Result: [num_gaussians, 1]
        metallic = (weights * self.projected_metallic).sum(dim=1)
        
        return metallic
    
    def compute_properties_with_combined_mlp(self):
        """Compute base_color, roughness, and metallic using the combined MLP.
        
        Returns:
            Tuple of (base_color, roughness, metallic) tensors
        """
        if not hasattr(self, 'combined_mlp') or not hasattr(self, 'projected_base_colors'):
            # Fallback to direct parameters
            base_color = self.base_color_activation(self._base_color) * self.base_color_scale[None, :]
            roughness = self.roughness_activation(self._roughness)
            metallic = self.metallic_activation(self._metallic)
            return base_color, roughness, metallic
        
        # Check if we've already computed the properties in this forward pass
        # This caching is important since get_base_color, get_roughness, get_metallic
        # may all be called separately, but we only want to run the MLP once
        if hasattr(self, '_cached_combined_mlp_outputs') and self._cached_combined_mlp_outputs is not None:
            return self._cached_combined_mlp_outputs
        
        # Prepare inputs
        positions = self.get_xyz  # [num_gaussians, 3]
        
        # Use combined MLP to compute all weights at once
        base_color_weights, roughness_weights, metallic_weights = self.combined_mlp(
            self.projected_base_colors,
            self.projected_roughness,
            self.projected_metallic,
            positions
        )
        
        # Compute weighted sums
        # base_color: [num_gaussians, 3]
        base_colors = (base_color_weights * self.projected_base_colors).sum(dim=1)
        base_colors = base_colors * self.base_color_scale[None, :]
        
        # roughness: [num_gaussians, 1]
        roughness = (roughness_weights * self.projected_roughness).sum(dim=1)
        
        # metallic: [num_gaussians, 1]
        metallic = (metallic_weights * self.projected_metallic).sum(dim=1)
        
        # Cache the results for this forward pass
        self._cached_combined_mlp_outputs = (base_colors, roughness, metallic)
        
        return base_colors, roughness, metallic
    
    def clear_combined_mlp_cache(self):
        """Clear the cached combined MLP outputs. Should be called at the start of each iteration."""
        self._cached_combined_mlp_outputs = None
    
    def compute_normal_with_mlp(self, camera_direction=None):
        """Compute normal using the MLP"""
        if not hasattr(self, 'normal_mlp') or not hasattr(self, 'projected_normals'):
            # Fallback to direct parameter
            return self.normal_activation(self._normal)
        
        # Get positions of all gaussians
        positions = self.get_xyz  # [num_gaussians, 3]
        
        # Handle camera direction input
        if camera_direction is None:
            # Use a default camera direction (e.g., looking down the Z-axis)
            camera_direction = torch.tensor([0.0, 0.0, -1.0], device=positions.device, dtype=positions.dtype)
            camera_direction = camera_direction.expand(positions.shape[0], -1)  # [num_gaussians, 3]
        elif camera_direction.dim() == 1:
            # Single camera direction provided, expand to match batch size
            camera_direction = camera_direction.expand(positions.shape[0], -1) 
        
        weights = self.normal_mlp(self.projected_normals, positions, camera_direction) 
        
        normals = (weights * self.projected_normals).sum(dim=1)
        
        normals = F.normalize(normals, dim=-1, eps=1e-6)
        
        return normals

    @property
    def get_brdf(self):
        return torch.cat([self.get_base_color, self.get_roughness], dim=-1)

    def get_by_names(self, names):
        if len(names) == 0:
            return None
        fs = []
        for name in names:
            fs.append(getattr(self, "get_" + name))
        return torch.cat(fs, dim=1)

    def split_by_names(self, features, names):
        results = {}
        last_idx = 0
        for name in names:
            current_shape = getattr(self, "_" + name).shape[1]
            results[name] = features[last_idx:last_idx + current_shape]
            last_idx += getattr(self, "_" + name).shape[1]
        return results

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling,
                                          scaling_modifier,
                                          self.get_rotation)

    def get_inverse_covariance(self, scaling_modifier=1):
        return self.covariance_activation(1 / self.get_scaling,
                                          1 / scaling_modifier,
                                          self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    @property
    def attribute_names(self):
        attribute_names = ['xyz', 'normal', 'shs_dc', 'shs_rest', 'scaling', 'rotation', 'opacity']
        if self.use_pbr:
            attribute_names.extend(['base_color', 'roughness', 'metallic',
                                    'incidents_dc', 'incidents_rest',
                                    'visibility_dc', 'visibility_rest'])
        return attribute_names

    def finetune_visibility(self, iterations=1000):
        visibility_sh_lr = 1e-2
        optimizer = torch.optim.Adam([
            {'params': [self._visibility_dc], 'lr': visibility_sh_lr},
            {'params': [self._visibility_rest], 'lr': visibility_sh_lr}
        ])
        means3D = self.get_xyz
        opacity = self.get_opacity[:, 0]
        scaling = self.get_scaling
        rotation = self.get_rotation
        normal = self.get_normal()
        cov_inv = self.get_inverse_covariance()
        tbar = tqdm(range(iterations), desc="Finetuning visibility shs")
        raytracer = RayTracer(means3D, scaling, rotation)
        visibility_shs_view = self.get_visibility.transpose(1, 2)
        vis_sh_degree = np.sqrt(visibility_shs_view.shape[-1]) - 1
        rays_o = means3D
        for iteration in tbar:
            rays_d = torch.randn_like(rays_o)
            rays_d = F.normalize(rays_d, dim=-1)
            mask = (rays_d * normal).sum(-1) < 0
            rays_d[mask] *= -1
            sample_sh2vis = eval_sh(vis_sh_degree, visibility_shs_view, rays_d)
            sample_vis = torch.clamp(sample_sh2vis + 0.5, 0.0, 1.0)
            trace_results = raytracer.trace_visibility(
                rays_o,
                rays_d,
                means3D,
                cov_inv,
                opacity,
                normal)
            visibility = trace_results["visibility"]
            loss = F.l1_loss(visibility, sample_vis)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    @torch.no_grad()
    def update_visibility(self, sample_num):
        raytracer = RayTracer(self.get_xyz, self.get_scaling, self.get_rotation)
        gaussians_xyz = self.get_xyz
        gaussians_inverse_covariance = self.get_inverse_covariance()
        gaussians_opacity = self.get_opacity[:, 0]
        gaussians_normal = self.get_normal()  # Call the method to get the tensor
        incident_visibility_results = []
        incident_dirs_results = []
        incident_areas_results = []
        chunk_size = gaussians_xyz.shape[0] // ((sample_num - 1) // 24 + 1)
        for offset in tqdm(range(0, gaussians_xyz.shape[0], chunk_size), "Update visibility with raytracing."):
            incident_dirs, incident_areas = sample_incident_rays(gaussians_normal[offset:offset + chunk_size], False,
                                                    sample_num)
            trace_results = raytracer.trace_visibility(
                gaussians_xyz[offset:offset + chunk_size, None].expand_as(incident_dirs),
                incident_dirs,
                gaussians_xyz,
                gaussians_inverse_covariance,
                gaussians_opacity,
                gaussians_normal)
            incident_visibility = trace_results["visibility"]
            incident_visibility_results.append(incident_visibility)
            incident_dirs_results.append(incident_dirs)
            incident_areas_results.append(incident_areas)
        incident_visibility_result = torch.cat(incident_visibility_results, dim=0)
        incident_dirs_result = torch.cat(incident_dirs_results, dim=0)
        incident_areas_result = torch.cat(incident_areas_results, dim=0)
        self._visibility_tracing = incident_visibility_result
        self._incident_dirs = incident_dirs_result
        self._incident_areas = incident_areas_result

    @classmethod
    def create_from_gaussians(cls, gaussians_list, dataset):
        assert len(gaussians_list) > 0
        sh_degree = max(g.max_sh_degree for g in gaussians_list)
        gaussians = GaussianModel(sh_degree=sh_degree,
                                  render_type=gaussians_list[0].render_type)
        
        # Copy configuration flags from the first gaussian
        first_gaussian = gaussians_list[0]
        gaussians.intersection_tracing = getattr(first_gaussian, 'intersection_tracing', False)
        gaussians.perform_intersection_tracing = getattr(first_gaussian, 'perform_intersection_tracing', False)
        gaussians.load_intersection_data = getattr(first_gaussian, 'load_intersection_data', False)
        
        # Copy MLP-related attributes if they exist
        if hasattr(first_gaussian, 'base_color_mlp'):
            gaussians.base_color_mlp = first_gaussian.base_color_mlp
            gaussians.roughness_mlp = first_gaussian.roughness_mlp
            gaussians.metallic_mlp = first_gaussian.metallic_mlp
            if hasattr(first_gaussian, 'projected_base_colors'):
                gaussians.projected_base_colors = first_gaussian.projected_base_colors
            if hasattr(first_gaussian, 'projected_roughness'):
                gaussians.projected_roughness = first_gaussian.projected_roughness
            if hasattr(first_gaussian, 'projected_metallic'):
                gaussians.projected_metallic = first_gaussian.projected_metallic
            if hasattr(first_gaussian, 'normal_mlp'):
                gaussians.normal_mlp = first_gaussian.normal_mlp
            if hasattr(first_gaussian, 'projected_normals'):
                gaussians.projected_normals = first_gaussian.projected_normals
        
        attribute_names = gaussians.attribute_names
        for attribute_name in attribute_names:
            setattr(gaussians, "_" + attribute_name,
                    nn.Parameter(torch.cat([getattr(g, "_" + attribute_name).data for g in gaussians_list],
                                           dim=0).requires_grad_(True)))

        return gaussians

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path, weights_only=False)

        (self.active_sh_degree,
         self._xyz,
         self._normal,
         self._shs_dc,
         self._shs_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         weights_accum,
         xyz_gradient_accum,
         normal_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args[:15]

        self.weights_accum = weights_accum
        self.normal_gradient_accum = normal_gradient_accum
        self.denom = denom

        if self.use_pbr:
            if len(model_args) > 15:
                # Handle both old (6 parameters) and new (7 parameters) checkpoint formats
                pbr_params = model_args[15:]
                if len(pbr_params) >= 7:
                    # New format with metallic
                    (self._base_color,
                     self._roughness,
                     self._metallic,
                     self._incidents_dc,
                     self._incidents_rest,
                     self._visibility_dc,
                     self._visibility_rest) = pbr_params[:7]
                else:
                    # Old format without metallic - initialize metallic to zeros
                    (self._base_color,
                     self._roughness,
                     self._incidents_dc,
                     self._incidents_rest,
                     self._visibility_dc,
                     self._visibility_rest) = pbr_params[:6]
                    # Initialize metallic as zeros
                    metallic = torch.zeros_like(self._xyz[..., :1])
                    self._metallic = nn.Parameter(metallic.requires_grad_(True))
            else:
                # Initialize all PBR parameters
                self._base_color = nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True))
                roughness = torch.zeros_like(self._xyz[..., :1])
                metallic = torch.zeros_like(self._xyz[..., :1])
                self._roughness = nn.Parameter(roughness.requires_grad_(True))
                self._metallic = nn.Parameter(metallic.requires_grad_(True))
                incidents = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()

                self._incidents_dc = nn.Parameter(
                    incidents[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
                self._incidents_rest = nn.Parameter(
                    incidents[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

                visibility = torch.zeros((self._xyz.shape[0], 1, 4 ** 2)).float().cuda()
                self._visibility_dc = nn.Parameter(
                    visibility[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
                self._visibility_rest = nn.Parameter(
                    visibility[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        if restore_optimizer:
            # TODO automatically match the opt_dict
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        # Try to load MLP data from checkpoint directory (only if MLPs are enabled)
        # Note: MLP loading is controlled by the --use_mlp flag in train.py
        # This method will be called from train.py which has access to the args
        # For now, we'll skip MLP loading here and let train.py handle it
        pass

        return first_iter

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_normal = torch.tensor(np.asarray(pcd.normals)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        shs = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        shs[:, :3, 0] = RGB2SH(fused_color)
        shs[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._normal = nn.Parameter(fused_normal.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._shs_dc = nn.Parameter(shs[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._shs_rest = nn.Parameter(shs[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.use_pbr:
            base_color = torch.zeros_like(fused_point_cloud)
            roughness = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
            
            try:
                metallic = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
            except (KeyError, ValueError):
                print("Warning: 'metallic' field not found in PLY file. Initializing to zeros.")
                metallic = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

            self._base_color = nn.Parameter(base_color.requires_grad_(True))
            self._roughness = nn.Parameter(roughness.requires_grad_(True))
            self._metallic = nn.Parameter(metallic.requires_grad_(True))

            incidents = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            self._incidents_dc = nn.Parameter(incidents[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._incidents_rest = nn.Parameter(incidents[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

            visibility = torch.zeros((self._xyz.shape[0], 1, 4 ** 2)).float().cuda()
            self._visibility_dc = nn.Parameter(visibility[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._visibility_rest = nn.Parameter(visibility[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

    def training_setup(self, training_args: OptimizationParams):
        # Persist freeze flag to control autograd and replacements later
        try:
            self.freeze_geometry_flag = bool(getattr(training_args, 'freeze_geometry', False))
        except Exception:
            self.freeze_geometry_flag = False

        self.percent_dense = training_args.percent_dense
        self.weights_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # If freezing geometry, disable grads for xyz/rotation/scaling (keep opacity trainable)
        if getattr(self, 'freeze_geometry_flag', False):
            try:
                self._xyz.requires_grad_(False)
                self._rotation.requires_grad_(False)
                self._scaling.requires_grad_(False)
                # keep opacity trainable to avoid attenuating PBR gradients
                self._normal.requires_grad_(False)
            except Exception:
                pass

        l = []
        # Geometry groups: when freezing geometry, still optimize opacity
        if not getattr(self, 'freeze_geometry_flag', False):
            l.extend([
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            ])
        else:
            l.extend([
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            ])

        # Non-geometry groups
        if not getattr(self, 'freeze_geometry_flag', False):
            l.append({'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"})
        else:
            try:
                self._normal.requires_grad_(False)
            except Exception:
                pass
        l.extend([
            {'params': [self._shs_dc], 'lr': training_args.sh_lr, "name": "f_dc"},
            {'params': [self._shs_rest], 'lr': training_args.sh_lr / 20.0, "name": "f_rest"}
        ])

        if self.use_pbr:
            if training_args.light_rest_lr < 0:
                training_args.light_rest_lr = training_args.light_lr / 20.0
            if training_args.visibility_rest_lr < 0:
                training_args.visibility_rest_lr = training_args.visibility_lr / 20.0

            l.extend([
                {'params': [self._base_color], 'lr': training_args.base_color_lr, "name": "base_color"},
                {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
                {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},
                {'params': [self._incidents_dc], 'lr': training_args.light_lr, "name": "incidents_dc"},
                {'params': [self._incidents_rest], 'lr': training_args.light_rest_lr, "name": "incidents_rest"},
                {'params': [self._visibility_dc], 'lr': training_args.visibility_lr, "name": "visibility_dc"},
                {'params': [self._visibility_rest], 'lr': training_args.visibility_rest_lr, "name": "visibility_rest"},
            ])
            
            # Add MLP parameters to optimizer if they exist
            # Check if using combined MLP or separate MLPs
            if hasattr(self, 'use_combined_mlp') and self.use_combined_mlp and hasattr(self, 'combined_mlp'):
                # Use combined MLP - single set of parameters
                combined_lr = training_args.base_color_lr  # Use base_color_lr as the learning rate for combined MLP
                mlp_params = [
                    {'params': self.combined_mlp.parameters(), 'lr': combined_lr, "name": "combined_mlp"},
                ]
                l.extend(mlp_params)
            elif hasattr(self, 'base_color_mlp') and self.base_color_mlp is not None:
                # Use separate MLPs
                mlp_params = [
                    {'params': self.base_color_mlp.parameters(), 'lr': training_args.base_color_lr, "name": "base_color_mlp"},
                ]
                if hasattr(self, 'roughness_mlp') and self.roughness_mlp is not None:
                    mlp_params.append({'params': self.roughness_mlp.parameters(), 'lr': training_args.roughness_lr, "name": "roughness_mlp"})
                if hasattr(self, 'metallic_mlp') and self.metallic_mlp is not None:
                    mlp_params.append({'params': self.metallic_mlp.parameters(), 'lr': training_args.metallic_lr, "name": "metallic_mlp"})
                l.extend(mlp_params)
            
            # Normal MLP is always separate
            if hasattr(self, 'normal_mlp'):
                l.extend([
                    {'params': self.normal_mlp.parameters(), 'lr': training_args.normal_lr, "name": "normal_mlp"},
                ])
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        # Add ExponentialLR scheduler for MLPs if they exist
        if hasattr(self, 'base_color_mlp'):
            self.mlp_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9999)
        else:
            self.mlp_scheduler = None

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Step the MLP scheduler if it exists
        if hasattr(self, 'mlp_scheduler') and self.mlp_scheduler is not None:
            self.mlp_scheduler.step()

    def update_learning_rate(self, iteration):
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._shs_dc.shape[1] * self._shs_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._shs_rest.shape[1] * self._shs_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.use_pbr:
            for i in range(self._base_color.shape[1]):
                l.append('base_color_{}'.format(i))
            l.append('roughness')
            for i in range(self._metallic.shape[1]):
                l.append('metallic_{}'.format(i))
            for i in range(self._incidents_dc.shape[1] * self._incidents_dc.shape[2]):
                l.append('incidents_dc_{}'.format(i))
            for i in range(self._incidents_rest.shape[1] * self._incidents_rest.shape[2]):
                l.append('incidents_rest_{}'.format(i))
            for i in range(self._visibility_dc.shape[1] * self._visibility_dc.shape[2]):
                l.append('visibility_dc_{}'.format(i))
            for i in range(self._visibility_rest.shape[1] * self._visibility_rest.shape[2]):
                l.append('visibility_rest_{}'.format(i))

        return l

    @torch.no_grad()
    def bake_mlp_outputs_to_parameters(self):
        """
        Query MLPs for all Gaussians, assign outputs to base_color, roughness, and metallic parameters,
        then remove MLPs and projected data.
        """
        if not (hasattr(self, 'base_color_mlp') and hasattr(self, 'projected_base_colors')):
            # No MLPs to bake
            return
        
        print("Baking MLP outputs to parameters...")
        
        # Query MLPs for all Gaussians
        num_gaussians = self.get_xyz.shape[0]
        
        # Base color
        base_colors = self.compute_base_color_with_mlp()  # [num_gaussians, 3]
        # Remove base_color_scale to get raw activated values
        base_colors_raw = base_colors / self.base_color_scale[None, :]
        # Convert to raw parameters using inverse activation
        # base_color_activation: sigmoid(x) * 0.77 + 0.03
        # Inverse: inverse_sigmoid((y - 0.03) / 0.77)
        base_colors_param = inverse_sigmoid((base_colors_raw - 0.03) / 0.77)
        # Ensure shape matches: _base_color should be [num_gaussians, 3] or [num_gaussians, 1]
        if self._base_color.shape != base_colors_param.shape:
            # If _base_color is [num_gaussians, 1], take mean across channels
            if self._base_color.shape[1] == 1 and base_colors_param.shape[1] == 3:
                base_colors_param = base_colors_param.mean(dim=1, keepdim=True)
            # If _base_color is [num_gaussians, 3] but we have [num_gaussians, 1], expand
            elif self._base_color.shape[1] == 3 and base_colors_param.shape[1] == 1:
                base_colors_param = base_colors_param.repeat(1, 3)
        self._base_color.data = base_colors_param
        
        # Roughness
        roughness = self.compute_roughness_with_mlp()  # [num_gaussians, 1]
        # Convert to raw parameters using inverse activation
        # roughness_activation: sigmoid(x) * 0.9 + 0.09
        # Inverse: inverse_sigmoid((y - 0.09) / 0.9)
        roughness_param = self.inverse_roughness_activation(roughness)
        self._roughness.data = roughness_param
        
        # Metallic
        metallic = self.compute_metallic_with_mlp()  # [num_gaussians, 1]
        # Convert to raw parameters using inverse activation
        # metallic_activation: sigmoid(x) * 0.9 + 0.09
        # Inverse: inverse_sigmoid((y - 0.09) / 0.9)
        metallic_param = inverse_sigmoid((metallic - 0.09) / 0.9)
        # Ensure shape matches: _metallic should be [num_gaussians, 1] or [num_gaussians, N]
        if self._metallic.shape != metallic_param.shape:
            # If _metallic has more channels, take mean
            if self._metallic.shape[1] > 1 and metallic_param.shape[1] == 1:
                metallic_param = metallic_param.repeat(1, self._metallic.shape[1])
            # If _metallic has 1 channel but we have more, take mean
            elif self._metallic.shape[1] == 1 and metallic_param.shape[1] > 1:
                metallic_param = metallic_param.mean(dim=1, keepdim=True)
        self._metallic.data = metallic_param
        
        print(f"✓ Baked MLP outputs to parameters:")
        print(f"  - Base color: {base_colors_param.shape}")
        print(f"  - Roughness: {roughness_param.shape}")
        print(f"  - Metallic: {metallic_param.shape}")
        
        # Remove MLPs and projected data
        if hasattr(self, 'base_color_mlp'):
            del self.base_color_mlp
        if hasattr(self, 'roughness_mlp'):
            del self.roughness_mlp
        if hasattr(self, 'metallic_mlp'):
            del self.metallic_mlp
        if hasattr(self, 'normal_mlp'):
            del self.normal_mlp
        if hasattr(self, 'projected_base_colors'):
            del self.projected_base_colors
        if hasattr(self, 'projected_roughness'):
            del self.projected_roughness
        if hasattr(self, 'projected_metallic'):
            del self.projected_metallic
        if hasattr(self, 'projected_normals'):
            del self.projected_normals
        
        print("✓ Removed MLPs and projected data")

    def save_ply(self, path, current_iteration=None, final_iteration=None):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normal = self._normal.detach().cpu().numpy()
        sh_dc = self._shs_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        sh_rest = self._shs_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        attributes_list = [xyz, normal, sh_dc, sh_rest, opacities, scale, rotation]
        if self.use_pbr:
            attributes_list.extend([
                self._base_color.detach().cpu().numpy(),
                self._roughness.detach().cpu().numpy(),
                self._metallic.detach().cpu().numpy(),
                self._incidents_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
                self._incidents_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
                self._visibility_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
                self._visibility_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
            ])

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attributes_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        # Save model configuration flags as a separate metadata file
        metadata_path = path.replace('.ply', '_metadata.json')
        import json
        
        # Determine if we have MLPs (either combined or separate)
        has_mlp = (hasattr(self, 'projected_base_colors') and 
                   (hasattr(self, 'combined_mlp') or 
                    (hasattr(self, 'base_color_mlp') and self.base_color_mlp is not None)))
        use_combined_mlp = getattr(self, 'use_combined_mlp', False)
        
        metadata = {
            'intersection_tracing': getattr(self, 'intersection_tracing', False),
            'use_pbr': self.use_pbr,
            'render_type': self.render_type,
            'use_mlp': has_mlp,
            'use_combined_mlp': use_combined_mlp,
            'mlps_baked': False,  # MLPs are not baked into parameters
            'perform_intersection_tracing': getattr(self, 'perform_intersection_tracing', False),
            'load_intersection_data': getattr(self, 'load_intersection_data', False)
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved model configuration metadata to {metadata_path}")
        
        # Save MLP data if MLPs exist (MLPs are not baked into parameters)
        if has_mlp and hasattr(self, 'projected_base_colors') and hasattr(self, 'projected_roughness') and hasattr(self, 'projected_metallic'):
            # Create a separate directory for MLP data
            mlp_data_dir = os.path.join(os.path.dirname(path), "mlp_data")
            mkdir_p(mlp_data_dir)
            
            # Save projected data as numpy arrays
            np.save(os.path.join(mlp_data_dir, "projected_base_colors.npy"), self.projected_base_colors.detach().cpu().numpy())
            np.save(os.path.join(mlp_data_dir, "projected_roughness.npy"), self.projected_roughness.detach().cpu().numpy())
            np.save(os.path.join(mlp_data_dir, "projected_metallic.npy"), self.projected_metallic.detach().cpu().numpy())
            # Only save projected_normals if normal_mlp is present
            if hasattr(self, 'normal_mlp') and hasattr(self, 'projected_normals'):
                np.save(os.path.join(mlp_data_dir, "projected_normals.npy"), self.projected_normals.detach().cpu().numpy())
                torch.save(self.normal_mlp.state_dict(), os.path.join(mlp_data_dir, "normal_mlp_weights.pth"))
            
            # Save MLP weights based on whether using combined or separate MLPs
            if use_combined_mlp and hasattr(self, 'combined_mlp'):
                torch.save(self.combined_mlp.state_dict(), os.path.join(mlp_data_dir, "combined_mlp_weights.pth"))
                print(f"  - Saved combined MLP weights")
            else:
                # Save separate MLP weights
                if hasattr(self, 'base_color_mlp') and self.base_color_mlp is not None:
                    torch.save(self.base_color_mlp.state_dict(), os.path.join(mlp_data_dir, "base_color_mlp_weights.pth"))
                if hasattr(self, 'roughness_mlp') and self.roughness_mlp is not None:
                    torch.save(self.roughness_mlp.state_dict(), os.path.join(mlp_data_dir, "roughness_mlp_weights.pth"))
                if hasattr(self, 'metallic_mlp') and self.metallic_mlp is not None:
                    torch.save(self.metallic_mlp.state_dict(), os.path.join(mlp_data_dir, "metallic_mlp_weights.pth"))
                print(f"  - Saved separate MLP weights")

            # Save MLP metadata
            mlp_metadata = {
                'num_train_images': self.projected_base_colors.shape[1],
                'net_width': 64,  # Default value, could be made configurable
                'has_mlps': True,
                'use_combined_mlp': use_combined_mlp,
                'has_normal_mlp': hasattr(self, 'normal_mlp'),
                'conditioning': 'position',
                'id_embedding_dim': None,
                'num_gaussians': int(self.get_xyz.shape[0])
            }
            import json
            with open(os.path.join(mlp_data_dir, "mlp_metadata.json"), 'w') as f:
                json.dump(mlp_metadata, f)
            
            print(f"✓ MLP projected data and weights saved to {mlp_data_dir}")

    def reset_opacity(self):
        # Skip opacity reset when geometry is frozen
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        # Load model configuration metadata if it exists
        metadata_path = path.replace('.ply', '_metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load all configuration flags
            self.intersection_tracing = metadata.get('intersection_tracing', False)
            self.perform_intersection_tracing = metadata.get('perform_intersection_tracing', False)
            self.load_intersection_data = metadata.get('load_intersection_data', False)
            
            print(f"✓ Loaded model configuration from {metadata_path}")
            print(f"  - intersection_tracing: {self.intersection_tracing}")
            print(f"  - use_mlp: {metadata.get('use_mlp', False)}")
            print(f"  - perform_intersection_tracing: {self.perform_intersection_tracing}")
            print(f"  - load_intersection_data: {self.load_intersection_data}")
        else:
            # Set default values if no metadata exists
            self.intersection_tracing = False
            self.perform_intersection_tracing = False
            self.load_intersection_data = False
            print("No model configuration metadata found, using default values")

        vertex = plydata.elements[0]
        prop_names = [p.name for p in vertex.properties]

        xyz = np.stack((np.asarray(vertex["x"]),
                        np.asarray(vertex["y"]),
                        np.asarray(vertex["z"])), axis=1)

        # Normals:
        # When convert_rgb_to_normals: derive normals from color (RGB or SH0) as normalize(rgb*2-1).
        # Otherwise: use nx,ny,nz if present, else SH0-derived, else zeros.
        normal = None
        convert_rgb = getattr(self, "convert_rgb_to_normals", False)

        def _rgb_to_normal(rgb: np.ndarray) -> np.ndarray:
            """normal = normalize(rgb*2 - 1)"""
            n = rgb.astype(np.float32) * 2.0 - 1.0
            norm = np.linalg.norm(n, axis=1, keepdims=True)
            norm = np.maximum(norm, 1e-8)
            return (n / norm).astype(np.float32)

        # Case 1: convert_rgb_to_normals + explicit red/green/blue
        if convert_rgb and all(c in prop_names for c in ("red", "green", "blue")):
            rgb = np.stack((np.asarray(vertex["red"]),
                            np.asarray(vertex["green"]),
                            np.asarray(vertex["blue"])), axis=1)
            if np.issubdtype(rgb.dtype, np.integer):
                rgb = rgb.astype(np.float32) / 255.0
            normal = _rgb_to_normal(rgb)

        # Case 2: convert_rgb_to_normals + SH0 (f_dc_0..2) — 3DGS PLY stores color as SH, not RGB
        if normal is None and convert_rgb and all(nm in prop_names for nm in ("f_dc_0", "f_dc_1", "f_dc_2")):
            f_dc0 = np.asarray(vertex["f_dc_0"])
            f_dc1 = np.asarray(vertex["f_dc_1"])
            f_dc2 = np.asarray(vertex["f_dc_2"])
            v = np.stack([f_dc0, f_dc1, f_dc2], axis=1).astype(np.float32)
            SH_C0 = 0.28209479177387814
            rgb = np.maximum(SH_C0 * v + 0.5, 0.0)
            normal = _rgb_to_normal(rgb)

        # Case 3: use stored nx,ny,nz (only when not in convert_rgb mode, or as fallback)
        if normal is None and all(nm in prop_names for nm in ("nx", "ny", "nz")):
            normal = np.stack((np.asarray(vertex["nx"]),
                               np.asarray(vertex["ny"]),
                               np.asarray(vertex["nz"])), axis=1).astype(np.float32)
            # If stored normals are all ~0, treat as missing so we can fall back to SH0
            if np.mean(np.linalg.norm(normal, axis=1)) < 1e-3:
                normal = None

        # Case 4: derive from SH0 when normals absent (generic fallback)
        if normal is None and all(nm in prop_names for nm in ("f_dc_0", "f_dc_1", "f_dc_2")):
            f_dc0 = np.asarray(vertex["f_dc_0"])
            f_dc1 = np.asarray(vertex["f_dc_1"])
            f_dc2 = np.asarray(vertex["f_dc_2"])
            v = np.stack([f_dc0, f_dc1, f_dc2], axis=1).astype(np.float32)
            SH_C0 = 0.28209479177387814
            rgb = np.maximum(SH_C0 * v + 0.5, 0.0)
            normal = _rgb_to_normal(rgb)

        # Case 5: final fallback
        if normal is None:
            normal = np.zeros((xyz.shape[0], 3), dtype=np.float32)

        # Opacity: accept "opacity" (this repo), "opacities", or "opacity_0" (some 3DGS/r3dgs exports)
        opacity_key = None
        for candidate in ("opacity", "opacities", "opacity_0"):
            if candidate in prop_names:
                opacity_key = candidate
                break
        if opacity_key is None:
            raise ValueError(
                f"PLY has no opacity field (tried 'opacity', 'opacities', 'opacity_0'). "
                f"Available properties: {prop_names}"
            )
        opacities = np.asarray(vertex[opacity_key])[..., np.newaxis]

        shs_dc = np.zeros((xyz.shape[0], 3, 1))
        shs_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        count_rest = len(extra_f_names)
        expected_rest = 3 * (self.max_sh_degree + 1) ** 2 - 3
        # If mismatch, try to infer degree from PLY and adjust
        if count_rest != expected_rest:
            if count_rest > 0:
                val = count_rest / 3.0 + 1.0
                d_inf = int(round(np.sqrt(val) - 1.0))
                if 3 * ((d_inf + 1) ** 2) - 3 == count_rest:
                    self.max_sh_degree = d_inf
                    expected_rest = count_rest
                else:
                    # keep expected_rest, will pad/truncate
                    pass
            else:
                # No rest coeffs present; set expected to 0 for deg 0
                expected_rest = 0

        # Read available rest coeffs
        tmp_rest = np.zeros((xyz.shape[0], count_rest), dtype=np.float32)
        for idx, attr_name in enumerate(extra_f_names):
            tmp_rest[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Prepare final rest tensor with expected size (may pad/truncate)
        coeffs_per_channel = (self.max_sh_degree + 1) ** 2 - 1
        target_cols = 3 * coeffs_per_channel
        final_rest_flat = np.zeros((xyz.shape[0], max(target_cols, 0)), dtype=np.float32)
        if target_cols > 0 and count_rest > 0:
            cols = min(target_cols, count_rest)
            final_rest_flat[:, :cols] = tmp_rest[:, :cols]
        # Reshape to (P,3,coeffs_per_channel)
        shs_extra = final_rest_flat.reshape((xyz.shape[0], 3, max(coeffs_per_channel, 0)))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._shs_dc = nn.Parameter(torch.tensor(
            shs_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._shs_rest = nn.Parameter(torch.tensor(
            shs_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        if self.use_pbr:
            base_color_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("base_color")]
            base_color_names = sorted(base_color_names, key=lambda x: int(x.split('_')[-1]))
            base_color = np.zeros((xyz.shape[0], len(base_color_names)))
            for idx, attr_name in enumerate(base_color_names):
                base_color[:, idx] = np.asarray(plydata.elements[0][attr_name])

            roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
            
            # Load metallic field - handle both old ("metallic") and new ("metallic_0") naming conventions
            metallic_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("metallic")]
            metallic_names = sorted(metallic_names, key=lambda x: int(x.split('_')[-1]))
            if len(metallic_names) > 0:
                metallic = np.zeros((xyz.shape[0], len(metallic_names)))
                for idx, attr_name in enumerate(metallic_names):
                    metallic[:, idx] = np.asarray(plydata.elements[0][attr_name])
            else:
                # Try the old naming convention (single "metallic" field)
                try:
                    metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]
                except (KeyError, ValueError):
                    print("Warning: 'metallic' field not found in PLY file. Initializing to zeros.")
                    metallic = np.zeros((xyz.shape[0], 1))

            self._base_color = nn.Parameter(
                torch.tensor(base_color, dtype=torch.float, device="cuda").requires_grad_(True))
            self._roughness = nn.Parameter(
                torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
            self._metallic = nn.Parameter(
                torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))

            # Incidents SH: be robust to missing fields
            props = [p.name for p in plydata.elements[0].properties]
            N = xyz.shape[0]

            # Defaults
            incidents_dc = np.zeros((N, 3, 1), dtype=np.float32)
            incidents_rest = np.zeros((N, 3, (self.max_sh_degree + 1) ** 2 - 1), dtype=np.float32)

            # DC components if present
            dc_ok = all(n in props for n in ["incidents_dc_0", "incidents_dc_1", "incidents_dc_2"])
            if dc_ok:
                incidents_dc[:, 0, 0] = np.asarray(plydata.elements[0]["incidents_dc_0"])
                incidents_dc[:, 1, 0] = np.asarray(plydata.elements[0]["incidents_dc_1"])
                incidents_dc[:, 2, 0] = np.asarray(plydata.elements[0]["incidents_dc_2"])

            # REST components if present and count matches expectation
            extra_incidents_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("incidents_rest_")]
            extra_incidents_names = sorted(extra_incidents_names, key=lambda x: int(x.split('_')[-1]))
            expected_inc_rest = 3 * (self.max_sh_degree + 1) ** 2 - 3
            if len(extra_incidents_names) == expected_inc_rest and expected_inc_rest > 0:
                incidents_extra_flat = np.zeros((N, len(extra_incidents_names)), dtype=np.float32)
                for idx, attr_name in enumerate(extra_incidents_names):
                    incidents_extra_flat[:, idx] = np.asarray(plydata.elements[0][attr_name])
                incidents_rest = incidents_extra_flat.reshape((N, 3, (self.max_sh_degree + 1) ** 2 - 1))

            self._incidents_dc = nn.Parameter(torch.tensor(incidents_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._incidents_rest = nn.Parameter(torch.tensor(incidents_rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

            # Visibility SH (degree 3 -> 4**2 coeffs): be robust to missing fields
            visibility_dc = np.zeros((N, 1, 1), dtype=np.float32)
            if "visibility_dc_0" in props:
                visibility_dc[:, 0, 0] = np.asarray(plydata.elements[0]["visibility_dc_0"])

            extra_visibility_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("visibility_rest_")]
            extra_visibility_names = sorted(extra_visibility_names, key=lambda x: int(x.split('_')[-1]))
            expected_vis_rest = 4 ** 2 - 1
            visibility_rest = np.zeros((N, 1, expected_vis_rest), dtype=np.float32)
            if len(extra_visibility_names) == expected_vis_rest and expected_vis_rest > 0:
                vis_extra_flat = np.zeros((N, len(extra_visibility_names)), dtype=np.float32)
                for idx, attr_name in enumerate(extra_visibility_names):
                    vis_extra_flat[:, idx] = np.asarray(plydata.elements[0][attr_name])
                visibility_rest = vis_extra_flat.reshape((N, 1, expected_vis_rest))

            self._visibility_dc = nn.Parameter(torch.tensor(visibility_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._visibility_rest = nn.Parameter(torch.tensor(visibility_rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        # Try to load MLP projected data and weights if they exist and use_mlp flag is set
        if hasattr(self, 'use_mlp') and self.use_mlp:
            mlp_data_dir = os.path.join(os.path.dirname(path), "mlp_data")
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
                        projected_normals = np.load(os.path.join(mlp_data_dir, "projected_normals.npy"))
                        
                        # Convert to tensors and store (not trainable)
                        self.projected_base_colors = torch.tensor(projected_base_colors, dtype=torch.float, device="cuda", requires_grad=False)
                        self.projected_roughness = torch.tensor(projected_roughness, dtype=torch.float, device="cuda", requires_grad=False)
                        self.projected_metallic = torch.tensor(projected_metallic, dtype=torch.float, device="cuda", requires_grad=False)
                        self.projected_normals = torch.tensor(projected_normals, dtype=torch.float, device="cuda", requires_grad=False)
                        
                        # Load normal MLP data if it exists
                        if mlp_metadata.get('has_normal_mlp', False):
                            try:
                                projected_normals = np.load(os.path.join(mlp_data_dir, "projected_normals.npy"))
                                self.projected_normals = torch.tensor(projected_normals, dtype=torch.float, device="cuda", requires_grad=False)
                                print(f"  - Normal MLP data loaded, shape: {self.projected_normals.shape}")
                            except Exception as e:
                                print(f"Warning: Failed to load normal MLP data: {e}")
                        
                        # Load MLP weights (MLPs will be initialized in train.py)
                        self.mlp_weights_path = mlp_data_dir
                        
                        print(f"✓ MLP projected data and weights path loaded from {mlp_data_dir}")
                        print(f"  - Projected data shapes: {self.projected_base_colors.shape}, {self.projected_roughness.shape}, {self.projected_metallic.shape}")
                        
                except Exception as e:
                    print(f"Warning: Failed to load MLP data from {mlp_data_dir}: {e}")
                    print("Model will be loaded without MLPs")
        else:
            # MLPs not enabled, skip loading
            pass

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                # Respect freeze flag for geometry params
                req = True
                if getattr(self, 'freeze_geometry_flag', False) and name in ["xyz", "rotation", "scaling", "opacity"]:
                    req = False
                group["params"][0] = nn.Parameter(tensor.requires_grad_(req))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        # MLP parameters don't need pruning since they're network weights, not per-gaussian parameters
        mlp_param_names = ["base_color_mlp", "roughness_mlp", "metallic_mlp", "normal_mlp", "combined_mlp"]
        
        for group in self.optimizer.param_groups:
            # Skip MLP parameters - they don't need pruning since they don't correspond to individual Gaussians
            if group["name"] in mlp_param_names:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue
                
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                # Respect freeze flag for geometry params
                req = True
                if getattr(self, 'freeze_geometry_flag', False) and group["name"] in ["xyz", "rotation", "scaling", "opacity"]:
                    req = False
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(req)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # Only apply mask to non-MLP parameters (MLPs have network weights, not per-gaussian parameters)
                mlp_param_names = ["base_color_mlp", "roughness_mlp", "metallic_mlp", "normal_mlp", "combined_mlp"]
                if group["name"] not in mlp_param_names:
                    req = True
                    if getattr(self, 'freeze_geometry_flag', False) and group["name"] in ["xyz", "rotation", "scaling", "opacity"]:
                        req = False
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(req))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        # Ensure mask is boolean and on same device as model tensors
        device = self._xyz.device
        if mask.dtype is not torch.bool:
            mask = mask.to(torch.bool)
        if mask.device != device:
            mask = mask.to(device)

        valid_points_mask = ~mask

        # Ensure auxiliary buffers exist and have the pre-prune size before masking
        num_points_before = valid_points_mask.shape[0]
        if self.weights_accum.numel() == 0 or self.weights_accum.shape[0] != num_points_before:
            self.weights_accum = torch.zeros((num_points_before, 1), device=device)
        if self.xyz_gradient_accum.numel() == 0 or self.xyz_gradient_accum.shape[0] != num_points_before:
            self.xyz_gradient_accum = torch.zeros((num_points_before, 1), device=device)
        if self.normal_gradient_accum.numel() == 0 or self.normal_gradient_accum.shape[0] != num_points_before:
            self.normal_gradient_accum = torch.zeros((num_points_before, 1), device=device)
        if self.denom.numel() == 0 or self.denom.shape[0] != num_points_before:
            self.denom = torch.zeros((num_points_before, 1), device=device)
        if self.max_radii2D.numel() == 0 or self.max_radii2D.shape[0] != num_points_before:
            self.max_radii2D = torch.zeros((num_points_before), device=device)

        # Prune optimizer-backed tensors first
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._shs_dc = optimizable_tensors["f_dc"]
        self._shs_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Now prune auxiliary buffers using the same mask
        self.weights_accum = self.weights_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.normal_gradient_accum = self.normal_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.use_pbr:
            self._base_color = optimizable_tensors["base_color"]
            self._roughness = optimizable_tensors["roughness"]
            self._metallic = optimizable_tensors["metallic"]
            self._incidents_dc = optimizable_tensors["incidents_dc"]
            self._incidents_rest = optimizable_tensors["incidents_rest"]
            self._visibility_dc = optimizable_tensors["visibility_dc"]
            self._visibility_rest = optimizable_tensors["visibility_rest"]
            # Prune ID embeddings (deprecated, no-op)
            try:
                if hasattr(self, 'base_color_mlp'):
                    self.base_color_mlp.prune_embeddings(valid_points_mask)
                if hasattr(self, 'roughness_mlp'):
                    self.roughness_mlp.prune_embeddings(valid_points_mask)
                if hasattr(self, 'metallic_mlp'):
                    self.metallic_mlp.prune_embeddings(valid_points_mask)
                if hasattr(self, 'normal_mlp'):
                    self.normal_mlp.prune_embeddings(valid_points_mask)
            except Exception as e:
                print(f"Warning: Failed to prune ID embeddings: {e}")

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        # MLP parameters don't need to be concatenated since they're network weights, not per-gaussian parameters
        mlp_param_names = ["base_color_mlp", "roughness_mlp", "metallic_mlp", "normal_mlp", "combined_mlp"]
        
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            
            # Skip MLP parameters as they don't need to be concatenated with new gaussians
            if group["name"] in mlp_param_names:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue
                
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]

                req = True
                if getattr(self, 'freeze_geometry_flag', False) and group["name"] in ["xyz", "rotation", "scaling", "opacity"]:
                    req = False
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(req))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                req = True
                if getattr(self, 'freeze_geometry_flag', False) and group["name"] in ["xyz", "rotation", "scaling", "opacity"]:
                    req = False
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(req))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_normal, new_shs_dc, new_shs_rest, new_opacities, new_scaling,
                              new_rotation, new_base_color=None, new_roughness=None, new_metallic=None,
                              new_incidents_dc=None, new_incidents_rest=None,
                              new_visibility_dc=None, new_visibility_rest=None):
        d = {"xyz": new_xyz,
             "normal": new_normal,
             "rotation": new_rotation,
             "scaling": new_scaling,
             "opacity": new_opacities,
             "f_dc": new_shs_dc,
             "f_rest": new_shs_rest}

        if self.use_pbr:
            d.update({
                "base_color": new_base_color,
                "roughness": new_roughness,
                "metallic": new_metallic,
                "incidents_dc": new_incidents_dc,
                "incidents_rest": new_incidents_rest,
                "visibility_dc": new_visibility_dc,
                "visibility_rest": new_visibility_rest,
            })

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._rotation = optimizable_tensors["rotation"]
        self._scaling = optimizable_tensors["scaling"]
        self._opacity = optimizable_tensors["opacity"]
        self._shs_dc = optimizable_tensors["f_dc"]
        self._shs_rest = optimizable_tensors["f_rest"]

        self.weights_accum = torch.cat([self.weights_accum, torch.ones((new_xyz.shape[0], 1), device="cuda")], dim=0)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.use_pbr:
            self._base_color = optimizable_tensors["base_color"]
            self._roughness = optimizable_tensors["roughness"]
            self._metallic = optimizable_tensors["metallic"]
            self._incidents_dc = optimizable_tensors["incidents_dc"]
            self._incidents_rest = optimizable_tensors["incidents_rest"]
            self._visibility_dc = optimizable_tensors["visibility_dc"]
            self._visibility_rest = optimizable_tensors["visibility_rest"]
            
            # Update projected data for new gaussians if MLPs exist
            if hasattr(self, 'base_color_mlp') and hasattr(self, 'projected_base_colors'):
                num_train_images = self.projected_base_colors.shape[1]  # Get the number of training images from existing data
                # Initialize new gaussians with reasonable values
                new_projected_base_colors = torch.ones((new_xyz.shape[0], num_train_images, 3), device='cuda', requires_grad=False)
                new_projected_roughness = torch.full((new_xyz.shape[0], num_train_images, 1), 0.5, device='cuda', requires_grad=False)
                new_projected_metallic = torch.full((new_xyz.shape[0], num_train_images, 1), 0.5, device='cuda', requires_grad=False)
                new_projected_normals = torch.tensor([0.0, 1.0, 0.0], device='cuda', requires_grad=False).expand(new_xyz.shape[0], num_train_images, 3)
                self.projected_base_colors = torch.cat([self.projected_base_colors, new_projected_base_colors], dim=0)
                self.projected_roughness = torch.cat([self.projected_roughness, new_projected_roughness], dim=0)
                self.projected_metallic = torch.cat([self.projected_metallic, new_projected_metallic], dim=0)
                self.projected_normals = torch.cat([self.projected_normals, new_projected_normals], dim=0)
                
                # Also update normal MLP projected data if it exists
                if hasattr(self, 'normal_mlp') and hasattr(self, 'projected_normals'):
                    new_projected_normals = torch.tensor([0.0, 1.0, 0.0], device='cuda', requires_grad=False).expand(new_xyz.shape[0], num_train_images, 3)
                    self.projected_normals = torch.cat([self.projected_normals, new_projected_normals], dim=0)
            
            # Extend ID embeddings (deprecated, no-op)
            try:
                if hasattr(self, 'base_color_mlp'):
                    self.base_color_mlp.extend_embeddings(new_xyz.shape[0], device=self._xyz.device)
                if hasattr(self, 'roughness_mlp'):
                    self.roughness_mlp.extend_embeddings(new_xyz.shape[0], device=self._xyz.device)
                if hasattr(self, 'metallic_mlp'):
                    self.metallic_mlp.extend_embeddings(new_xyz.shape[0], device=self._xyz.device)
                if hasattr(self, 'normal_mlp'):
                    self.normal_mlp.extend_embeddings(new_xyz.shape[0], device=self._xyz.device)
            except Exception as e:
                print(f"Warning: Failed to extend ID embeddings: {e}")

    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_normal, grad_normal_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_grad_normal = torch.zeros((n_init_points), device="cuda")
        padded_grad_normal[:grads_normal.shape[0]] = grads_normal.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask_normal = torch.where(padded_grad_normal >= grad_normal_threshold, True, False)
        print("densify_and_split_normal:", selected_pts_mask_normal.sum().item(), "/", self.get_xyz.shape[0])

        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_normal)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        # print("densify_and_split:", selected_pts_mask.sum().item(), "/", self.get_xyz.shape[0])

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)  # (N, 3)
        means = torch.zeros((stds.size(0), 3), device="cuda")  # (N, 3)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_shs_dc = self._shs_dc[selected_pts_mask].repeat(N, 1, 1)
        new_shs_rest = self._shs_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        args = [new_xyz, new_normal, new_shs_dc, new_shs_rest, new_opacity, new_scaling, new_rotation]
        if self.use_pbr:
            new_base_color = self._base_color[selected_pts_mask].repeat(N, 1)
            new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
            new_metallic = self._metallic[selected_pts_mask].repeat(N, 1)
            new_incidents_dc = self._incidents_dc[selected_pts_mask].repeat(N, 1, 1)
            new_incidents_rest = self._incidents_rest[selected_pts_mask].repeat(N, 1, 1)
            new_visibility_dc = self._visibility_dc[selected_pts_mask].repeat(N, 1, 1)
            new_visibility_rest = self._visibility_rest[selected_pts_mask].repeat(N, 1, 1)
            args.extend([
                new_base_color,
                new_roughness,
                new_metallic,
                new_incidents_dc,
                new_incidents_rest,
                new_visibility_dc,
                new_visibility_rest,
            ])

        self.densification_postfix(*args)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_normal, grad_normal_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_normal = torch.where(torch.norm(grads_normal, dim=-1) >= grad_normal_threshold, True, False)
        # print("densify_and_clone_normal:", selected_pts_mask_normal.sum().item(), "/", self.get_xyz.shape[0])
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_normal)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)
        # print("densify_and_clone:", selected_pts_mask.sum().item(), "/", self.get_xyz.shape[0])

        new_xyz = self._xyz[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_shs_dc = self._shs_dc[selected_pts_mask]
        new_shs_rest = self._shs_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        args = [new_xyz, new_normal, new_shs_dc, new_shs_rest, new_opacities,
                new_scaling, new_rotation]
        if self.use_pbr:
            new_base_color = self._base_color[selected_pts_mask]
            new_roughness = self._roughness[selected_pts_mask]
            new_metallic = self._metallic[selected_pts_mask]
            new_incidents_dc = self._incidents_dc[selected_pts_mask]
            new_incidents_rest = self._incidents_rest[selected_pts_mask]
            new_visibility_dc = self._visibility_dc[selected_pts_mask]
            new_visibility_rest = self._visibility_rest[selected_pts_mask]

            args.extend([
                new_base_color,
                new_roughness,
                new_metallic,
                new_incidents_dc,
                new_incidents_rest,
                new_visibility_dc,
                new_visibility_rest,
            ])

        self.densification_postfix(*args)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_normal, weights_threshold=1e-4):
        # print(self.xyz_gradient_accum.shape)
        grads = self.xyz_gradient_accum / self.denom
        grads_normal = self.normal_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads_normal[grads_normal.isnan()] = 0.0

        # if self._xyz.shape[0] < 1000000:
        self.densify_and_clone(grads, max_grad, extent, grads_normal, max_grad_normal)
        self.densify_and_split(grads, max_grad, extent, grads_normal, max_grad_normal)
        # self.densify_and_compact()

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        weight_mask = self.weights_accum[:, 0] < weights_threshold
        prune_mask = torch.logical_or(weight_mask, prune_mask)
        print("weights_accum:", weight_mask.sum().item(), "/", self.get_xyz.shape[0])
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)
        self.weights_accum.data[:] = 0.0

        torch.cuda.empty_cache()

    def prune(self, min_opacity, extent, max_screen_size, weights_threshold=1e-4):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        weight_mask = self.weights_accum[:, 0] < weights_threshold
        prune_mask = torch.logical_or(weight_mask, prune_mask)
        print("weights_accum:", weight_mask.sum().item(), "/", self.get_xyz.shape[0])
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)
        self.weights_accum.data[:] = 0.0

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, weights):
        self.weights_accum += weights
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.normal_gradient_accum[update_filter] += torch.norm(
            self._normal.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def debug_mlp_outputs(self):
        """Debug method to check MLP outputs and diagnose issues"""
        # Check if any MLP is initialized
        has_combined = hasattr(self, 'combined_mlp') and self.combined_mlp is not None
        has_separate = hasattr(self, 'base_color_mlp') and self.base_color_mlp is not None
        
        if not has_combined and not has_separate:
            print("MLPs not initialized")
            return
        
        use_combined = getattr(self, 'use_combined_mlp', False)
        print("\n=== MLP Debug Information ===")
        print(f"Using combined MLP: {use_combined}")
        
        # Check projected data
        print(f"Projected base colors shape: {self.projected_base_colors.shape}")
        print(f"Projected roughness shape: {self.projected_roughness.shape}")
        print(f"Projected metallic shape: {self.projected_metallic.shape}")
        
        print(f"Projected base colors range: [{self.projected_base_colors.min():.3f}, {self.projected_base_colors.max():.3f}]")
        print(f"Projected roughness range: [{self.projected_roughness.min():.3f}, {self.projected_roughness.max():.3f}]")
        print(f"Projected metallic range: [{self.projected_metallic.min():.3f}, {self.projected_metallic.max():.3f}]")
        
        # Check MLP outputs
        with torch.no_grad():
            positions = self.get_xyz
            
            # Test with a small batch to avoid memory issues
            test_size = min(1000, positions.shape[0])
            test_positions = positions[:test_size]
            test_base_colors = self.projected_base_colors[:test_size]
            test_roughness = self.projected_roughness[:test_size]
            test_metallic = self.projected_metallic[:test_size]
            
            if use_combined and has_combined:
                # Test combined MLP
                base_color_output, roughness_output, metallic_output = self.combined_mlp(
                    test_base_colors, test_roughness, test_metallic, test_positions
                )
                print(f"\nCombined MLP Output Statistics (first {test_size} gaussians):")
                print(f"Base color weights range: [{base_color_output.min():.3f}, {base_color_output.max():.3f}]")
                print(f"Base color weights mean: {base_color_output.mean():.3f}")
                print(f"Roughness weights range: [{roughness_output.min():.3f}, {roughness_output.max():.3f}]")
                print(f"Roughness weights mean: {roughness_output.mean():.3f}")
                print(f"Metallic weights range: [{metallic_output.min():.3f}, {metallic_output.max():.3f}]")
                print(f"Metallic weights mean: {metallic_output.mean():.3f}")
                
                # Check if outputs are all the same (indicating a problem)
                if base_color_output.std() < 1e-6:
                    print("⚠️  WARNING: Combined MLP base color outputs are nearly identical - possible initialization issue!")
                if roughness_output.std() < 1e-6:
                    print("⚠️  WARNING: Combined MLP roughness outputs are nearly identical - possible initialization issue!")
                if metallic_output.std() < 1e-6:
                    print("⚠️  WARNING: Combined MLP metallic outputs are nearly identical - possible initialization issue!")
            else:
                # Test separate MLPs
                base_color_output = self.base_color_mlp(test_base_colors, test_positions)
                if hasattr(self, 'roughness_mlp') and self.roughness_mlp is not None:
                    roughness_output = self.roughness_mlp(test_roughness, test_positions)
                if hasattr(self, 'metallic_mlp') and self.metallic_mlp is not None:
                    metallic_output = self.metallic_mlp(test_metallic, test_positions)
                
                print(f"\nSeparate MLP Output Statistics (first {test_size} gaussians):")
                print(f"Base color output range: [{base_color_output.min():.3f}, {base_color_output.max():.3f}]")
                print(f"Base color output mean: {base_color_output.mean():.3f}")
                if hasattr(self, 'roughness_mlp') and self.roughness_mlp is not None:
                    print(f"Roughness output range: [{roughness_output.min():.3f}, {roughness_output.max():.3f}]")
                    print(f"Roughness output mean: {roughness_output.mean():.3f}")
                else:
                    print("Roughness MLP: Not enabled (using direct parameter)")
                if hasattr(self, 'metallic_mlp') and self.metallic_mlp is not None:
                    print(f"Metallic output range: [{metallic_output.min():.3f}, {metallic_output.max():.3f}]")
                    print(f"Metallic output mean: {metallic_output.mean():.3f}")
                
                # Check if outputs are all the same (indicating a problem)
                if base_color_output.std() < 1e-6:
                    print("⚠️  WARNING: Base color MLP outputs are nearly identical - possible initialization issue!")
                if hasattr(self, 'roughness_mlp') and self.roughness_mlp is not None and roughness_output.std() < 1e-6:
                    print("⚠️  WARNING: Roughness MLP outputs are nearly identical - possible initialization issue!")
                if hasattr(self, 'metallic_mlp') and self.metallic_mlp is not None and metallic_output.std() < 1e-6:
                    print("⚠️  WARNING: Metallic MLP outputs are nearly identical - possible initialization issue!")
            
            # Check the final output after applying scale
            final_base_color = base_color_output * self.base_color_scale[None, :]
            print(f"\nFinal base color (after scale) range: [{final_base_color.min():.3f}, {final_base_color.max():.3f}]")
            print(f"Final base color (after scale) mean: {final_base_color.mean():.3f}")
            print(f"Base color scale: {self.base_color_scale}")
        
        print("=" * 40)
    
    def test_mlp_call(self):
        """Simple test to verify MLP is being called and producing non-zero outputs"""
        if not hasattr(self, 'base_color_mlp'):
            print("No MLPs found")
            return False
        
        print("\n=== Testing MLP Call ===")
        
        # Test with just one gaussian
        test_positions = self.get_xyz[:1]  # [1, 3]
        test_base_colors = self.projected_base_colors[:1]  # [1, num_images, 3]
        test_roughness = self.projected_roughness[:1]  # [1, num_images, 1]
        test_metallic = self.projected_metallic[:1]  # [1, num_images, 1]
        test_normals = self.projected_normals[:1]  # [1, num_images, 3]
        
        print(f"Test positions shape: {test_positions.shape}")
        print(f"Test base colors shape: {test_base_colors.shape}")
        
        with torch.no_grad():
            if hasattr(self.base_color_mlp, "use_id_conditioning") and self.base_color_mlp.use_id_conditioning:
                ids = torch.arange(1, device=test_positions.device, dtype=torch.long)
                base_color_output = self.base_color_mlp(test_base_colors, gaussian_ids=ids)
                if hasattr(self, 'roughness_mlp'):
                    roughness_output = self.roughness_mlp(test_roughness, gaussian_ids=ids)
                metallic_output = self.metallic_mlp(test_metallic, gaussian_ids=ids)
            else:
                base_color_output = self.base_color_mlp(test_base_colors, test_positions)
                if hasattr(self, 'roughness_mlp'):
                    roughness_output = self.roughness_mlp(test_roughness, test_positions)
                metallic_output = self.metallic_mlp(test_metallic, test_positions)
            
            print(f"Base color output shape: {base_color_output.shape}")
            print(f"Base color output: {base_color_output}")
            if hasattr(self, 'roughness_mlp'):
                print(f"Roughness output: {roughness_output}")
            else:
                print("Roughness MLP: Not enabled (using direct parameter)")
            print(f"Metallic output: {metallic_output}")
            
            # Check if outputs are zero
            if torch.allclose(base_color_output, torch.zeros_like(base_color_output), atol=1e-6):
                print("❌ Base color MLP output is zero!")
                return False
            else:
                print("✅ Base color MLP output is non-zero")
                return True
        
        print("=" * 40)

    def quick_debug_mlp_status(self):
        """Quick debug to check MLP status and basic info"""
        print("\n=== Quick MLP Status Check ===")
        
        # Check if using combined MLP
        use_combined = getattr(self, 'use_combined_mlp', False)
        has_combined_mlp = hasattr(self, 'combined_mlp') and self.combined_mlp is not None
        print(f"Using combined MLP: {use_combined}")
        print(f"Combined MLP exists: {has_combined_mlp}")
        
        # Check if separate MLPs exist
        has_base_mlp = hasattr(self, 'base_color_mlp') and self.base_color_mlp is not None
        has_roughness_mlp = hasattr(self, 'roughness_mlp') and self.roughness_mlp is not None
        has_metallic_mlp = hasattr(self, 'metallic_mlp') and self.metallic_mlp is not None
        
        if not use_combined:
            print(f"Base color MLP exists: {has_base_mlp}")
            print(f"Roughness MLP exists: {has_roughness_mlp}")
            print(f"Metallic MLP exists: {has_metallic_mlp}")
        
        # Check if projected data exists
        has_projected_base = hasattr(self, 'projected_base_colors')
        has_projected_roughness = hasattr(self, 'projected_roughness')
        has_projected_metallic = hasattr(self, 'projected_metallic')
        has_projected_normals = hasattr(self, 'projected_normals')
        
        print(f"Projected base colors exist: {has_projected_base}")
        print(f"Projected roughness exist: {has_projected_roughness}")
        print(f"Projected metallic exist: {has_projected_metallic}")
        
        if has_projected_base:
            print(f"Projected base colors shape: {self.projected_base_colors.shape}")
            print(f"Projected base colors range: [{self.projected_base_colors.min():.3f}, {self.projected_base_colors.max():.3f}]")
        
        # Check intersection_tracing flag
        print(f"Intersection tracing flag: {getattr(self, 'intersection_tracing', False)}")
        
        # Check use_pbr flag
        print(f"Use PBR flag: {getattr(self, 'use_pbr', False)}")
        
        print("=" * 40)
