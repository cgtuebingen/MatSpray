import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math
import OpenEXR
import Imath
import os
import json
from utils.graphics_utils import rgb_to_srgb, fov2focal, focal2fov
from scene.cameras import Camera
from tqdm import tqdm
import time

def safe_normalize(x, dim=-1, eps=1e-6):
    # Pure device-side, branchless-ish path (no host sync via .all()/.any())
    x = x.clone()
    x.masked_fill_(~torch.isfinite(x), 0.0)
    x.clamp_(min=-1e6, max=1e6)
    norm2 = (x * x).sum(dim=dim, keepdim=True)
    inv_norm = torch.rsqrt(norm2.clamp_min(eps * eps))
    y = x * inv_norm
    y.masked_fill_(~torch.isfinite(y), 0.0)
    return y

def fast_normalize(x, dim=-1, eps=1e-6):
    # Minimal, fast normalization without finiteness checks (GPU-friendly)
    norm2 = (x * x).sum(dim=dim, keepdim=True)
    inv_norm = torch.rsqrt(norm2.clamp_min(eps * eps))
    return x * inv_norm

def load_exr_image(path):
    """Load EXR image with full precision"""
    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()
    
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = list(header['channels'].keys())
    
    print(f"EXR channels found: {channels}")
    
    # Try RGB channels first
    if 'R' in channels and 'G' in channels and 'B' in channels:
        (r, g, b) = exr_file.channels("RGB", FLOAT)
        r = np.frombuffer(r, dtype=np.float32).reshape((height, width))
        g = np.frombuffer(g, dtype=np.float32).reshape((height, width))
        b = np.frombuffer(b, dtype=np.float32).reshape((height, width))
        img = np.stack([r, g, b], axis=-1)
    # Try XYZ channels (common for normals)
    elif 'X' in channels and 'Y' in channels and 'Z' in channels:
        x = exr_file.channel('X', FLOAT)
        y = exr_file.channel('Y', FLOAT)
        z = exr_file.channel('Z', FLOAT)
        x = np.frombuffer(x, dtype=np.float32).reshape((height, width))
        y = np.frombuffer(y, dtype=np.float32).reshape((height, width))
        z = np.frombuffer(z, dtype=np.float32).reshape((height, width))
        img = np.stack([x, y, z], axis=-1)
    # Try numbered channels (0, 1, 2)
    elif '0' in channels and '1' in channels and '2' in channels:
        c0 = exr_file.channel('0', FLOAT)
        c1 = exr_file.channel('1', FLOAT)
        c2 = exr_file.channel('2', FLOAT)
        c0 = np.frombuffer(c0, dtype=np.float32).reshape((height, width))
        c1 = np.frombuffer(c1, dtype=np.float32).reshape((height, width))
        c2 = np.frombuffer(c2, dtype=np.float32).reshape((height, width))
        img = np.stack([c0, c1, c2], axis=-1)
    # If we have exactly 3 channels, use them in order
    elif len(channels) == 3:
        sorted_channels = sorted(channels)
        c0 = exr_file.channel(sorted_channels[0], FLOAT)
        c1 = exr_file.channel(sorted_channels[1], FLOAT)
        c2 = exr_file.channel(sorted_channels[2], FLOAT)
        c0 = np.frombuffer(c0, dtype=np.float32).reshape((height, width))
        c1 = np.frombuffer(c1, dtype=np.float32).reshape((height, width))
        c2 = np.frombuffer(c2, dtype=np.float32).reshape((height, width))
        img = np.stack([c0, c1, c2], axis=-1)
    else:
        # Single channel fallback
        channel_name = channels[0]
        channel_data = exr_file.channel(channel_name, FLOAT)
        img = np.frombuffer(channel_data, dtype=np.float32).reshape((height, width))
    
    exr_file.close()
    return img

def load_image(path, linear=False):
    """Load regular image files"""
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    
    if linear:
        img = np.where(img <= 0.04045, img / 12.92, np.power((img + 0.055) / 1.055, 2.4))
    
    return img

def load_camera_from_json(json_path, camera_name):
    """Load specific camera from JSON file"""
    if not json_path.endswith('.json'):
        json_path = os.path.join(json_path, "transforms_test.json")
    
    with open(json_path, 'r', encoding='UTF-8') as f:
        camera_data = json.load(f)
    
    H = camera_data.get("h", 512)
    W = camera_data.get("w", 512)
    fovx = camera_data.get("camera_angle_x", 0.6911112070083618)
    fovy = focal2fov(fov2focal(fovx, W), H)
    
    frames = camera_data.get("frames", [])
    target_frame = None
    
    for frame in frames:
        file_path = frame.get("file_path", "")
        frame_name = os.path.basename(file_path)
        if frame_name == camera_name:
            target_frame = frame
            break
    
    if target_frame is None:
        raise ValueError(f"Camera '{camera_name}' not found in {json_path}")
    
    c2w = np.array(target_frame.get("transform_matrix", []), dtype=np.float32)
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1
    
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T
    T = w2c[:3, 3]
    
    return {
        'R': R,
        'T': T,
        'fovx': fovx,
        'fovy': fovy,
        'H': H,
        'W': W,
        'camera_position': c2w[:3, 3],
        'w2c': w2c
    }

def create_environment_background(envmap_path, camera_params):
    """Create environment background"""
    from scene.cameras import Camera
    from scene.envmap import EnvLight
    
    light = EnvLight(path=envmap_path, scale=1.0)
    
    custom_cam = Camera(
        colmap_id=0, 
        R=camera_params['R'], 
        T=camera_params['T'],
        FoVx=camera_params['fovx'], 
        FoVy=camera_params['fovy'], 
        fx=None, fy=None, cx=None, cy=None,
        image=torch.zeros(3, camera_params['H'], camera_params['W']), 
        image_name=None, 
        uid=0
    )
    
    directions = custom_cam.get_world_directions()
    env_background = light.direct_light(directions.permute(1, 2, 0)).permute(2, 0, 1)
    
    return env_background, light

def ggx_importance_sampling(n_pixels, n_samples, roughness, device):
    """Generate GGX importance-sampled half-vectors in tangent space"""
    # Generate n_pixels * n_samples total samples
    total_samples = n_pixels * n_samples
    
    # Expand roughness to match total_samples
    roughness_expanded = roughness.repeat_interleave(n_samples)  # Shape: (total_samples,)
    # Convert roughness to alpha (UE4 convention: alpha = roughness^2)
    alpha = torch.clamp(roughness_expanded, 1e-6, 1.0) ** 2
    
    # Generate uniform random samples
    u1 = torch.rand(total_samples, device=device)
    u2 = torch.rand(total_samples, device=device)
    
    # GGX importance sampling in spherical coordinates
    # Sample theta (polar angle) according to GGX distribution
    # Note: alpha is already roughness^2, so we use alpha directly (not alpha^2)
    cos_theta = torch.sqrt((1.0 - u1) / (u1 * (alpha - 1.0) + 1.0))
    sin_theta = torch.sqrt(1.0 - cos_theta ** 2)
    
    # Sample phi (azimuthal angle) uniformly
    phi = 2.0 * math.pi * u2
    
    # Convert to Cartesian coordinates (tangent space)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta
    
    # Reshape to (n_pixels, n_samples, 3)
    half_vectors = torch.stack([x, y, z], dim=-1)
    half_vectors = half_vectors.view(n_pixels, n_samples, 3)
    
    return half_vectors

def ggx_vndf_importance_sampling(n_pixels, n_samples, roughness, normals, view_dirs, device):
    """Sample GGX Visible Normal Distribution (VNDF) half-vectors in world space.
    Args:
        n_pixels: number of points
        n_samples: samples per point
        roughness: (n_pixels,) or (n_pixels, 1) roughness parameter r in [0,1]
        normals: (n_pixels, 3) world-space normals
        view_dirs: (n_pixels, 3) world-space view directions
    Returns:
        half_vectors_world: (n_pixels, n_samples, 3) world-space half-vectors
    """
    N = n_pixels
    S = n_samples
    dtype = normals.dtype

    normals = fast_normalize(normals, dim=-1)
    view_dirs = fast_normalize(view_dirs, dim=-1)

    # Build orthonormal frame (tangent, bitangent, normal)
    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand_as(normals)
    close_to_up = torch.abs(torch.sum(normals * up, dim=-1, keepdim=True)) > 0.9
    ref = torch.where(
        close_to_up,
        torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(normals),
        up,
    )
    tangent = fast_normalize(torch.cross(ref, normals, dim=-1), dim=-1)
    bitangent = fast_normalize(torch.cross(normals, tangent, dim=-1), dim=-1)

    # Transform V to local frame
    v_x = torch.sum(view_dirs * tangent, dim=-1, keepdim=True)
    v_y = torch.sum(view_dirs * bitangent, dim=-1, keepdim=True)
    v_z = torch.sum(view_dirs * normals, dim=-1, keepdim=True)
    v_local = torch.cat([v_x, v_y, v_z], dim=-1)  # (N,3)

    # Roughness to alpha (GGX parameter)
    roughness = roughness.view(-1, 1).to(device=device, dtype=dtype)
    alpha = torch.clamp(roughness, 1e-6, 1.0) ** 2  # alpha = r^2

    # Stretch view
    Vh = v_local.clone()
    Vh[..., 0] = alpha.squeeze(-1) * Vh[..., 0]
    Vh[..., 1] = alpha.squeeze(-1) * Vh[..., 1]
    Vh = fast_normalize(Vh, dim=-1)

    # Build orthonormal basis around Vh
    lensq = (Vh[..., 0:1] ** 2 + Vh[..., 1:2] ** 2)
    T1 = torch.zeros_like(Vh)
    nonzero = lensq.squeeze(-1) > 0
    # For non-degenerate cases
    T1[nonzero, 0] = -Vh[nonzero, 1]
    T1[nonzero, 1] = Vh[nonzero, 0]
    T1[nonzero, 2] = 0.0
    T1[nonzero] = fast_normalize(T1[nonzero], dim=-1)
    # Fallback when lensq == 0
    T1[~nonzero] = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    T2 = fast_normalize(torch.cross(Vh, T1, dim=-1), dim=-1)

    # Random samples
    total = N * S
    u1 = torch.rand(total, device=device, dtype=dtype)
    u2 = torch.rand(total, device=device, dtype=dtype)
    r = torch.sqrt(u1).view(N, S, 1)
    phi = (2.0 * math.pi * u2).view(N, S, 1)
    t1 = r * torch.cos(phi)
    t2 = r * torch.sin(phi)

    # Interpolate t2 as in Heitz 2018
    s = 0.5 * (1.0 + Vh[..., 2:3])  # (N,1)
    s = s.unsqueeze(1).expand(-1, S, -1)  # (N,S,1)
    t2 = (1.0 - s) * torch.sqrt(torch.clamp(1.0 - t1 * t1, min=0.0)) + s * t2

    # Compute Nh in stretched space
    T1e = T1.unsqueeze(1).expand(-1, S, -1)
    T2e = T2.unsqueeze(1).expand(-1, S, -1)
    Vhe = Vh.unsqueeze(1).expand(-1, S, -1)
    tmp = torch.clamp(1.0 - t1 * t1 - t2 * t2, min=0.0)
    Nh = t1 * T1e + t2 * T2e + torch.sqrt(tmp) * Vhe
    Nh = fast_normalize(Nh, dim=-1)

    # Unstretch and clamp to upper hemisphere
    alpha_e = alpha.unsqueeze(1).expand(-1, S, -1)
    m_local = Nh.clone()
    m_local[..., 0] = alpha_e.squeeze(-1) * m_local[..., 0]
    m_local[..., 1] = alpha_e.squeeze(-1) * m_local[..., 1]
    m_local[..., 2] = torch.clamp(m_local[..., 2], min=0.0)
    m_local = fast_normalize(m_local, dim=-1)

    # Transform back to world
    half_vectors_world = (
        m_local[..., 0:1] * tangent.unsqueeze(1)
        + m_local[..., 1:2] * bitangent.unsqueeze(1)
        + m_local[..., 2:3] * normals.unsqueeze(1)
    )
    half_vectors_world = fast_normalize(half_vectors_world, dim=-1)

    return half_vectors_world

def cosine_weighted_hemisphere_sampling(n_pixels, n_samples, device):
    """Generate cosine-weighted hemisphere samples in tangent space"""
    # Generate n_pixels * n_samples total samples
    total_samples = n_pixels * n_samples
    
    # Generate uniform random samples
    u1 = torch.rand(total_samples, device=device)
    u2 = torch.rand(total_samples, device=device)
    
    # Cosine-weighted sampling
    cos_theta = torch.sqrt(u1)
    sin_theta = torch.sqrt(1.0 - u1)
    phi = 2.0 * math.pi * u2
    
    # Convert to Cartesian coordinates (tangent space)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta
    
    # Reshape to (n_pixels, n_samples, 3)
    samples = torch.stack([x, y, z], dim=-1)
    samples = samples.view(n_pixels, n_samples, 3)
    
    return samples

def uniform_hemisphere_sampling(n_pixels, n_samples, device):
    """Generate uniform hemisphere samples in tangent space"""
    total_samples = n_pixels * n_samples
    
    u1 = torch.rand(total_samples, device=device)
    u2 = torch.rand(total_samples, device=device)
    
    cos_theta = u1  # Uniform distribution over [0, 1]
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=0.0))
    phi = 2.0 * math.pi * u2
    
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta
    
    samples = torch.stack([x, y, z], dim=-1)
    samples = samples.view(n_pixels, n_samples, 3)
    
    return samples

def half_vector_to_light_direction(half_vectors, view_dirs):
    """Convert half-vectors to light directions using reflection formula"""
    # half_vectors: (n_pixels, n_samples, 3)
    # view_dirs: (n_pixels, 3)
    
    # Expand view_dirs to match half_vectors shape
    view_dirs_expanded = view_dirs.unsqueeze(1).expand(-1, half_vectors.shape[1], -1)
    
    # Compute dot product H·V
    dot_hv = torch.sum(half_vectors * view_dirs_expanded, dim=-1, keepdim=True)
    
    # Reflection formula: L = 2(H·V)H - V
    light_dirs = 2.0 * dot_hv * half_vectors - view_dirs_expanded
    
    return fast_normalize(light_dirs, dim=-1)

def compute_ggx_pdf(roughness, dot_nh, dot_vh):
    """Compute GGX importance sampling PDF"""
    # Interpret input as GGX alpha (aka roughness); use a2 = alpha^2 in NDF
    alpha = torch.clamp(roughness, 1e-6, 1.0)
    alpha2 = alpha * alpha
    
    # GGX normal distribution
    D = alpha2 / (torch.pi * (dot_nh ** 2 * (alpha2 - 1.0) + 1.0) ** 2 + 1e-6)
    
    # PDF = D * cos(theta_h) / (4 * dot(V, H))
    pdf = D * dot_nh / (4.0 * dot_vh + 1e-6)
    
    return pdf

def compute_cosine_pdf(cos_theta):
    """Compute cosine-weighted hemisphere sampling PDF"""
    # PDF = cos(theta) / π
    return cos_theta / torch.pi

def transform_hemisphere_to_normal_robust(hemisphere_samples, normals, view_dirs=None):
    """Transform hemisphere samples using quaternion rotation from [0,0,1] to normal"""
    device = normals.device
    n_valid, n_samples = hemisphere_samples.shape[:2]

    def safe_normalize(x, dim=-1, eps=1e-6):
        if not torch.isfinite(x).all():
            x = x.clone()
            x[~torch.isfinite(x)] = 0.0
        norm = torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)
        if not (torch.isfinite(norm).all() and (norm >= eps).all()):
            norm = norm.clone()
            bad = (~torch.isfinite(norm)) | (norm < eps)
            norm[bad] = eps
        y = x / norm
        if not torch.isfinite(y).all():
            y = y.clone()
            y[~torch.isfinite(y)] = 0.0
        return y
    
    def quaternion_from_vectors(from_vec, to_vec):
        """Compute quaternion that rotates from_vec to to_vec"""
        # Normalize input vectors (fast path)
        from_vec = fast_normalize(from_vec, dim=-1)
        to_vec = fast_normalize(to_vec, dim=-1)
        
        # Compute dot product
        dot = torch.sum(from_vec * to_vec, dim=-1)
        
        # Handle special cases
        eps = 1e-6
        
        # Case 1: Vectors are already aligned (dot ≈ 1)
        aligned_mask = dot > (1.0 - eps)
        
        # Case 2: Vectors are opposite (dot ≈ -1)  
        opposite_mask = dot < (-1.0 + eps)
        
        # Case 3: General case
        general_mask = ~(aligned_mask | opposite_mask)
        
        # Initialize quaternion [x, y, z, w]
        quat = torch.zeros(len(to_vec), 4, device=device)
        
        # Case 1: Identity quaternion
        quat[aligned_mask, 3] = 1.0  # w = 1, xyz = 0
        
        # Case 2: 180-degree rotation - find perpendicular axis
        if opposite_mask.any():
            # For [0,0,1] reference, use [1,0,0] as perpendicular axis when opposite
            perp = torch.tensor([1.0, 0.0, 0.0], device=device).expand(opposite_mask.sum(), -1)
            
            # 180-degree rotation quaternion: [axis, 0]
            quat[opposite_mask, :3] = perp
            quat[opposite_mask, 3] = 0.0
        
        # Case 3: General rotation
        if general_mask.any():
            from_gen = from_vec[general_mask]  
            to_gen = to_vec[general_mask]
            dot_gen = dot[general_mask]
            
            # Cross product gives rotation axis
            cross = torch.cross(from_gen, to_gen, dim=-1)
            
            # Quaternion components
            quat[general_mask, :3] = cross  # xyz = cross product
            quat[general_mask, 3] = 1.0 + dot_gen  # w = 1 + dot
            
            # Normalize
            quat[general_mask] = fast_normalize(quat[general_mask], dim=-1)
        
        return quat
    
    def rotate_points_by_quaternion(points, quaternions):
        """Rotate points using quaternions"""
        # points: (n_valid, n_samples, 3)
        # quaternions: (n_valid, 4) -> expand to (n_valid, n_samples, 4)
        
        q = quaternions.unsqueeze(1).expand(-1, n_samples, -1)
        
        # Extract quaternion components
        qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        # Extract point components  
        px, py, pz = points[..., 0], points[..., 1], points[..., 2]
        
        # Quaternion rotation formula: v' = q * v * q^(-1)
        # Optimized version
        
        # First: q * (0, px, py, pz)
        w1 = -qx*px - qy*py - qz*pz
        x1 = qw*px + qy*pz - qz*py
        y1 = qw*py + qz*px - qx*pz
        z1 = qw*pz + qx*py - qy*px
        
        # Second: result * q_conjugate
        rx = -w1*qx + x1*qw - y1*qz + z1*qy
        ry = -w1*qy + y1*qw - z1*qx + x1*qz
        rz = -w1*qz + z1*qw - x1*qy + y1*qx
        
        return torch.stack([rx, ry, rz], dim=-1)
    
    # Reference vector [0, 0, 1] - Z-axis up
    reference = torch.tensor([0.0, 0.0, 1.0], device=device)

    # Sanitize normals: replace NaN/Inf and zeros with reference
    if not torch.isfinite(normals).all():
        normals = normals.clone()
        normals[~torch.isfinite(normals)] = 0.0
    normal_len2 = (normals * normals).sum(dim=-1, keepdim=True)
    invalid_normals = ~torch.isfinite(normal_len2) | (normal_len2 < 1e-12)
    # Unconditional masked assignment (avoids host sync)
    mask = invalid_normals.expand_as(normals)
    if mask.any():  # minor guard to avoid expand cost on zero mask at runtime
        normals = normals.clone()
        ref_exp = reference.unsqueeze(0).expand_as(normals)
        normals[mask] = ref_exp[mask]
    
    # Compute quaternions that rotate [0,0,1] to each normal
    quaternions = quaternion_from_vectors(reference.unsqueeze(0).expand(n_valid, -1), normals)
    
    if view_dirs is not None:
        # Perfect reflection mode
        view_dirs_expanded = view_dirs.unsqueeze(1).expand(-1, n_samples, -1)
        dot_nv = torch.sum(normals.unsqueeze(1) * view_dirs_expanded, dim=-1, keepdim=True)
        reflection_dirs = 2.0 * dot_nv * normals.unsqueeze(1) - view_dirs_expanded
        world_directions = fast_normalize(reflection_dirs, dim=-1)
    else:
        # Rotate hemisphere samples by the quaternions
        world_directions = rotate_points_by_quaternion(hemisphere_samples, quaternions)
        world_directions = fast_normalize(world_directions, dim=-1)
    
    # Sanitize backward gradients by default; disable with GRAD_SANITIZE=0
    if os.environ.get("GRAD_SANITIZE", "1") != "0":
        try:
            world_directions.register_hook(lambda g: (torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) if g is not None else None))
        except Exception:
            pass

    return world_directions

def depth_to_world_position(depth, camera_position, world_directions):
    """Convert depth to world positions using camera directions"""
    # world_directions shape: (3, H, W)
    # depth shape: (H, W)
    # camera_position shape: (3,)
    
    # Multiply each direction by its corresponding depth
    world_offsets = world_directions * depth.unsqueeze(0)  # (3, H, W)
    
    # Add camera position to get world coordinates
    world_positions = camera_position.unsqueeze(-1).unsqueeze(-1) + world_offsets  # (3, H, W)
    
    return world_positions.permute(1, 2, 0)  # (H, W, 3)

def fresnel_schlick(dot_vh, f0):
    return f0 + (1.0 - f0) * torch.pow(2.0, (-5.55473 * dot_vh - 6.98316) * dot_vh)

def ndf_ggx(roughness, dot_nh):
    alpha = roughness ** 2
    alpha2 = alpha ** 2
    return alpha2 / (torch.pi * (dot_nh ** 2 * (alpha2 - 1.0) + 1.0) ** 2 + 1e-6)

def geometry_smith(roughness, dot_nv, dot_nl, dot_vh, dot_lh):
    def geometry_schlick_ggx(dot_nx, roughness):
        alpha = roughness ** 2  # UE4: α = roughness²
        k = alpha / 2.0  # UE4 IBL: k = α/2
        return dot_nx / (dot_nx * (1.0 - k) + k + 1e-6)
    
    ggx2 = geometry_schlick_ggx(dot_nv, roughness)
    ggx1 = geometry_schlick_ggx(dot_nl, roughness)
    
    return ggx1 * ggx2

def compute_pbr_brdf(base_color, metalness, roughness, normals, viewdirs, light_dirs):
    """Compute PBR BRDF terms"""
    # Optional faceforward for robustness at grazing/backfacing cases
    if os.environ.get("FACEFORWARD_NORMALS", "0") != "0":
        dot_nv_raw_ff = torch.sum(normals * viewdirs, dim=-1, keepdim=True)
        flip_mask = dot_nv_raw_ff < 0.0
        if flip_mask.any():
            normals = torch.where(flip_mask, -normals, normals)

    half_vector = fast_normalize(light_dirs + viewdirs.unsqueeze(-2), dim=-1)
    
    dot_nv = torch.sum(normals * viewdirs, dim=-1, keepdim=True)
    dot_nl = torch.max(torch.sum(normals.unsqueeze(-2) * light_dirs, dim=-1, keepdim=True), torch.tensor(0.0))
    dot_nh = torch.max(torch.sum(normals.unsqueeze(-2) * half_vector, dim=-1, keepdim=True), torch.tensor(0.0))
    dot_vh = torch.max(torch.sum(viewdirs.unsqueeze(-2) * half_vector, dim=-1, keepdim=True), torch.tensor(0.0))
    dot_lh = torch.max(torch.sum(light_dirs * half_vector, dim=-1, keepdim=True), torch.tensor(0.0))

    base_color_exp = base_color.unsqueeze(-2)
    metalness_exp = metalness.unsqueeze(-2)
    roughness_exp = roughness.unsqueeze(-2)
    
    f0 = 0.04 * (1.0 - metalness_exp) + base_color_exp * metalness_exp

    # Compute BRDF components
    Fr = fresnel_schlick(dot_vh, f0)
    D = ndf_ggx(roughness_exp, dot_nh)
    # Robustify dot(N,V) with optional epsilon
    try:
        dotnv_eps = float(os.environ.get("DOTNV_EPS", "0.0"))
    except Exception:
        dotnv_eps = 0.0
    try:
        dotnv_tol = float(os.environ.get("DOTNV_TOL", "0.05"))
    except Exception:
        dotnv_tol = 0.05
    dot_nv_safe = torch.where(
        dot_nv >= 0.0,
        torch.clamp(dot_nv, min=dotnv_eps),
        torch.where(dot_nv >= -dotnv_tol, torch.full_like(dot_nv, dotnv_eps), torch.zeros_like(dot_nv))
    )
    G = geometry_smith(roughness_exp, dot_nv_safe.unsqueeze(-2), dot_nl, dot_vh, dot_lh)
    
    # Use a more robust epsilon that scales with the dot products
    # This prevents division by very small numbers at grazing angles
    denominator_base = 4.0 * dot_nv_safe.unsqueeze(-2) * dot_nl
    denominator_eps = torch.max(denominator_base * 1e-4, torch.tensor(1e-6, device=denominator_base.device))
    denominator = denominator_base + denominator_eps
    f_s = (Fr * D * G) / denominator
    
    # Clamp specular BRDF to prevent fireflies
    # This prevents extremely high values that can occur with uniform sampling
    # especially at grazing angles or with very low roughness
    max_specular = float(os.environ.get("MAX_SPECULAR_BRDF", "100.0"))
    f_s = torch.clamp(f_s, min=0.0, max=max_specular)
    
    f_d = (1.0 - metalness_exp) * base_color_exp / torch.pi

    # No specular debugging output

    return f_d, f_s

def render_pbr_with_envmap_background(base_color, roughness, metallic, normals, depth, 
                                     env_background, light, camera_position, custom_cam, 
                                     n_samples=128, chunk_size=32):
    """Render PBR with environment map background using shared hemisphere samples"""
    start_time = time.time()
    
    H, W = depth.shape
    device = base_color.device
    
    depth_mask = (depth > 0) & (depth < 10000) & torch.isfinite(depth)
    normal_lengths = torch.norm(normals, dim=-1)
    normal_mask = normal_lengths > 0.1
    valid_mask = depth_mask & normal_mask
    
    final_image = env_background.permute(1, 2, 0).clone()
    
    if valid_mask.any():
        # Get world directions from camera and convert depth to world positions
        world_directions = custom_cam.get_world_directions()  # (3, H, W)
        world_positions = depth_to_world_position(depth, camera_position, world_directions)
        view_dirs = fast_normalize(camera_position.unsqueeze(0).unsqueeze(0) - world_positions, dim=-1)
        
        valid_h, valid_w = torch.where(valid_mask)
        n_valid = len(valid_h)
        
        if n_valid > 0:
            print(f"Rendering {n_valid:,} valid pixels with {n_samples} samples each...")
            
            valid_base_color = base_color[valid_mask]
            valid_roughness = roughness[valid_mask]
            valid_metallic = metallic[valid_mask]
            
            # Extract valid normals
            valid_normals_raw = normals[valid_mask]  # Shape: (n_valid, 3)
            valid_normals = fast_normalize(valid_normals_raw, dim=-1)
            
            valid_view_dirs = view_dirs[valid_mask]
            
            # Check normal orientations
            dot_nv_check = torch.sum(valid_normals * valid_view_dirs, dim=-1)
            print(f"Normal-View dot products range: [{dot_nv_check.min():.3f}, {dot_nv_check.max():.3f}]")
            
            accumulated_diffuse = None
            accumulated_specular = None
            
            use_chunking = chunk_size is not None and chunk_size < n_samples
            if use_chunking:
                total_chunks = (n_samples + chunk_size - 1) // chunk_size
                chunk_iterator = tqdm(
                    range(0, n_samples, chunk_size),
                    desc="Rendering chunks",
                    unit="chunk",
                    total=total_chunks,
                    ncols=80
                )
            else:
                chunk_iterator = [0]

            for chunk_start in chunk_iterator:
                current_chunk_size = (
                    min(chunk_size, n_samples - chunk_start)
                    if use_chunking else n_samples
                )

                if use_chunking:
                    chunk_iterator.set_postfix({
                        'samples': f"{chunk_start + current_chunk_size}/{n_samples}",
                        'pixels': f"{n_valid:,}",
                        'memory': f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
                    })
                
                hemisphere_samples = uniform_hemisphere_sampling(n_valid, current_chunk_size, device)
                
                light_dirs = transform_hemisphere_to_normal_robust(hemisphere_samples, valid_normals)
                light_dirs = fast_normalize(light_dirs, dim=-1)
                
                env_colors_exp = light.direct_light(light_dirs)
                
                f_d, f_s = compute_pbr_brdf(
                    valid_base_color, valid_metallic, valid_roughness, 
                    valid_normals, valid_view_dirs, light_dirs
                )
                
                cos_theta = torch.clamp(
                    torch.sum(valid_normals.unsqueeze(-2) * light_dirs, dim=-1, keepdim=True), 
                    0.0, 1.0
                )

                # Monte Carlo weight for uniform hemisphere sampling (pdf = 1 / (2π))
                mc_weight = cos_theta * (2.0 * math.pi)

                diffuse_contribution = f_d * env_colors_exp * mc_weight
                specular_contribution = f_s * env_colors_exp * mc_weight

                diffuse_chunk_sum = diffuse_contribution.sum(dim=-2)
                specular_chunk_sum = specular_contribution.sum(dim=-2)

                if accumulated_diffuse is None:
                    accumulated_diffuse = diffuse_chunk_sum
                else:
                    accumulated_diffuse = accumulated_diffuse + diffuse_chunk_sum

                if accumulated_specular is None:
                    accumulated_specular = specular_chunk_sum
                else:
                    accumulated_specular = accumulated_specular + specular_chunk_sum
            
            if use_chunking:
                chunk_iterator.close()
            
            final_diffuse = accumulated_diffuse / n_samples if accumulated_diffuse is not None else torch.zeros(n_valid, 3, device=device)
            final_specular = accumulated_specular / n_samples if accumulated_specular is not None else torch.zeros(n_valid, 3, device=device)
            final_image[valid_mask] = final_diffuse + final_specular
    
    total_time = time.time() - start_time
    print(f"✓ Rendering completed in {total_time:.2f} seconds")
    
    return final_image

def render_pbr_importance_sampling(base_color, roughness, metallic, normals, depth, 
                                  env_background, light, camera_position, custom_cam, 
                                  n_diffuse_samples=64, n_specular_samples=128, chunk_size=32):
    """Render PBR using importance sampling for both diffuse and specular terms"""
    start_time = time.time()
    
    H, W = depth.shape
    device = base_color.device
    
    depth_mask = (depth > 0) & (depth < 10000) & torch.isfinite(depth)
    normal_lengths = torch.norm(normals, dim=-1)
    normal_mask = normal_lengths > 0.1
    valid_mask = depth_mask & normal_mask
    
    final_image = env_background.permute(1, 2, 0).clone()
    
    if valid_mask.any():
        # Get world directions from camera and convert depth to world positions
        world_directions = custom_cam.get_world_directions()  # (3, H, W)
        world_positions = depth_to_world_position(depth, camera_position, world_directions)
        view_dirs = fast_normalize(camera_position.unsqueeze(0).unsqueeze(0) - world_positions, dim=-1)
        
        valid_h, valid_w = torch.where(valid_mask)
        n_valid = len(valid_h)
        
        if n_valid > 0:
            print(f"Rendering {n_valid:,} valid pixels with {n_diffuse_samples} diffuse + {n_specular_samples} specular samples each...")
            
            valid_base_color = base_color[valid_mask]
            valid_roughness = roughness[valid_mask]
            valid_metallic = metallic[valid_mask]
            
            # Extract valid normals
            valid_normals_raw = normals[valid_mask]  # Shape: (n_valid, 3)
            valid_normals = fast_normalize(valid_normals_raw, dim=-1)
            
            valid_view_dirs = view_dirs[valid_mask]
            
            # Check normal orientations
            dot_nv_check = torch.sum(valid_normals * valid_view_dirs, dim=-1)
            print(f"Normal-View dot products range: [{dot_nv_check.min():.3f}, {dot_nv_check.max():.3f}]")
            
            accumulated_diffuse = None
            accumulated_specular = None
            
            total_samples = n_diffuse_samples + n_specular_samples
            use_chunking = chunk_size is not None and chunk_size < total_samples
            if use_chunking:
                total_chunks = (total_samples + chunk_size - 1) // chunk_size

                # Progress bar for chunk processing
                chunk_iterator = tqdm(
                    range(0, total_samples, chunk_size),
                    desc="Rendering chunks",
                    unit="chunk",
                    total=total_chunks,
                    ncols=80
                )
            else:
                chunk_iterator = [0]

            for chunk_start in chunk_iterator:
                current_chunk_size = (
                    min(chunk_size, total_samples - chunk_start)
                    if use_chunking else total_samples
                )

                if use_chunking:
                    chunk_iterator.set_postfix({
                        'samples': f"{chunk_start + current_chunk_size}/{total_samples}",
                        'pixels': f"{n_valid:,}",
                        'memory': f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
                    })
                
                # === DIFFUSE SAMPLING ===
                if chunk_start < n_diffuse_samples:
                    diffuse_chunk_size = min(current_chunk_size, n_diffuse_samples - chunk_start)
                    
                    # Generate cosine-weighted hemisphere samples
                    cosine_samples = cosine_weighted_hemisphere_sampling(n_valid, diffuse_chunk_size, device)
                    
                    # Transform to world space
                    light_dirs_diffuse = transform_hemisphere_to_normal_robust(cosine_samples, valid_normals)
                    light_dirs_diffuse = fast_normalize(light_dirs_diffuse, dim=-1)
                    
                    # Get environment colors
                    env_colors_diffuse = light.direct_light(light_dirs_diffuse)
                    
                    # Compute cosine term
                    cos_theta_diffuse = torch.clamp(
                        torch.sum(valid_normals.unsqueeze(-2) * light_dirs_diffuse, dim=-1, keepdim=True), 
                        0.0, 1.0
                    )
                    
                    # Compute diffuse BRDF (Lambertian)
                    f_d = (1.0 - valid_metallic.unsqueeze(-2)) * valid_base_color.unsqueeze(-2) / torch.pi
                    
                    # Compute PDF for cosine-weighted sampling
                    pdf_diffuse = compute_cosine_pdf(cos_theta_diffuse.squeeze(-1))
                    
                    # Monte Carlo estimator: f(x) * L(x) * cos(theta) / pdf(x)
                    # For cosine-weighted sampling: pdf = cos(theta)/π
                    # So: f_d * L * cos(theta) / (cos(theta)/π) = f_d * L * π
                    diffuse_contribution = f_d * env_colors_diffuse * torch.pi
                    diffuse_chunk_sum = diffuse_contribution.sum(dim=-2)
                    if accumulated_diffuse is None:
                        accumulated_diffuse = diffuse_chunk_sum
                    else:
                        accumulated_diffuse = accumulated_diffuse + diffuse_chunk_sum
                
                # === SPECULAR SAMPLING ===
                specular_chunk_start = max(0, chunk_start - n_diffuse_samples)
                specular_chunk_end = min(chunk_start + current_chunk_size - n_diffuse_samples, n_specular_samples)
                
                if specular_chunk_start < specular_chunk_end:
                    specular_chunk_size = specular_chunk_end - specular_chunk_start
                    
                    # Generate GGX importance-sampled half-vectors
                    half_vectors_tangent = ggx_importance_sampling(n_valid, specular_chunk_size, valid_roughness, device)
                    
                    # Transform half-vectors to world space
                    half_vectors_world = transform_hemisphere_to_normal_robust(half_vectors_tangent, valid_normals)
                    half_vectors_world = fast_normalize(half_vectors_world, dim=-1)
                    
                    # Convert half-vectors to light directions
                    light_dirs_specular = half_vector_to_light_direction(half_vectors_world, valid_view_dirs)
                    
                    # Check if light directions are above the surface
                    cos_theta_specular = torch.sum(valid_normals.unsqueeze(-2) * light_dirs_specular, dim=-1, keepdim=True)
                    valid_specular_mask = cos_theta_specular > 0.0
                    
                    if valid_specular_mask.any():
                        # Get environment colors
                        env_colors_specular = light.direct_light(light_dirs_specular)
                        
                        # Compute all dot products needed for BRDF
                        dot_nv = torch.max(torch.sum(valid_normals * valid_view_dirs, dim=-1, keepdim=True), torch.tensor(0.0))
                        dot_nl = torch.clamp(cos_theta_specular, 0.0, 1.0)
                        dot_nh = torch.clamp(torch.sum(valid_normals.unsqueeze(-2) * half_vectors_world, dim=-1, keepdim=True), 0.0, 1.0)
                        dot_vh = torch.clamp(torch.sum(valid_view_dirs.unsqueeze(-2) * half_vectors_world, dim=-1, keepdim=True), 0.0, 1.0)
                        dot_lh = torch.clamp(torch.sum(light_dirs_specular * half_vectors_world, dim=-1, keepdim=True), 0.0, 1.0)
                        
                        # Compute specular BRDF components
                        f0 = 0.04 * (1.0 - valid_metallic.unsqueeze(-2)) + valid_base_color.unsqueeze(-2) * valid_metallic.unsqueeze(-2)
                        Fr = fresnel_schlick(dot_vh, f0)
                        D = ndf_ggx(valid_roughness.unsqueeze(-2), dot_nh)
                        G = geometry_smith(valid_roughness.unsqueeze(-2), dot_nv.unsqueeze(-2), dot_nl, dot_vh, dot_lh)
                        
                        # Specular BRDF
                        denominator = 4.0 * dot_nv.unsqueeze(-2) * dot_nl + 1e-6
                        f_s = (Fr * D * G) / denominator
                        
                        # Compute GGX PDF - fix shape issues
                        # Ensure all inputs have matching dimensions
                        roughness_for_pdf = valid_roughness.squeeze(-1).unsqueeze(-1).expand(-1, specular_chunk_size)
                        dot_nh_for_pdf = dot_nh.squeeze(-1)  # Remove last dimension
                        dot_vh_for_pdf = dot_vh.squeeze(-1)  # Remove last dimension
                        
                        pdf_specular = compute_ggx_pdf(roughness_for_pdf, dot_nh_for_pdf, dot_vh_for_pdf)
                        
                        # Monte Carlo estimator: f(x) * L(x) * cos(theta) / pdf(x)
                        specular_contribution = f_s * env_colors_specular * dot_nl / (pdf_specular.unsqueeze(-1) + 1e-6)
                        
                        # Apply validity mask
                        specular_contribution = specular_contribution * valid_specular_mask.float()
                        specular_chunk_sum = specular_contribution.sum(dim=-2)
                        if accumulated_specular is None:
                            accumulated_specular = specular_chunk_sum
                        else:
                            accumulated_specular = accumulated_specular + specular_chunk_sum
            
            if use_chunking:
                chunk_iterator.close()
            
            # Normalize by number of samples
            final_diffuse = accumulated_diffuse / n_diffuse_samples if accumulated_diffuse is not None else torch.zeros(n_valid, 3, device=device)
            final_specular = accumulated_specular / n_specular_samples if accumulated_specular is not None else torch.zeros(n_valid, 3, device=device)

            # Combine diffuse and specular components
            final_color = final_diffuse + final_specular
            
            # Quick test: Additional global brightness boost if needed for Blender matching
            # Adjust this value based on visual comparison with Blender ground truth
            global_brightness_boost = 1.0
            final_color = final_color * global_brightness_boost
            
            final_image[valid_mask] = final_color
    
    total_time = time.time() - start_time
    print(f"✓ Importance sampling rendering completed in {total_time:.2f} seconds")
    
    return final_image

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Add this import
    from scene.cameras import Camera
    
    # Configuration
    json_path = "/path/to/your/dataset/transforms_test.json"
    camera_name = "r_0"
    envmap_path = "/path/to/your/env_maps/envmap12.exr"
    
    base_color_path = f"/path/to/your/dataset/test/base_basecolor/{camera_name}_basecolor0001.png"
    roughness_path = f"/path/to/your/dataset/test/base_metallicroughness/{camera_name}_metallicroughness0001.png"
    metallic_path = f"/path/to/your/dataset/test/base_metallicroughness/{camera_name}_metallicroughness0001.png"
    normals_path = f"/path/to/your/dataset/test/base_normal/{camera_name}_normal0001.exr"
    depth_path = f"/path/to/your/dataset/test/base_depth/{camera_name}_depth0001.exr"
    
    # Load camera
    camera_params = load_camera_from_json(json_path, camera_name)
    H, W = camera_params['H'], camera_params['W']
    
    # Create environment background
    env_background, light = create_environment_background(envmap_path, camera_params)
    
    # Environment map loaded successfully
    
    # Calculate intrinsics
    fovx, fovy = camera_params['fovx'], camera_params['fovy']
    fx = fov2focal(fovx, W)
    fy = fov2focal(fovy, H)
    cx, cy = W / 2.0, H / 2.0
    intrinsics_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # Convert to tensors
    camera_position = torch.tensor(camera_params['camera_position'], dtype=torch.float32).to(device)
    camera_pose = torch.tensor(camera_params['w2c'], dtype=torch.float32).to(device)
    intrinsics = torch.tensor(intrinsics_np, dtype=torch.float32).to(device)
    env_background = env_background.to(device)
    
    # Print camera info
    print(f"Camera position: [{camera_position[0]:.6f}, {camera_position[1]:.6f}, {camera_position[2]:.6f}]")
    c2w_matrix = torch.linalg.inv(camera_pose)
    camera_forward = -c2w_matrix[:3, 2]
    print(f"Camera forward direction: [{camera_forward[0]:.6f}, {camera_forward[1]:.6f}, {camera_forward[2]:.6f}]")
    
    # Load material data
    base_color_data = load_image(base_color_path, linear=True)
    roughness_data = load_image(roughness_path, linear=False)
    metallic_data = load_image(metallic_path, linear=False)
    # metallic_data = np.clip(metallic_data, 1.0, 1.0)
    normals_data = load_exr_image(normals_path)
    depth_data = load_exr_image(depth_path)
    
    # Convert to tensors
    base_color = torch.from_numpy(base_color_data).to(device).float()
    roughness = torch.from_numpy(roughness_data).to(device).float()
    metallic = torch.from_numpy(metallic_data).to(device).float()
    normals = torch.from_numpy(normals_data).to(device).float()
    depth = torch.from_numpy(depth_data).to(device).float()
    
    # Handle tensor shapes
    if base_color.shape[0] == 3: base_color = base_color.permute(1, 2, 0)
    if roughness.shape[0] == 3: roughness = roughness.permute(1, 2, 0)
    if metallic.shape[0] == 3: metallic = metallic.permute(1, 2, 0)
    if normals.shape[0] == 3: normals = normals.permute(1, 2, 0)
    
    if len(depth.shape) == 3: depth = depth[:, :, 0]
    if roughness.shape[-1] > 1: roughness = roughness[..., 1:2]
    if metallic.shape[-1] > 1: metallic = metallic[..., 0:1]
    
    normals = fast_normalize(normals, dim=-1)

    # Create custom camera for directions
    custom_cam = Camera(
        colmap_id=0, 
        R=camera_params['R'], 
        T=camera_params['T'],
        FoVx=camera_params['fovx'], 
        FoVy=camera_params['fovy'], 
        fx=None, fy=None, cx=None, cy=None,
        image=torch.zeros(3, H, W), 
        image_name=None, 
        uid=0
    )
    
    print("Starting PBR hemisphere sampling rendering...")
    overall_start = time.time()
    
    # Render without importance sampling (shared hemisphere samples)
    with torch.no_grad():
        rendered = render_pbr_with_envmap_background(
            base_color, roughness, metallic, normals, depth, 
            env_background, light, camera_position, custom_cam, 
            n_samples=64, chunk_size=None
        )
    
    overall_time = time.time() - overall_start
    
    # Save result
    print("Saving results...")
    os.makedirs("screen_space_renders", exist_ok=True)
    rendered_tensor = rendered.permute(2, 0, 1)
    rendered_srgb = rgb_to_srgb(rendered_tensor, clip=True)
    rendered_np = rendered_srgb.permute(1, 2, 0).cpu().numpy()
    
    filename = f"screen_space_renders/pbr_uniform_sampling_{camera_name}_redblackroof.png"
    Image.fromarray((rendered_np * 255).astype(np.uint8)).save(filename)
    
    print(f"✓ Complete pipeline finished in {overall_time:.2f} seconds")
    print(f"✓ Saved: {filename}")
if __name__ == "__main__":
    main()