import torch
import numpy as np


def turbo_colormap(x):
    """
    Turbo colormap implementation using pure NumPy.
    Maps values in [0, 1] to RGB colors similar to matplotlib's turbo colormap.
    """
    x = np.clip(x, 0, 1)
    # Turbo colormap key points (simplified version)
    # Colors: dark blue -> cyan -> green -> yellow -> red
    colors = np.array([
        [0.18995, 0.07176, 0.23217],  # Dark blue
        [0.19483, 0.28339, 0.36834],  # Blue
        [0.19962, 0.40753, 0.55848],  # Cyan-blue
        [0.16866, 0.54953, 0.74535],  # Cyan
        [0.09570, 0.68895, 0.87130],  # Light cyan
        [0.36997, 0.81734, 0.84472],  # Green-cyan
        [0.60870, 0.93526, 0.78355],  # Light green
        [0.76147, 0.98340, 0.65355],  # Yellow-green
        [0.87843, 0.95213, 0.53245],  # Yellow
        [0.99920, 0.97515, 0.52521],  # Light yellow
        [0.95646, 0.83843, 0.27468],  # Orange
        [0.90194, 0.63319, 0.06331],  # Red-orange
        [0.70567, 0.01557, 0.15023],  # Red
    ], dtype=np.float32)
    
    # Interpolate between color points
    n_colors = len(colors)
    x_scaled = x * (n_colors - 1)
    indices = np.clip(x_scaled.astype(np.int32), 0, n_colors - 2)
    t = (x_scaled - indices)[..., np.newaxis]
    
    # Linear interpolation using vectorized operations
    c0 = colors[indices]
    c1 = colors[indices + 1]
    result = c0 * (1 - t) + c1 * t
    
    return result


def visualize_depth(depth, near=0.2, far=13):
    depth = depth[0].detach().cpu().numpy()
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = turbo_colormap(depth)

    out_depth = np.clip(np.nan_to_num(vis), 0., 1.)
    return torch.from_numpy(out_depth).float().cuda().permute(2, 0, 1)


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    return 20 * torch.log10(1.0 / torch.sqrt(mse(img1, img2)))
