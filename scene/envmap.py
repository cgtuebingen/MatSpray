
import torch
import torch.nn.functional as F
import numpy as np
import nvdiffrast.torch as dr
import imageio
import os
from utils.graphics_utils import srgb_to_rgb
from torchvision.transforms import GaussianBlur

# Optional dependency: pyexr (not required if OpenEXR is available)
try:
    import pyexr  # type: ignore
except Exception:  # pragma: no cover
    pyexr = None  # Fallback to OpenEXR reader below

# Avoid failing on environments without internet/freeimage; EXR loading doesn't need this
try:
    imageio.plugins.freeimage.download()  # type: ignore[attr-defined]
except Exception:
    pass

class EnvLight(torch.nn.Module):
    def __init__(self, path=None, scale=1.0):
        super().__init__()
        self.device = "cuda"  # only supports cuda
        self.scale = scale  # scale of the hdr values
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")

        self.envmap = self.load(path, scale=self.scale, device=self.device)
        

        self.transform = None

    @staticmethod
    def load(envmap_path, scale, device):
        if not envmap_path.endswith(".exr"):
            image = srgb_to_rgb(imageio.imread(envmap_path)[:, :, :3] / 255)
        else:
            # Load EXR without using OpenEXR bindings to avoid native segfaults
            # Drop alpha channel if present
            normalized_path = os.path.expanduser(os.path.expandvars(envmap_path))
            if not os.path.exists(normalized_path):
                raise FileNotFoundError(f"Environment map not found: {normalized_path}")

            image = None
            # Try imageio first (FreeImage/OpenImageIO backends)
            try:
                img = imageio.imread(normalized_path)
                image = img
            except Exception:
                # Fallback to pyexr if available
                if pyexr is not None:
                    try:
                        img = pyexr.open(normalized_path).get()
                        image = img
                    except Exception:
                        pass
            if image is None:
                raise RuntimeError(f"Failed to load EXR environment map: {normalized_path}")

            # Ensure HxWxC float32, keep only RGB, drop alpha if present
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            if image.shape[-1] == 4:
                image = image[..., :3]
            elif image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            elif image.shape[-1] >= 3:
                image = image[..., :3]
            else:
                h, w = image.shape[:2]
                pad = np.zeros((h, w, 3), dtype=image.dtype)
                pad[..., :image.shape[-1]] = image
                image = pad
            image = image.astype(np.float32, copy=False)

        image = image * scale

        env_map_torch = torch.tensor(image, dtype=torch.float32, device=device, requires_grad=False)

        return env_map_torch

    def direct_light(self, dirs, transform=None):
        shape = dirs.shape
        dirs = dirs.reshape(-1, 3)

        if transform is not None:
            dirs = dirs @ transform.T
        elif self.transform is not None:
            dirs = dirs @ self.transform.T

        envir_map =  self.envmap.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        phi = torch.arccos(dirs[:, 2]).reshape(-1) - 1e-6
        theta = torch.atan2(dirs[:, 1], dirs[:, 0]).reshape(-1)
        # normalize to [-1, 1]
        query_y = (phi / np.pi) * 2 - 1
        query_x = - theta / np.pi
        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
        light_rgbs = F.grid_sample(
            envir_map, grid, 
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
    
        return light_rgbs.reshape(*shape)
