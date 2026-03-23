import array

import cv2
import Imath
import numpy as np
import OpenEXR
import torch
import torch.nn.functional as F


def get_camera_to_world_rotation_matrix(camera):
    """
    Extract the camera-to-world rotation matrix from a camera object.
    Applies the same axis adjustments used by legacy normal transformation code.
    """
    c2w_matrix = camera.c2w.detach().cpu().numpy()
    camera_rotation = c2w_matrix[:3, :3].copy()
    camera_rotation[:, 1] = -camera_rotation[:, 1]
    camera_rotation = camera_rotation.T
    return camera_rotation


def load_normal_image(normal_path):
    """
    Load a normal image from EXR/PNG/JPG and return a normalized tensor [3, H, W] in [-1, 1].
    """
    if normal_path.endswith(".exr"):
        exr_file = OpenEXR.InputFile(normal_path)
        data_window = exr_file.header()["dataWindow"]
        size = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)

        float_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = exr_file.channels("RGB", float_type)
        rgb = [array.array("f", channels[i]).tolist() for i in range(3)]
        normal = np.zeros((size[1], size[0], 3), dtype=np.float32)
        for i in range(3):
            normal[:, :, i] = np.array(rgb[i]).reshape(size[1], size[0])
        exr_file.close()
    else:
        normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
        if normal is None:
            raise ValueError(f"Could not load image from {normal_path}")
        if len(normal.shape) == 3 and normal.shape[2] == 3:
            normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        normal = normal.astype(np.float32)
        if normal.max() > 1.0:
            normal = normal / 255.0

    if normal.shape[0] == 3:
        normal = normal.transpose(1, 2, 0)

    normal_tensor = torch.from_numpy(normal).float()
    if normal_tensor.shape[-1] == 3:
        normal_tensor = normal_tensor.permute(2, 0, 1)

    scale = torch.tensor([0.5, 0.5, -0.5], dtype=torch.float32)
    offset = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    normal_tensor = (normal_tensor - offset.view(3, 1, 1)) / scale.view(3, 1, 1)
    normal_tensor = F.normalize(normal_tensor, dim=0, eps=1e-6)
    return normal_tensor
