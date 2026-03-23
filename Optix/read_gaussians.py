import torch
import numpy as np
import struct

class gaussians:
    def __init__(self):
        self.active_sh_degree = 0
        self._xyz = None
        self._normal = None
        self._shs_dc = None
        self._shs_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None
        self.max_radii2D = None
        self.spatial_lr_scale = 1.0
        self.xyz_gradient_accum = None
        self.normal_gradient_accum = None
        self.denom = None

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)

        (self.active_sh_degree,
         self._xyz,
         self._normal,
         self._shs_dc,
         self._shs_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         normal_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args[:14]

        self.xyz_gradient_accum = xyz_gradient_accum
        self.normal_gradient_accum = normal_gradient_accum
        self.denom = denom

        print("xyz: ",torch.max(self._xyz), torch.min(self._xyz))
        print("scaling: ", torch.max(torch.exp(self._scaling)), torch.min(torch.exp(self._scaling)))
        print("Rotations: ", torch.max(self._rotation), torch.min(self._rotation))

        return first_iter

    def save_to_binary(self, file_path):
        """Save all attributes that have 'self.' in front into a binary file."""
        with open(file_path, 'wb') as f:
            # Write scalar values (e.g., active_sh_degree, spatial_lr_scale)
            f.write(struct.pack('i', self.active_sh_degree))  # 'i' for integer
            f.write(struct.pack('f', self.spatial_lr_scale))  # 'f' for float

            def write_tensor(tensor):
                if tensor is not None:
                    tensor_np = tensor.detach().cpu().numpy().astype(np.float32)

                    # Reshape if tensor has more than 2 dimensions
                    if tensor_np.ndim > 2:
                        print(
                            f"Reshaping tensor from shape {tensor_np.shape} to {tensor_np.shape[0]}x{tensor_np.shape[1] * tensor_np.shape[2]}")
                        tensor_np = tensor_np.reshape(tensor_np.shape[0], -1)  # Flatten the last dimensions

                    rows, cols = tensor_np.shape
                    print(f"Writing tensor of size: {rows}x{cols}")  # Print the tensor's shape
                    f.write(struct.pack('i', rows))  # Write number of rows
                    f.write(struct.pack('i', cols))  # Write number of columns
                    f.write(tensor_np.tobytes())  # Write tensor data
                else:
                    f.write(struct.pack('i', 0))  # Write size 0 for None

            # Write all tensors with their shapes
            write_tensor(self._xyz)
            write_tensor(self._normal)
            write_tensor(self._shs_dc)
            write_tensor(self._shs_rest)
            write_tensor(torch.exp(self._scaling))
            write_tensor(torch.nn.functional.normalize(self._rotation))
            write_tensor(torch.sigmoid(self._opacity))

        print(f"Saved data to {file_path}")

gaussians = gaussians()
gaussians.create_from_ckpt("/path/to/your/checkpoint/chkpnt30000.pth")
print(gaussians._xyz.shape)
print(gaussians._rotation.shape)
gaussians.save_to_binary("./models/ship_3dgs_30000.bin")
