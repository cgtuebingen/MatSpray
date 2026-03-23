import torch

from scene.mlp import BaseColorMLP, MetallicMLP, NormalMLP, RoughnessMLP


def mlp_gradient_flow_test():
    torch.manual_seed(42)
    num_gaussians = 32
    num_train_images = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_color_mlp = BaseColorMLP(num_train_images=num_train_images, net_width=32).to(device)
    roughness_mlp = RoughnessMLP(num_train_images=num_train_images, net_width=32).to(device)
    metallic_mlp = MetallicMLP(num_train_images=num_train_images, net_width=32).to(device)
    normal_mlp = NormalMLP(num_train_images=num_train_images, net_width=32).to(device)

    dummy_base_colors = torch.ones((num_gaussians, num_train_images, 3), device=device)
    dummy_roughness = torch.full((num_gaussians, num_train_images, 1), 0.5, device=device)
    dummy_metallic = torch.zeros((num_gaussians, num_train_images, 1), device=device)
    dummy_normals = torch.tensor([0.0, 1.0, 0.0], device=device).expand(num_gaussians, num_train_images, 3)
    positions = torch.randn((num_gaussians, 3), device=device)
    camera_direction = torch.tensor([0.0, 0.0, -1.0], device=device)

    base_color_out = base_color_mlp(dummy_base_colors, positions)
    roughness_out = roughness_mlp(dummy_roughness, positions)
    metallic_out = metallic_mlp(dummy_metallic, positions)
    normal_out = normal_mlp(dummy_normals, positions, camera_direction)
    loss = base_color_out.sum() + roughness_out.sum() + metallic_out.sum() + normal_out.sum()
    loss.backward()

    def grad_norm(module):
        return sum((p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None))

    print("BaseColorMLP grad norm:", grad_norm(base_color_mlp))
    print("RoughnessMLP grad norm:", grad_norm(roughness_mlp))
    print("MetallicMLP grad norm:", grad_norm(metallic_mlp))
    print("NormalMLP grad norm:", grad_norm(normal_mlp))
    if grad_norm(base_color_mlp) == 0:
        print("[WARNING] BaseColorMLP has zero gradients!")
    if grad_norm(roughness_mlp) == 0:
        print("[WARNING] RoughnessMLP has zero gradients!")
    if grad_norm(metallic_mlp) == 0:
        print("[WARNING] MetallicMLP has zero gradients!")
    if grad_norm(normal_mlp) == 0:
        print("[WARNING] NormalMLP has zero gradients!")

    print(f"\nNormal MLP output shape: {normal_out.shape}")
    print(f"Expected shape: [{num_gaussians}, {num_train_images}, 1]")
    print(f"Softmax sum per gaussian: {normal_out.sum(dim=1).flatten()}")
    print("✓ Normal MLP test completed successfully!")
