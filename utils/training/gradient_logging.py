def log_mlp_gradients(gaussians, iteration, loss):
    """
    Log gradient stats for active MLPs and print periodic summaries.
    """
    try:
        use_combined_mlp = getattr(gaussians, "use_combined_mlp", False)
        if use_combined_mlp and hasattr(gaussians, "combined_mlp") and gaussians.combined_mlp is not None:
            combined_stats = gaussians.combined_mlp.log_gradients(iteration, loss.item())
            if iteration % 1000 == 0:
                print(f"\n=== MLP Gradient Summary (Iteration {iteration}) ===")
                print(f"Total Loss: {loss.item():.6f}")
                if combined_stats:
                    print(f"Combined MLP - Total Gradient Norm: {combined_stats.get('total_gradient_norm', 0):.6f}")
                if hasattr(gaussians, "normal_mlp") and gaussians.normal_mlp is not None:
                    normal_stats = gaussians.normal_mlp.log_gradients(iteration, loss.item())
                    if normal_stats:
                        print(f"Normal MLP - Total Gradient Norm: {normal_stats.get('total_gradient_norm', 0):.6f}")
            return

        base_color_stats = gaussians.base_color_mlp.log_gradients(iteration, loss.item())
        roughness_stats = gaussians.roughness_mlp.log_gradients(iteration, loss.item())
        metallic_stats = gaussians.metallic_mlp.log_gradients(iteration, loss.item())
        if iteration % 1000 == 0:
            print(f"\n=== MLP Gradient Summary (Iteration {iteration}) ===")
            print(f"Total Loss: {loss.item():.6f}")
            if base_color_stats:
                print(f"Base Color MLP - Total Gradient Norm: {base_color_stats.get('total_gradient_norm', 0):.6f}")
            if roughness_stats:
                print(f"Roughness MLP - Total Gradient Norm: {roughness_stats.get('total_gradient_norm', 0):.6f}")
            if metallic_stats:
                print(f"Metallic MLP - Total Gradient Norm: {metallic_stats.get('total_gradient_norm', 0):.6f}")
            if hasattr(gaussians, "normal_mlp") and gaussians.normal_mlp is not None:
                normal_stats = gaussians.normal_mlp.log_gradients(iteration, loss.item())
                if normal_stats:
                    print(f"Normal MLP - Total Gradient Norm: {normal_stats.get('total_gradient_norm', 0):.6f}")
    except Exception as e:
        print(f"Warning: Failed to log MLP gradients: {e}")
