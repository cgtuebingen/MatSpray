import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.gradient_utils import GradientLogger, compute_gradient_norm, check_gradient_flow, print_gradient_flow_analysis

class BaseColorMLP(nn.Module):
    """MLP for predicting base color from variable number of input colors and position"""
    def __init__(self, num_train_images: int, net_width: int = 64, enable_gradient_logging: bool = False,
                 use_id_conditioning: bool = False, num_gaussians: int = None, id_embedding_dim: int = 32):
        super().__init__()
        self.net_width = net_width
        self.degree = 4
        self.num_train_images = num_train_images
        self.enable_gradient_logging = enable_gradient_logging
        
        # Initialize gradient logger if enabled
        if self.enable_gradient_logging:
            self.gradient_logger = GradientLogger(log_dir="gradient_logs/base_color_mlp", log_interval=100)
        
        # Input: num_train_images colors (num_train_images * 3 values) + position (3 values) + positional encoding
        input_size = num_train_images * 3 + 3 + 3 * 2 * self.degree
        
        # Output: num_train_images (one output per training image)
        output_size = num_train_images
        
        self.main = nn.Sequential(
            nn.Linear(input_size, self.net_width * 2),
            nn.LeakyReLU(),
            nn.Linear(self.net_width * 2, self.net_width),
            nn.LeakyReLU(),
            nn.Linear(self.net_width, self.net_width),
            nn.LeakyReLU(),
        )
        
        self.output = nn.Sequential(
            nn.Linear(self.net_width, output_size),
        )
        
        # Custom activation to match the regular base_color_activation: sigmoid(x) * 0.77 + 0.03
        self.base_color_activation = lambda x: torch.sigmoid(x) * 0.77 + 0.03
    
    def positional_encoding(self, x):
        """Positional encoding of the inputs"""
        result = []
        for d in range(self.degree):
            for fn in [torch.sin, torch.cos]:
                result.append(fn(2.0 ** d * math.pi * x))
        return torch.cat(result, dim=-1)
    
    def forward(self, input_colors, positions=None, gaussian_ids=None):
        """
        Forward pass of the base color MLP
        
        Args:
            input_colors: [batch, num_train_images, 3] - num_train_images input colors per gaussian
            positions: [batch, 3] - positions of gaussians (required)
            gaussian_ids: [batch] - integer IDs of gaussians (deprecated, ignored)
            
        Returns:
            [batch, num_train_images, 1] - one output value for each training image
        """
        batch_size = input_colors.shape[0]
        
        # Flatten input colors: [batch, num_train_images * 3]
        flattened_colors = input_colors.reshape(batch_size, -1)
        
        assert positions is not None, "positions must be provided"
        # Positional encoding of positions
        encoded_positions = self.positional_encoding(positions)
        # Concatenate all inputs
        x = torch.cat([flattened_colors, positions, encoded_positions], dim=-1)
        
        # Process through network
        x = self.main(x)
        # Apply output layer to get num_train_images channels
        x = self.output(x)
        
        # Reshape to [batch, num_train_images, 1]
        x = x.view(batch_size, self.num_train_images, 1)
        
        # Apply softmax across the training images dimension
        x = F.softmax(x, dim=1)

        return x
    
    def log_gradients(self, iteration: int, loss: float):
        """Log gradient statistics if enabled"""
        if self.enable_gradient_logging:
            return self.gradient_logger.log_gradients(self, iteration, loss, "base_color_mlp")
        return {}
    
    def get_gradient_summary(self, last_n_iterations: int = 10):
        """Get gradient summary if logging is enabled"""
        if self.enable_gradient_logging:
            return self.gradient_logger.get_gradient_summary("base_color_mlp", last_n_iterations)
        return {}
    
    def print_gradient_analysis(self):
        """Print gradient flow analysis"""
        if self.enable_gradient_logging:
            analysis = check_gradient_flow(self)
            print_gradient_flow_analysis(analysis)
            print(f"Total gradient norm: {compute_gradient_norm(self):.6f}")
    
    @torch.no_grad()
    def prune_embeddings(self, valid_mask: torch.Tensor):
        """Prune ID embeddings according to a boolean mask over gaussians (deprecated, no-op)"""
        pass
    
    @torch.no_grad()
    def extend_embeddings(self, num_new: int, device: torch.device = None):
        """Extend ID embeddings to accommodate new gaussians (deprecated, no-op)"""
        pass


class RoughnessMLP(nn.Module):
    """MLP for predicting roughness from variable number of input scalars and position"""
    def __init__(self, num_train_images: int, net_width: int = 64, enable_gradient_logging: bool = False,
                 use_id_conditioning: bool = False, num_gaussians: int = None, id_embedding_dim: int = 32):
        super().__init__()
        self.net_width = net_width
        self.degree = 4
        self.num_train_images = num_train_images
        self.enable_gradient_logging = enable_gradient_logging
        
        # Initialize gradient logger if enabled
        if self.enable_gradient_logging:
            self.gradient_logger = GradientLogger(log_dir="gradient_logs/roughness_mlp", log_interval=100)
        
        # Input: num_train_images scalars (num_train_images values) + position (3 values) + positional encoding
        input_size = num_train_images + 3 + 3 * 2 * self.degree
        
        # Output: num_train_images (one roughness value per training image)
        output_size = num_train_images
        
        self.main = nn.Sequential(
            nn.Linear(input_size, self.net_width * 2),
            nn.LeakyReLU(),
            nn.Linear(self.net_width * 2, self.net_width),
            nn.LeakyReLU(),
            nn.Linear(self.net_width, self.net_width),
            nn.LeakyReLU(),
        )
        
        self.output = nn.Sequential(
            nn.Linear(self.net_width, output_size),
        )
    
    def positional_encoding(self, x):
        """Positional encoding of the inputs"""
        result = []
        for d in range(self.degree):
            for fn in [torch.sin, torch.cos]:
                result.append(fn(2.0 ** d * math.pi * x))
        return torch.cat(result, dim=-1)
    
    def forward(self, input_scalars, positions=None, gaussian_ids=None):
        """
        Forward pass of the roughness MLP
        
        Args:
            input_scalars: [batch, num_train_images, 1] - num_train_images input scalars per gaussian
            positions: [batch, 3] - positions of gaussians (required)
            gaussian_ids: [batch] - integer IDs of gaussians (deprecated, ignored)
            
        Returns:
            [batch, num_train_images, 1] - roughness values for each training image
        """
        batch_size = input_scalars.shape[0]
        
        # Flatten input scalars: [batch, num_train_images]
        flattened_scalars = input_scalars.reshape(batch_size, -1)
        
        assert positions is not None, "positions must be provided"
        # Positional encoding of positions
        encoded_positions = self.positional_encoding(positions)
        # Concatenate all inputs
        x = torch.cat([flattened_scalars, positions, encoded_positions], dim=-1)
        
        # Process through network
        x = self.main(x)
        x = self.output(x)
        
        # Reshape to [batch, num_train_images, 1]
        x = x.view(batch_size, self.num_train_images, 1)
        
        # Apply softmax across the training images dimension
        x = F.softmax(x, dim=1)
        
        return x
    
    def log_gradients(self, iteration: int, loss: float):
        """Log gradient statistics if enabled"""
        if self.enable_gradient_logging:
            return self.gradient_logger.log_gradients(self, iteration, loss, "roughness_mlp")
        return {}
    
    def get_gradient_summary(self, last_n_iterations: int = 10):
        """Get gradient summary if logging is enabled"""
        if self.enable_gradient_logging:
            return self.gradient_logger.get_gradient_summary("roughness_mlp", last_n_iterations)
        return {}
    
    def print_gradient_analysis(self):
        """Print gradient flow analysis"""
        if self.enable_gradient_logging:
            analysis = check_gradient_flow(self)
            print_gradient_flow_analysis(analysis)
            print(f"Total gradient norm: {compute_gradient_norm(self):.6f}")
    
    @torch.no_grad()
    def prune_embeddings(self, valid_mask: torch.Tensor):
        """Prune ID embeddings according to a boolean mask over gaussians (deprecated, no-op)"""
        pass
    
    @torch.no_grad()
    def extend_embeddings(self, num_new: int, device: torch.device = None):
        """Extend ID embeddings to accommodate new gaussians (deprecated, no-op)"""
        pass


class MetallicMLP(nn.Module):
    """MLP for predicting metallic from variable number of input scalars and position"""
    def __init__(self, num_train_images: int, net_width: int = 64, enable_gradient_logging: bool = False,
                 use_id_conditioning: bool = False, num_gaussians: int = None, id_embedding_dim: int = 32):
        super().__init__()
        self.net_width = net_width
        self.degree = 4
        self.num_train_images = num_train_images
        self.enable_gradient_logging = enable_gradient_logging
        
        # Initialize gradient logger if enabled
        if self.enable_gradient_logging:
            self.gradient_logger = GradientLogger(log_dir="gradient_logs/metallic_mlp", log_interval=100)
        
        # Input: num_train_images scalars (num_train_images values) + position (3 values) + positional encoding
        input_size = num_train_images + 3 + 3 * 2 * self.degree
        
        # Output: num_train_images (one metallic value per training image)
        output_size = num_train_images
        
        self.main = nn.Sequential(
            nn.Linear(input_size, self.net_width * 2),
            nn.LeakyReLU(),
            nn.Linear(self.net_width * 2, self.net_width),
            nn.LeakyReLU(),
            nn.Linear(self.net_width, self.net_width),
            nn.LeakyReLU(),
        )
        
        self.output = nn.Sequential(
            nn.Linear(self.net_width, output_size),
        )
    
    def positional_encoding(self, x):
        """Positional encoding of the inputs"""
        result = []
        for d in range(self.degree):
            for fn in [torch.sin, torch.cos]:
                result.append(fn(2.0 ** d * math.pi * x))
        return torch.cat(result, dim=-1)
    
    def forward(self, input_scalars, positions=None, gaussian_ids=None):
        """
        Forward pass of the metallic MLP
        
        Args:
            input_scalars: [batch, num_train_images, 1] - num_train_images input scalars per gaussian
            positions: [batch, 3] - positions of gaussians (required)
            gaussian_ids: [batch] - integer IDs of gaussians (deprecated, ignored)
            
        Returns:
            [batch, num_train_images, 1] - metallic values for each training image
        """
        batch_size = input_scalars.shape[0]
        
        # Flatten input scalars: [batch, num_train_images]
        flattened_scalars = input_scalars.reshape(batch_size, -1)
        
        assert positions is not None, "positions must be provided"
        # Positional encoding of positions
        encoded_positions = self.positional_encoding(positions)
        # Concatenate all inputs
        x = torch.cat([flattened_scalars, positions, encoded_positions], dim=-1)
        
        # Process through network
        x = self.main(x)
        x = self.output(x)
        
        # Reshape to [batch, num_train_images, 1]
        x = x.view(batch_size, self.num_train_images, 1)
        
        # Apply softmax across the training images dimension
        x = F.softmax(x, dim=1)
        
        return x
    
    def log_gradients(self, iteration: int, loss: float):
        """Log gradient statistics if enabled"""
        if self.enable_gradient_logging:
            return self.gradient_logger.log_gradients(self, iteration, loss, "metallic_mlp")
        return {}
    
    def get_gradient_summary(self, last_n_iterations: int = 10):
        """Get gradient summary if logging is enabled"""
        if self.enable_gradient_logging:
            return self.gradient_logger.get_gradient_summary("metallic_mlp", last_n_iterations)
        return {}
    
    def print_gradient_analysis(self):
        """Print gradient flow analysis"""
        if self.enable_gradient_logging:
            analysis = check_gradient_flow(self)
            print_gradient_flow_analysis(analysis)
            print(f"Total gradient norm: {compute_gradient_norm(self):.6f}")
    
    @torch.no_grad()
    def prune_embeddings(self, valid_mask: torch.Tensor):
        """Prune ID embeddings according to a boolean mask over gaussians (deprecated, no-op)"""
        pass
    
    @torch.no_grad()
    def extend_embeddings(self, num_new: int, device: torch.device = None):
        """Extend ID embeddings to accommodate new gaussians (deprecated, no-op)"""
        pass


class CombinedMLP(nn.Module):
    """Combined MLP that predicts base_color, roughness, and metallic weights simultaneously.
    
    Instead of having three separate MLPs, this combines them into a single network that
    shares the initial layers and has three separate output heads.
    """
    def __init__(self, num_train_images: int, net_width: int = 64, enable_gradient_logging: bool = False,
                 use_id_conditioning: bool = False, num_gaussians: int = None, id_embedding_dim: int = 32):
        super().__init__()
        self.net_width = net_width
        self.degree = 4
        self.num_train_images = num_train_images
        self.enable_gradient_logging = enable_gradient_logging
        
        # Initialize gradient logger if enabled
        if self.enable_gradient_logging:
            self.gradient_logger = GradientLogger(log_dir="gradient_logs/combined_mlp", log_interval=100)
        
        # Input: base_color (num_train_images * 3) + roughness (num_train_images) + metallic (num_train_images) 
        #        + position (3) + positional encoding (3 * 2 * degree)
        # Total: num_train_images * 3 + num_train_images + num_train_images + 3 + 3 * 2 * degree
        #      = num_train_images * 5 + 3 + 3 * 2 * degree
        input_size = num_train_images * 5 + 3 + 3 * 2 * self.degree
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(input_size, self.net_width * 2),
            nn.LeakyReLU(),
            nn.Linear(self.net_width * 2, self.net_width * 2),
            nn.LeakyReLU(),
            nn.Linear(self.net_width * 2, self.net_width),
            nn.LeakyReLU(),
        )
        
        # Output heads for each property
        # Each outputs num_train_images weights (one per training image)
        self.base_color_head = nn.Linear(self.net_width, num_train_images)
        self.roughness_head = nn.Linear(self.net_width, num_train_images)
        self.metallic_head = nn.Linear(self.net_width, num_train_images)
        
        # Custom activation to match the regular base_color_activation: sigmoid(x) * 0.77 + 0.03
        self.base_color_activation = lambda x: torch.sigmoid(x) * 0.77 + 0.03
    
    def positional_encoding(self, x):
        """Positional encoding of the inputs"""
        result = []
        for d in range(self.degree):
            for fn in [torch.sin, torch.cos]:
                result.append(fn(2.0 ** d * math.pi * x))
        return torch.cat(result, dim=-1)
    
    def forward(self, input_base_colors, input_roughness, input_metallic, positions=None, gaussian_ids=None):
        """
        Forward pass of the combined MLP
        
        Args:
            input_base_colors: [batch, num_train_images, 3] - base colors per gaussian
            input_roughness: [batch, num_train_images, 1] - roughness per gaussian
            input_metallic: [batch, num_train_images, 1] - metallic per gaussian
            positions: [batch, 3] - positions of gaussians (required)
            gaussian_ids: [batch] - integer IDs of gaussians (deprecated, ignored)
            
        Returns:
            Tuple of three tensors:
            - base_color_weights: [batch, num_train_images, 1] - weights for base color
            - roughness_weights: [batch, num_train_images, 1] - weights for roughness
            - metallic_weights: [batch, num_train_images, 1] - weights for metallic
        """
        batch_size = input_base_colors.shape[0]
        
        # Flatten inputs
        flattened_base_colors = input_base_colors.reshape(batch_size, -1)  # [batch, num_train_images * 3]
        flattened_roughness = input_roughness.reshape(batch_size, -1)  # [batch, num_train_images]
        flattened_metallic = input_metallic.reshape(batch_size, -1)  # [batch, num_train_images]
        
        assert positions is not None, "positions must be provided"
        # Positional encoding of positions
        encoded_positions = self.positional_encoding(positions)
        
        # Concatenate all inputs
        x = torch.cat([flattened_base_colors, flattened_roughness, flattened_metallic, 
                       positions, encoded_positions], dim=-1)
        
        # Process through shared backbone
        x = self.shared(x)
        
        # Get outputs from each head
        base_color_out = self.base_color_head(x)  # [batch, num_train_images]
        roughness_out = self.roughness_head(x)  # [batch, num_train_images]
        metallic_out = self.metallic_head(x)  # [batch, num_train_images]
        
        # Reshape and apply softmax
        base_color_weights = F.softmax(base_color_out.view(batch_size, self.num_train_images, 1), dim=1)
        roughness_weights = F.softmax(roughness_out.view(batch_size, self.num_train_images, 1), dim=1)
        metallic_weights = F.softmax(metallic_out.view(batch_size, self.num_train_images, 1), dim=1)
        
        return base_color_weights, roughness_weights, metallic_weights
    
    def log_gradients(self, iteration: int, loss: float):
        """Log gradient statistics if enabled"""
        if self.enable_gradient_logging:
            return self.gradient_logger.log_gradients(self, iteration, loss, "combined_mlp")
        return {}
    
    def get_gradient_summary(self, last_n_iterations: int = 10):
        """Get gradient summary if logging is enabled"""
        if self.enable_gradient_logging:
            return self.gradient_logger.get_gradient_summary("combined_mlp", last_n_iterations)
        return {}
    
    def print_gradient_analysis(self):
        """Print gradient flow analysis"""
        if self.enable_gradient_logging:
            analysis = check_gradient_flow(self)
            print_gradient_flow_analysis(analysis)
            print(f"Total gradient norm: {compute_gradient_norm(self):.6f}")
    
    @torch.no_grad()
    def prune_embeddings(self, valid_mask: torch.Tensor):
        """Prune ID embeddings according to a boolean mask over gaussians (deprecated, no-op)"""
        pass
    
    @torch.no_grad()
    def extend_embeddings(self, num_new: int, device: torch.device = None):
        """Extend ID embeddings to accommodate new gaussians (deprecated, no-op)"""
        pass


class NormalMLP(nn.Module):
    """MLP for predicting normal from variable number of input normals, position, and camera direction"""
    def __init__(self, num_train_images: int, net_width: int = 64, enable_gradient_logging: bool = False,
                 use_id_conditioning: bool = False, num_gaussians: int = None, id_embedding_dim: int = 32):
        super().__init__()
        self.net_width = net_width
        self.degree = 4
        self.num_train_images = num_train_images
        self.enable_gradient_logging = enable_gradient_logging
        
        # Initialize gradient logger if enabled
        if self.enable_gradient_logging:
            self.gradient_logger = GradientLogger(log_dir="gradient_logs/normal_mlp", log_interval=100)
        
        # Input: num_train_images normals (num_train_images * 3 values) + position (3 values) + 
        # camera direction (3 values) + positional encoding
        input_size = num_train_images * 3 + 3 + 3 + 3 * 2 * self.degree
        
        # Output: num_train_images (one output per training image)
        output_size = num_train_images
        
        self.main = nn.Sequential(
            nn.Linear(input_size, self.net_width * 2),
            nn.LeakyReLU(),
            nn.Linear(self.net_width * 2, self.net_width),
            nn.LeakyReLU(),
            nn.Linear(self.net_width, self.net_width),
            nn.LeakyReLU(),
        )
        
        self.output = nn.Sequential(
            nn.Linear(self.net_width, output_size),
        )
    
    def positional_encoding(self, x):
        """Positional encoding of the inputs"""
        result = []
        for d in range(self.degree):
            for fn in [torch.sin, torch.cos]:
                result.append(fn(2.0 ** d * math.pi * x))
        return torch.cat(result, dim=-1)
    
    def forward(self, input_normals, positions=None, camera_direction=None, gaussian_ids=None):
        """
        Forward pass of the normal MLP
        
        Args:
            input_normals: [batch, num_train_images, 3] - num_train_images input normals per gaussian
            positions: [batch, 3] - positions of gaussians (required)
            camera_direction: [batch, 3] - camera direction vector (required)
            gaussian_ids: [batch] - integer IDs of gaussians (deprecated, ignored)
            
        Returns:
            [batch, num_train_images, 1] - one output value for each training image
        """
        batch_size = input_normals.shape[0]
        
        # Flatten input normals: [batch, num_train_images * 3]
        flattened_normals = input_normals.reshape(batch_size, -1)
        
        assert positions is not None and camera_direction is not None, "positions and camera_direction must be provided"
        # Positional encoding of positions
        encoded_positions = self.positional_encoding(positions)
        # Concatenate all inputs: normals + positions + camera_direction + encoded_positions
        x = torch.cat([flattened_normals, positions, camera_direction, encoded_positions], dim=-1)
        
        # Process through network
        x = self.main(x)
        # Apply output layer to get num_train_images channels
        x = self.output(x)
        
        # Reshape to [batch, num_train_images, 1]
        x = x.view(batch_size, self.num_train_images, 1)
        
        # Apply softmax across the training images dimension
        x = F.softmax(x, dim=1)

        return x
    
    def log_gradients(self, iteration: int, loss: float):
        """Log gradient statistics if enabled"""
        if self.enable_gradient_logging:
            return self.gradient_logger.log_gradients(self, iteration, loss, "normal_mlp")
        return {}
    
    def get_gradient_summary(self, last_n_iterations: int = 10):
        """Get gradient summary if logging is enabled"""
        if self.enable_gradient_logging:
            return self.gradient_logger.get_gradient_summary("normal_mlp", last_n_iterations)
        return {}
    
    def print_gradient_analysis(self):
        """Print gradient flow analysis"""
        if self.enable_gradient_logging:
            analysis = check_gradient_flow(self)
            print_gradient_flow_analysis(analysis)
            print(f"Total gradient norm: {compute_gradient_norm(self):.6f}")
    
    @torch.no_grad()
    def prune_embeddings(self, valid_mask: torch.Tensor):
        """Prune ID embeddings according to a boolean mask over gaussians (deprecated, no-op)"""
        pass
    
    @torch.no_grad()
    def extend_embeddings(self, num_new: int, device: torch.device = None):
        """Extend ID embeddings to accommodate new gaussians (deprecated, no-op)"""
        pass
        