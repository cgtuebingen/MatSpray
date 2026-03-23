import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import json
from datetime import datetime


class GradientLogger:
    """Utility class for logging gradient statistics during training"""
    
    def __init__(self, log_dir: str = "gradient_logs", log_interval: int = 100):
        """
        Initialize gradient logger
        
        Args:
            log_dir: Directory to save gradient logs
            log_interval: How often to log gradients (every N iterations)
        """
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.gradient_stats = {}
        self.iteration = 0
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(log_dir, f"gradient_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    def log_gradients(self, model: nn.Module, iteration: int, loss: float, 
                     model_name: str = "model") -> Dict:
        """
        Log gradient statistics for a model
        
        Args:
            model: PyTorch model to log gradients for
            iteration: Current training iteration
            loss: Current loss value
            model_name: Name identifier for the model
            
        Returns:
            Dictionary containing gradient statistics
        """
        if iteration % self.log_interval != 0:
            return {}
            
        self.iteration = iteration
        stats = {
            'iteration': iteration,
            'loss': float(loss),
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': {}
        }
        
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Calculate gradient statistics
                grad_data = param.grad.data.cpu().numpy()
                grad_mean = float(np.mean(grad_data))
                grad_std = float(np.std(grad_data))
                grad_min = float(np.min(grad_data))
                grad_max = float(np.max(grad_data))
                grad_norm = float(param_norm.item())
                
                stats['parameters'][name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'min': grad_min,
                    'max': grad_max,
                    'shape': list(param.grad.shape),
                    'numel': param.grad.numel()
                }
            else:
                stats['parameters'][name] = {
                    'norm': 0.0,
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'shape': list(param.shape),
                    'numel': param.numel(),
                    'no_grad': True
                }
        
        # Calculate total gradient norm
        total_norm = total_norm ** (1. / 2)
        stats['total_gradient_norm'] = total_norm
        stats['param_count'] = param_count
        
        # Store stats
        if model_name not in self.gradient_stats:
            self.gradient_stats[model_name] = []
        self.gradient_stats[model_name].append(stats)
        
        # Save to file
        self._save_logs()
        
        return stats
    
    def log_optimizer_stats(self, optimizer: torch.optim.Optimizer, iteration: int,
                           model_name: str = "model") -> Dict:
        """
        Log optimizer statistics including learning rates
        
        Args:
            optimizer: PyTorch optimizer
            iteration: Current training iteration
            model_name: Name identifier for the model
            
        Returns:
            Dictionary containing optimizer statistics
        """
        if iteration % self.log_interval != 0:
            return {}
            
        stats = {
            'iteration': iteration,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'param_groups': []
        }
        
        for i, param_group in enumerate(optimizer.param_groups):
            group_stats = {
                'group_id': i,
                'lr': param_group.get('lr', 0.0),
                'weight_decay': param_group.get('weight_decay', 0.0),
                'betas': param_group.get('betas', (0.9, 0.999)),
                'eps': param_group.get('eps', 1e-8),
                'param_count': len(param_group['params'])
            }
            stats['param_groups'].append(group_stats)
        
        return stats
    
    def _save_logs(self):
        """Save gradient statistics to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.gradient_stats, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save gradient logs: {e}")
    
    def get_gradient_summary(self, model_name: str = None, 
                           last_n_iterations: int = 10) -> Dict:
        """
        Get a summary of gradient statistics
        
        Args:
            model_name: Specific model to summarize (None for all)
            last_n_iterations: Number of recent iterations to include
            
        Returns:
            Summary statistics
        """
        if model_name is None:
            models = list(self.gradient_stats.keys())
        else:
            models = [model_name] if model_name in self.gradient_stats else []
        
        summary = {}
        
        for model in models:
            if model not in self.gradient_stats:
                continue
                
            recent_stats = self.gradient_stats[model][-last_n_iterations:]
            
            if not recent_stats:
                continue
                
            # Calculate summary statistics
            total_norms = [s['total_gradient_norm'] for s in recent_stats]
            losses = [s['loss'] for s in recent_stats]
            
            summary[model] = {
                'avg_gradient_norm': np.mean(total_norms),
                'std_gradient_norm': np.std(total_norms),
                'min_gradient_norm': np.min(total_norms),
                'max_gradient_norm': np.max(total_norms),
                'avg_loss': np.mean(losses),
                'loss_trend': 'increasing' if losses[-1] > losses[0] else 'decreasing',
                'iterations_analyzed': len(recent_stats)
            }
        
        return summary
    
    def print_gradient_summary(self, model_name: str = None, 
                             last_n_iterations: int = 10):
        """Print a formatted summary of gradient statistics"""
        summary = self.get_gradient_summary(model_name, last_n_iterations)
        
        print(f"\n=== Gradient Summary (last {last_n_iterations} iterations) ===")
        for model, stats in summary.items():
            print(f"\nModel: {model}")
            print(f"  Average Gradient Norm: {stats['avg_gradient_norm']:.6f} ± {stats['std_gradient_norm']:.6f}")
            print(f"  Gradient Norm Range: [{stats['min_gradient_norm']:.6f}, {stats['max_gradient_norm']:.6f}]")
            print(f"  Average Loss: {stats['avg_loss']:.6f}")
            print(f"  Loss Trend: {stats['loss_trend']}")
            print(f"  Iterations Analyzed: {stats['iterations_analyzed']}")


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute the total gradient norm for a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Total gradient norm (L2 norm)
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def check_gradient_flow(model: nn.Module, threshold: float = 1e-6) -> Dict:
    """
    Check for gradient flow issues in a model
    
    Args:
        model: PyTorch model
        threshold: Threshold for considering gradients as "vanishing"
        
    Returns:
        Dictionary with gradient flow analysis
    """
    analysis = {
        'vanishing_gradients': [],
        'exploding_gradients': [],
        'no_gradients': [],
        'healthy_gradients': [],
        'total_params': 0,
        'params_with_grad': 0
    }
    
    for name, param in model.named_parameters():
        analysis['total_params'] += 1
        
        if param.grad is None:
            analysis['no_gradients'].append(name)
        else:
            analysis['params_with_grad'] += 1
            grad_norm = param.grad.data.norm(2).item()
            
            if grad_norm < threshold:
                analysis['vanishing_gradients'].append((name, grad_norm))
            elif grad_norm > 10.0:  # Arbitrary threshold for exploding gradients
                analysis['exploding_gradients'].append((name, grad_norm))
            else:
                analysis['healthy_gradients'].append((name, grad_norm))
    
    return analysis


def print_gradient_flow_analysis(analysis: Dict):
    """Print a formatted gradient flow analysis"""
    print(f"\n=== Gradient Flow Analysis ===")
    print(f"Total Parameters: {analysis['total_params']}")
    print(f"Parameters with Gradients: {analysis['params_with_grad']}")
    print(f"Parameters without Gradients: {len(analysis['no_gradients'])}")
    print(f"Vanishing Gradients: {len(analysis['vanishing_gradients'])}")
    print(f"Exploding Gradients: {len(analysis['exploding_gradients'])}")
    print(f"Healthy Gradients: {len(analysis['healthy_gradients'])}")
    
    if analysis['vanishing_gradients']:
        print(f"\nVanishing Gradients (norm < 1e-6):")
        for name, norm in analysis['vanishing_gradients'][:5]:  # Show first 5
            print(f"  {name}: {norm:.2e}")
    
    if analysis['exploding_gradients']:
        print(f"\nExploding Gradients (norm > 10):")
        for name, norm in analysis['exploding_gradients'][:5]:  # Show first 5
            print(f"  {name}: {norm:.2e}")
    
    if analysis['no_gradients']:
        print(f"\nParameters without Gradients:")
        for name in analysis['no_gradients'][:5]:  # Show first 5
            print(f"  {name}") 