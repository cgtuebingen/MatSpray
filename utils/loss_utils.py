#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from kornia.filters import laplacian, spatial_gradient

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def scale_invariant_l1_loss(pred, gt, mask=None, eps=1e-8):
    """
    Scale-invariant L1 loss.
    
    Computes the optimal scale factor s that minimizes ||s*pred - gt||_2^2,
    then returns the L1 loss between the scaled prediction and ground truth.
    
    The optimal scale factor (in least squares sense) is:
        s = (pred · gt) / (pred · pred)
    
    Args:
        pred: Predicted tensor [C, H, W] or [B, C, H, W]
        gt: Ground truth tensor, same shape as pred
        mask: Optional binary mask, same spatial dimensions as pred/gt
        eps: Small epsilon for numerical stability
    
    Returns:
        Scale-invariant L1 loss (scalar)
    """
    if mask is not None:
        # Apply mask
        pred_masked = pred
        gt_masked = gt * mask
    else:
        pred_masked = pred
        gt_masked = gt
    
    # Compute optimal scale factor (least squares solution)
    # s = (pred · gt) / (pred · pred)
    pred_dot_gt = (pred_masked * gt_masked).sum()
    pred_dot_pred = (pred_masked * pred_masked).sum()
    
    # Compute scale with numerical stability
    scale = pred_dot_gt / (pred_dot_pred + eps)
    
    # Clamp scale to reasonable range to avoid extreme values
    scale = torch.clamp(scale, min=0.1, max=10.0)
    
    # Compute L1 loss with scaled prediction
    scaled_pred = scale * pred_masked
    loss = torch.abs(scaled_pred - gt_masked).mean()
    
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def cal_gradient(data):
    """
    data: [1, C, H, W]
    """
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(data.device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(data.device)

    weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    grad_x = F.conv2d(data, weight_x, padding='same')
    grad_y = F.conv2d(data, weight_y, padding='same')
    gradient = torch.abs(grad_x) + torch.abs(grad_y)

    return gradient


def bilateral_smooth_loss(data, image, mask):
    """
    image: [C, H, W]
    data: [C, H, W]
    mask: [C, H, W]
    """
    rgb_grad = cal_gradient(image.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]
    data_grad = cal_gradient(data.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]

    smooth_loss = (data_grad * (-rgb_grad).exp() * mask).mean()

    return smooth_loss


def second_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=2)[0, :, [0, 2]].abs() * torch.exp(-10*spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()


def first_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()

def first_order_edge_aware_norm_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].norm(dim=1, keepdim=True))).sum(1).mean()

def first_order_loss(data):
    return spatial_gradient(data[None], order=1)[0].abs().sum(1).mean()

def tv_loss(depth):
    # return spatial_gradient(data[None], order=2)[0, :, [0, 2]].abs().sum(1).mean()
    h_tv = torch.square(depth[..., 1:, :] - depth[..., :-1, :]).mean()
    w_tv = torch.square(depth[..., :, 1:] - depth[..., :, :-1]).mean()
    return h_tv + w_tv
