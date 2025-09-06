"""
SEA-RAFT model implementation.
"""

import torch
from torch.nn import functional as F
from types import SimpleNamespace

# Import core modules - these need to be accessible from the package
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.raft import RAFT
from core.utils.utils import load_ckpt

cfg = {
    "name": "spring-M",
    "dataset": "spring",
    "gpus": [0, 1, 2, 3, 4, 5, 6, 7],

    "use_var": True,
    "var_min": 0,
    "var_max": 10,
    "pretrain": "resnet34",
    "initial_dim": 64,
    "block_dims": [64, 128, 256],
    "radius": 4,
    "dim": 128,
    "num_blocks": 2,
    "iters": 4,

    "image_size": [540, 960],
    "scale": -1,
    "batch_size": 32,
    "epsilon": 1e-8,
    "lr": 4e-4,
    "wdecay": 1e-5,
    "dropout": 0,
    "clip": 1.0,
    "gamma": 0.85,
    "num_steps": 120000,
    
    "restore_ckpt": None,
    "coarse_config": None
}


class SEA_RAFT(torch.nn.Module):
    """
    SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow
    
    Args:
        checkpoint_path (str): Path to the pretrained checkpoint file
    """
    def __init__(self, checkpoint_path):
        super().__init__()
        global cfg
        raft_cfg = SimpleNamespace(**cfg)
        self.raft = RAFT(raft_cfg)
        self.raft.eval()
        load_ckpt(self.raft, checkpoint_path)

    @torch.no_grad()
    def forward(self, image1, image2, iters=4, scale=0.5):
        """
        Forward pass of SEA-RAFT
        
        Args:
            image1 (torch.Tensor): First image [B, 3, H, W] in [0, 255]
            image2 (torch.Tensor): Second image [B, 3, H, W] in [0, 255]
            iters (int): Number of iterations for refinement
            scale (float): Scale factor for input images
            
        Returns:
            tuple: (flow, info) where flow is [B, 2, H, W] and info is [B, 1, H, W]
        """
        # image: [B, 3, H, W] in [0, 255]
        img1 = F.interpolate(image1, scale_factor=scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=scale, mode='bilinear', align_corners=False)
        H, W = img1.shape[2:]
        output = self.raft(img1, img2, iters=4, test_mode=True)
        flow = output['flow'][-1]
        info = output['info'][-1]
        flow_down = F.interpolate(flow, scale_factor=1 / scale, mode='bilinear', align_corners=False)
        flow_down = flow_down * (1.0 / scale)
        info_down = F.interpolate(info, scale_factor=1 / scale, mode='area')
        return flow_down, info_down


def remap(image, flow):
    """
    Remap image using optical flow
    
    Args:
        image (torch.Tensor): Image tensor [B, C, H, W] or [C, H, W]
        flow (torch.Tensor): Flow tensor [B, H, W, 2] or [H, W, 2]
        
    Returns:
        torch.Tensor: Remapped image
    """
    if image.ndim == 3:
        image = image[None]
    if flow.ndim == 3:
        flow = flow[None]

    _, c, h, w = image.shape

    # Normalize grid to [-1, 1]
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).float().type_as(flow)[None] + flow
    grid[..., 0] = 2.0 * grid[..., 0] / (w - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (h - 1) - 1.0
    remapped = torch.nn.functional.grid_sample(image, grid, mode='bilinear', align_corners=True)
    return remapped
