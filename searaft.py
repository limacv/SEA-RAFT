import torch
from torch.nn import functional as F

from core.raft import RAFT
from core.utils.utils import load_ckpt
from types import SimpleNamespace

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
    def __init__(self, checkpoint_path):
        super().__init__()
        global cfg
        raft_cfg = SimpleNamespace(**cfg)
        self.raft = RAFT(raft_cfg)
        self.raft.eval()
        load_ckpt(self.raft, checkpoint_path)

    @torch.no_grad()
    def forward(self, image1, image2, iters=4, scale=0.5):
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


def remap(image, flow):   # image in [B, C, H, W], flow in [B, H, W, 2]
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


if __name__ == "__main__":
    import cv2
    from core.utils.flow_viz import flow_to_image
    import time
    import imageio

    raft = SEA_RAFT("./checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth")
    image1 = cv2.imread("./custom/image1.jpg")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread("./custom/image2.jpg")
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB )
    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    H, W = image1.shape[1:]
    image1 = image1[None].cuda()
    image2 = image2[None].cuda()
    raft = raft.cuda()
    for i in range(4):
        start = time.time()
        flow, info = raft(image1, image2)
        end = time.time()
        print(f"time: {end - start:.4f}s")

    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"./custom/flow_new.jpg", flow_vis)
    flow = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
    remapped_image = remap(image2, flow)
    imageio.imwrite(f"./custom/remapped_new.jpg", remapped_image[0].permute(1, 2, 0).cpu().numpy().astype('uint8'))
