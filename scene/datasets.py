import os
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.camera_utils import loadCam
from utils.loss_utils import l1_loss, ssim

class GSDataset(Dataset):
    def __init__(self, cameras, scene, args, pipe=None, scale=1):
        self.pre_load = pipe.pre_load if hasattr(pipe, 'pre_load') else False
        self.cameras = cameras
        self.scale = scale
        self.args = args
        
        if hasattr(pipe, 'blur_level') and pipe.blur_level > 0:
            self.blur_level = pipe.blur_level
        else:
            self.blur_level = 0
            
        if len(self.cameras) > 300:
            self.pre_load = False
        
        if self.pre_load:
            camera_list = []
            for id, c in enumerate(self.cameras):
                camera_list.append(loadCam(args, id, c, self.scale))
            self.cameras = camera_list

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        if self.pre_load:
            viewpoint_cam = self.cameras[idx]
        else:
            c = self.cameras[idx]
            viewpoint_cam = loadCam(self.args, idx, c, self.scale)
        x = {
            "FoVx": viewpoint_cam.FoVx,
            "FoVy": viewpoint_cam.FoVy,
            "image_name": viewpoint_cam.image_name,
            "image_height": viewpoint_cam.image_height,
            "image_width": viewpoint_cam.image_width,
            "camera_center": viewpoint_cam.camera_center,
            "world_view_transform": viewpoint_cam.world_view_transform,
            "full_proj_transform": viewpoint_cam.full_proj_transform,
        }
        y = viewpoint_cam.original_image

        if self.blur_level > 0:
            y = F.gaussian_blur(y, self.blur_level * 30 +1)
        
        return x, y
    
    def contract_to_unisphere(
        self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: float = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x