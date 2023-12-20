import os
import torch
from torch.utils.data import Dataset
from utils.camera_utils import loadCam

class GSDataset(Dataset):
    def __init__(self, cameras, scale, args):
        self.cameras = cameras
        self.scale = scale
        self.args = args

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        c = self.cameras[idx]
        viewpoint_cam = loadCam(self.args, id, c, self.scale)
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
        
        return x, y