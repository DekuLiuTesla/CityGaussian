import os
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.camera_utils import loadCam

class GSDataset(Dataset):
    def __init__(self, cameras, scene, args, pipe=None, scale=1):
        self.pre_load = pipe.pre_load if hasattr(pipe, 'pre_load') else False
        self.cameras = cameras
        self.scale = scale
        self.args = args
        
        if hasattr(args, 'block_id') and args.block_id >= 0:
            self.block_filtering(scene.gaussians, args, pipe)
            print(f"Filtered Cameras: {len(self.cameras)}")
        
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

        if self.blur_level > 0:
            y = F.gaussian_blur(y, self.blur_level * 30 +1)
        
        return x, y
    
    def contract_to_unisphere(self,
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
    
    def block_filtering(self, gaussians, args, pp):
        xyz_org = gaussians.get_xyz
        if len(args.aabb) == 4:
            aabb = [args.aabb[0], args.aabb[1], xyz_org[:, -1].min(), 
                    args.aabb[2], args.aabb[3], xyz_org[:, -1].max()]
        elif len(args.aabb) == 6:
            aabb = args.aabb
        else:
            assert False, "Unknown aabb format!"
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=xyz_org.device)
        block_id_z = args.block_id // (args.block_dim[0] * args.block_dim[1])
        block_id_y = (args.block_id % (args.block_dim[0] * args.block_dim[1])) // args.block_dim[0]
        block_id_x = (args.block_id % (args.block_dim[0] * args.block_dim[1])) % args.block_dim[0]

        if hasattr(args, 'xyz_limited') and args.xyz_limited:
            xyz = xyz_org
            min_x = self.aabb[0] + (self.aabb[3] - self.aabb[0]) * float(block_id_x) / args.block_dim[0]
            max_x = self.aabb[0] + (self.aabb[3] - self.aabb[0]) * float(block_id_x + 1) / args.block_dim[0]
            min_y = self.aabb[1] + (self.aabb[4] - self.aabb[1]) * float(block_id_y) / args.block_dim[1]
            max_y = self.aabb[1] + (self.aabb[4] - self.aabb[1]) * float(block_id_y + 1) / args.block_dim[1]
            min_z = self.aabb[2] + (self.aabb[5] - self.aabb[2]) * float(block_id_z) / args.block_dim[2]
            max_z = self.aabb[2] + (self.aabb[5] - self.aabb[2]) * float(block_id_z + 1) / args.block_dim[2]
        else:
            xyz = self.contract_to_unisphere(xyz_org, self.aabb, ord=torch.inf)
            min_x, max_x = float(block_id_x) / args.block_dim[0], float(block_id_x + 1) / args.block_dim[0]
            min_y, max_y = float(block_id_y) / args.block_dim[1], float(block_id_y + 1) / args.block_dim[1]
            min_z, max_z = float(block_id_z) / args.block_dim[2], float(block_id_z + 1) / args.block_dim[2]

        block_num = args.block_dim[0] * args.block_dim[1] * args.block_dim[2]
        block_mask = (xyz[:, 0] >= min_x) & (xyz[:, 0] < max_x)  \
                    & (xyz[:, 1] >= min_y) & (xyz[:, 1] < max_y) \
                    & (xyz[:, 2] >= min_z) & (xyz[:, 2] < max_z)
        
        sh_degree = gaussians.max_sh_degree
        masked_gaussians = GaussianModel(sh_degree)
        masked_gaussians._xyz = xyz_org[block_mask]
        masked_gaussians._scaling = gaussians._scaling[block_mask]
        masked_gaussians._rotation = gaussians._rotation[block_mask]
        masked_gaussians._features_dc = gaussians._features_dc[block_mask]
        masked_gaussians._features_rest = gaussians._features_rest[block_mask]
        masked_gaussians._opacity = gaussians._opacity[block_mask]
        masked_gaussians.max_radii2D = gaussians.max_radii2D[block_mask]

        filtered_cameras = []
        print(f"Getting Data of Block {args.block_id} / {block_num}")
        with torch.no_grad():
            for idx in tqdm(range(len(self.cameras))):
                bg_color = [1,1,1] if args.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device=xyz_org.device)
                c = self.cameras[idx]
                viewpoint_cam = loadCam(self.args, id, c, self.scale)
                render_pkg_block = render(viewpoint_cam, masked_gaussians, pp, background)
                visibility_filter = render_pkg_block["visibility_filter"]
                total_opacity = render_pkg_block["geometry"][visibility_filter, 6].sum()
        
                if total_opacity > args.opacity_threshold:
                    filtered_cameras.append(c)
        self.cameras = filtered_cameras