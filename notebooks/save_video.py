# %%
import os
import sys
sys.path.append('/home/yang_liu/python_workspace/3DGS')

import yaml
import torch
import torch_scatter
import torchvision
import wandb
import time
import inspect
import imageio
import numpy as np
import open3d as o3d
import pynvml
import matplotlib.pyplot as plt
from tqdm import tqdm
from arguments import GroupParams
from scene import LargeScene
from os import makedirs
from gaussian_renderer import render_lod, render
from utils.general_utils import safe_state
from utils.large_utils import which_block, block_filtering
from utils.sh_utils import SH2RGB
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from torch.utils.data import DataLoader
from utils.camera_utils import loadCamV2
from transforms3d.euler import mat2euler, euler2mat
from transforms3d.quaternions import mat2quat, quat2mat
from utils.general_utils import PILtoTorch
from scene.cameras import Camera
from scipy.spatial import ConvexHull
from matplotlib.backends.backend_agg import FigureCanvasAgg

WARNED = False

# %%
class BlockedGaussian:

    gaussians : GaussianModel

    def __init__(self, gaussians, lp, range=[0, 1], scale=1.0, compute_cov3D_python=False):
        self.cell_corners = []
        self.xyz = None
        self.feats = None
        self.max_sh_degree = lp.sh_degree
        self.device = gaussians.get_xyz.device
        self.compute_cov3D_python = compute_cov3D_python
        self.cell_ids = torch.zeros(gaussians.get_opacity.shape[0], dtype=torch.long, device=self.device)
        self.mask = torch.zeros(gaussians.get_opacity.shape[0], dtype=torch.bool, device=self.device)

        self.block_dim = lp.block_dim
        self.num_cell = lp.block_dim[0] * lp.block_dim[1] * lp.block_dim[2]
        self.aabb = lp.aabb
        self.scale = scale
        self.range = range

        self.cell_divider(gaussians)
        self.cell_corners = torch.stack(self.cell_corners, dim=0)

    def cell_divider(self, gaussians, n=4):
        with torch.no_grad():
            if self.compute_cov3D_python:
                geometry = gaussians.get_covariance(self.scale).to(self.device)
            else:
                geometry = torch.cat([gaussians.get_scaling,
                                      gaussians.get_rotation], dim=1)
            self.xyz = gaussians.get_xyz
            self.feats = torch.cat([gaussians.get_opacity,  
                                    gaussians.get_features.reshape(geometry.shape[0], -1),
                                    geometry], dim=1).half()
            
            for cell_idx in range(self.num_cell):
                cell_mask = block_filtering(cell_idx, self.xyz, self.aabb, self.block_dim, self.scale)
                self.cell_ids[cell_mask] = cell_idx
                # MAD to eliminate influence of outsiders
                xyz_median = torch.median(self.xyz[cell_mask], dim=0)[0]
                delta_median = torch.median(torch.abs(self.xyz[cell_mask] - xyz_median), dim=0)[0]
                xyz_min = xyz_median - n * delta_median
                xyz_min = torch.max(xyz_min, torch.min(self.xyz[cell_mask], dim=0)[0])
                xyz_max = xyz_median + n * delta_median
                xyz_max = torch.min(xyz_max, torch.max(self.xyz[cell_mask], dim=0)[0])
                corners = torch.tensor([[xyz_min[0], xyz_min[1], xyz_min[2]],
                                       [xyz_min[0], xyz_min[1], xyz_max[2]],
                                       [xyz_min[0], xyz_max[1], xyz_min[2]],
                                       [xyz_min[0], xyz_max[1], xyz_max[2]],
                                       [xyz_max[0], xyz_min[1], xyz_min[2]],
                                       [xyz_max[0], xyz_min[1], xyz_max[2]],
                                       [xyz_max[0], xyz_max[1], xyz_min[2]],
                                       [xyz_max[0], xyz_max[1], xyz_max[2]]], device=self.xyz.device)
                self.cell_corners.append(corners)
    
    def get_feats(self, indices, distances):
        out_xyz = torch.tensor([], device=self.device, dtype=self.xyz.dtype)
        out_feats = torch.tensor([], device=self.device, dtype=self.feats.dtype)
        block_mask = (distances >= self.range[0]) & (distances < self.range[1])
        if block_mask.sum() > 0:
            self.mask = torch.isin(self.cell_ids, indices[block_mask].to(self.device))
            out_xyz = self.xyz[self.mask]
            out_feats = self.feats[self.mask]
        return out_xyz, out_feats

    def get_feats_ptwise(self, viewpoint_cam):
        out_xyz = torch.tensor([], device=self.device, dtype=self.xyz.dtype)
        out_feats = torch.tensor([], device=self.device, dtype=self.feats.dtype)

        homo_xyz = torch.cat([self.xyz, torch.ones_like(self.xyz[..., [0]])], dim=-1)
        cam_center = viewpoint_cam.camera_center
        viewmatrix = viewpoint_cam.world_view_transform
        xyz_cam = homo_xyz @ viewmatrix
        self.mask = (xyz_cam[..., 2] > 0.2)
        if self.mask.sum() == 0:
            return out_xyz, out_feats

        distances = torch.norm(self.xyz - cam_center[None, :3], dim=-1)
        self.mask &= (distances >= self.range[0]) & (distances < self.range[1])
        if self.mask.sum() > 0:
            out_xyz = self.xyz[self.mask]
            out_feats = self.feats[self.mask]
        return out_xyz, out_feats

def load_gaussians(cfg, config_name, iteration=30_000, load_vq=False, device='cuda', source_path='data/matrix_city/aerial/test/block_all_test'):
    
    lp, op, pp = parse_cfg(cfg)
    setattr(lp, 'config_path', cfg)
    lp.source_path = source_path
    lp.model_path = os.path.join("../output/", config_name)

    modules = __import__('scene')
    
    with torch.no_grad():
        if 'apply_voxelize' in lp.model_config['kwargs'].keys():
            lp.model_config['kwargs']['apply_voxelize'] = False
        gaussians = getattr(modules, lp.model_config['name'])(lp.sh_degree, device=device, **lp.model_config['kwargs'])
        scene = LargeScene(lp, gaussians, load_iteration=iteration, load_vq=load_vq, shuffle=False)
        print(f'Init {config_name} with {len(gaussians.get_opacity)} points\n')

    return gaussians, scene

def parse_cfg(cfg):
    lp = GroupParams()
    op = GroupParams()
    pp = GroupParams()

    for arg in cfg['model_params'].items():
        setattr(lp, arg[0], arg[1])
    
    for arg in cfg['optim_params'].items():
        setattr(op, arg[0], arg[1]) 

    for arg in cfg['pipeline_params'].items():
        setattr(pp, arg[0], arg[1])
    
    return lp, op, pp

def loadCamV3(args, id, cam_info, resolution_scale, xyz=None, z_dim=None, yaw=None):
    # use appointed pitch and height
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if yaw is not None and z_dim is not None and xyz is not None:
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = cam_info.R.transpose()
        Rt[:3, 3] = cam_info.T
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        euler = np.array(mat2euler(C2W[:3, :3]))
        euler[z_dim] = yaw
        C2W[:3, :3] = euler2mat(*euler)
        C2W[:3, 3] = xyz
        Rt = np.linalg.inv(C2W)

        R = Rt[:3, :3].transpose()
        T = Rt[:3, 3]
    else:
        R = cam_info.R
        T = cam_info.T

    return Camera(colmap_id=cam_info.uid, R=R, T=T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def loadCamV4(args, id, cam_info, resolution_scale, xyz=None, angle=None):
    # use appointed pitch and height
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if angle is not None and xyz is not None:
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = cam_info.R.transpose()
        Rt[:3, 3] = cam_info.T
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        euler = np.array(mat2euler(C2W[:3, :3]))
        euler = angle
        C2W[:3, :3] = euler2mat(*euler)
        C2W[:3, 3] = xyz
        Rt = np.linalg.inv(C2W)

        R = Rt[:3, :3].transpose()
        T = Rt[:3, 3]
    else:
        R = cam_info.R
        T = cam_info.T

    return Camera(colmap_id=cam_info.uid, R=R, T=T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

# find the a & b points
def get_bezier_coef(points):
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B

# returns the general Bezier cubic formula given 4 control points
def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

# return one cubic curve for each consecutive points
def get_bezier_cubic(points):
    A, B = get_bezier_coef(points)
    return [
        get_cubic(points[i], A[i], B[i], points[i + 1])
        for i in range(len(points) - 1)
    ]

# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(points, n):
    curves = get_bezier_cubic(points)
    seg_length = [np.linalg.norm(points[i] - points[i + 1]) for i in range(len(points) - 1)]
    mean_length = np.mean(np.array(seg_length))
    return np.array([fun(t) for length, fun in zip(seg_length, curves) for t in np.linspace(0, 1, max(int(n * length / mean_length), 1))])

def evaluate_bezier_v2(points, n):
    curves = get_bezier_cubic(points)
    seg_length = [np.linalg.norm(points[i] - points[i + 1]) for i in range(len(points) - 1)]
    mean_length = np.mean(np.array(seg_length))
    seg_pts_num = np.array([max(int(n * length / mean_length), 1) for length in seg_length])
    acc_pts_num = np.cumsum(seg_pts_num)
    pts = np.array([fun(t) for length, fun in zip(seg_length, curves) for t in np.linspace(0, 1, max(int(n * length / mean_length), 1))])
    return pts, acc_pts_num

def evaluate_bezier_v3(points, n):
    curves = get_bezier_cubic(points)
    seg_samples = [n]
    speed_start = None
    speed_end_pre = curves[0](1) - curves[0](1-1/n)
    for idx in range(1, len(curves)):
        speed_start = curves[idx](1/n) - curves[idx](0)
        n = int(n * np.linalg.norm(speed_start) / np.linalg.norm(speed_end_pre))
        seg_samples.append(n)
        speed_end_pre = curves[idx](1) - curves[idx](1-1/n)
        
    acc_pts_num = np.cumsum(np.array(seg_samples))
    pts = np.array([fun(t) for samples, fun in zip(seg_samples, curves) for t in np.linspace(0, 1, max(samples, 1))])
    return pts, acc_pts_num

def np_move_avg(a,n,mode="same"):
    b = a.copy()
    b[n//2:-n//2] = np.convolve(a, np.ones((n,))/n, mode=mode)[n//2:-n//2]
    return b

def np_move_avg_v2(a,n,mode="same"):
    b = a.copy()
    if len(b.shape) == 1:
        b[n//2:-n//2] = np.convolve(a, np.ones((n,))/n, mode=mode)[n//2:-n//2]
    else:
        for i in range(b.shape[1]):
            b[n//2:-n//2, i] = np.convolve(a[:, i], np.ones((n,))/n, mode=mode)[n//2:-n//2]
    return b

# %% [markdown]
# ## 1.2 No LoD, MatrixCity

# %%
load_vq = False
iteration = 30_000
custom_test = '../data/matrix_city/aerial/test/block_all_test'
if load_vq:
    iteration = None

config = "../config/block_mc_aerial_block_all_lr_c36_loss_8_num_10w_lr2.yaml"
model_path = os.path.join('../output', os.path.basename(config).split('.')[0])
with open(config) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    lp, op, pp = parse_cfg(cfg)
    setattr(lp, 'config_path', config)
    if lp.model_path == '':
        lp.model_path = model_path

with torch.no_grad():
    modules = __import__('scene')
    model_config = lp.model_config
    gaussians = getattr(modules, model_config['name'])(lp.sh_degree, **model_config['kwargs'])

    if custom_test:
        lp.source_path = custom_test
        filename = os.path.basename(lp.source_path)
    scene = LargeScene(lp, gaussians, load_iteration=iteration, load_vq=load_vq, shuffle=False)

    bg_color = [1,1,1] if lp.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if custom_test:
        views = scene.getTrainCameras() + scene.getTestCameras()

org_scaling = gaussians._scaling.clone()
org_opacity = gaussians._opacity.clone()

# %%
cam_center_list = []
dim = 1
for idx in range(len(views)):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = views[idx].R.transpose()
    Rt[:3, 3] = views[idx].T
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center_list.append(C2W[:3, -1])

# %% [markdown]
# ### 1.2.1 Get Convex Hull of Training / Test Poses

# %%
plt.style.use('seaborn-white')
plt.figure(figsize=(6, 6))
plt.axis('equal')
for idx in range(len(views)):
    color = 'r'
    cam_center = cam_center_list[idx]
    plt.scatter(cam_center[0], cam_center[dim], s=100, c=color, marker='x')
    # plt.text(cam_center[0], cam_center[1], f'{idx}', fontsize=12)

cam_center_array = np.array(cam_center_list)
hull = ConvexHull(np.array(cam_center_array)[:, [0, dim]], 'Qg')
hull_list=hull.vertices.tolist()

simplified_hull = []
dist = 3
for idx in range(len(hull_list)):
    dist_1 = np.linalg.norm(cam_center_array[hull_list[idx], [0, dim]] - cam_center_array[hull_list[idx-1], [0, dim]])
    if dist_1 > dist:
        simplified_hull.append(hull_list[idx])
        
simplified_hull = simplified_hull[2:] + simplified_hull[:2]
simplified_hull.append(simplified_hull[0])

plt.plot(cam_center_array[simplified_hull,0], cam_center_array[simplified_hull, dim], 'b--^',lw=2)

# %% [markdown]
# ### 1.2.2 Plot Distribution

# %%

with torch.no_grad():
    voxel_size = torch.tensor([0.15, 0.15])
    xy_range = torch.tensor([-9, -7, 7.5, 6])
    xyz = gaussians.get_xyz.cpu()
    voxel_index = torch.div(torch.tensor(xyz[:, :2]).float() - xy_range[None, :2], voxel_size[None, :], rounding_mode='floor')
    voxel_coords = voxel_index * voxel_size[None, :] + xy_range[None, :2] + voxel_size[None, :] / 2
    voxel_dim = torch.tensor([int((xy_range[2] - xy_range[0]) / voxel_size[0]), int((xy_range[3] - xy_range[1]) / voxel_size[1])])
    print(f"Dimension of Voxels: {voxel_dim}")

    new_coors, unq_inv, unq_cnt = torch.unique(voxel_coords, return_inverse=True, return_counts=True, dim=0)
    feat_mean = torch_scatter.scatter(xyz[:, -1], unq_inv, dim=0, reduce='mean')
    feat_std = torch_scatter.scatter_std(xyz[:, -1], unq_inv, dim=0)

# %%
with torch.no_grad():
    feat = feat_mean + 3 * feat_std
    feat_inv = feat[unq_inv]
    mask = xyz[:, -1] > feat_inv

# %% [markdown]
# ### 1.2.3 Fly Around Certain Point

# %%
with torch.no_grad():
    gaussians._scaling = org_scaling.clone()
    gaussians._opacity = org_opacity.clone()
    gaussians._scaling[mask] = -1e6
    gaussians._opacity[mask] = -1e6

# %%
# find corner poses
region_center1 = np.array([3.2, 2.8, 2.5])
region_center2 = np.array([1.4, -2.75, 5.0])
radius = 2

poses = [views[i] for i in simplified_hull]

video_path = os.path.join(lp.model_path, filename, "ours_lod_video")
makedirs(video_path, exist_ok=True)

# second parameter appoints number of points interpolated between each consecutive points to draw the curve
xyz_pre = None

fontsize = 20
fig = plt.Figure(figsize=(12, 9), dpi=100)
fig.set_tight_layout(True)
canvas = FigureCanvasAgg(fig)
ax = fig.gca()
gs_xyz = gaussians.get_xyz.cpu().detach().numpy()
ax.scatter(gs_xyz[::200, 0], gs_xyz[::200, dim], c='dimgrey', s=0.5)
# ax.plot(cam_center_array[simplified_hull,0], cam_center_array[simplified_hull, dim], 'r--^',lw=2)
ax.axis('equal')
ax.set_xlabel('x/100m', fontsize=fontsize)
ax.set_ylabel('y/100m', fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.set_xlim(np.min(cam_center_array[:, 0]), np.max(cam_center_array[:, 0]))
ax.set_ylim(np.min(cam_center_array[:, dim]), np.max(cam_center_array[:, dim]))

pose_per_segment = 150
pose_centers = np.array([
    [3.2, 5.8, 3.5],
    [1.2, 3.8, 2.5],
    [3.2, 1.8, 1.8],
    [5.2, 3.8, 2.5],
    [3.2, 5.8, 3.5],
    [-3.0, 4.1, 5.0],
    [-7.37, -0.46, 5.0],
    [-3.16, -4.7,  6.0],
    [3.5, -5.74, 5.0],
    [7, -2, 5.0],
    [5.0, 1.0, 3.0],
    [2.0, 0.5, 2.0],
    [-3.0, 2.9, 2.0],
    [-5.5, 1.2, 2.0],
    [-4.0, -1, 3.5],
    [1.0, 0.0, 6],
    [3.5, -1.8, 5.5],
    [1.0, -3.3, 4.5],
    [0.5, 0, 6],
    [0.5, 0, 10],
    [0.5, 0, 15],
])

interp_xyz, acc_pts = evaluate_bezier_v3(pose_centers, pose_per_segment)

Rt = np.zeros((4, 4))
Rt[:3, :3] = poses[0].R.transpose()
Rt[:3, 3] = poses[0].T
Rt[3, 3] = 1.0

C2W = np.linalg.inv(Rt)
ref_angle = np.array(mat2euler(C2W[:3, :3]))

key_point1, interval1 = acc_pts[3], 10
key_point2, interval2 = acc_pts[9], 50
key_point3, interval3 = acc_pts[14], 50
key_point4, interval4 = acc_pts[17], 50
# key_point2, interval2 = acc_pts[2], 50
# key_point3, interval3 = acc_pts[7], 50
# key_point4, interval4 = acc_pts[10], 50
xyz_pre = None


xyz_list, angle_list = [], []

for t in range(len(interp_xyz)):
    xyz = interp_xyz[t]
    angle = ref_angle.copy()

    if xyz_pre is not None and not np.any(xyz - xyz_pre):
        continue

    if t < key_point1 - 10:
        # stage 1
        delta = region_center1 - xyz
        angle[2] = np.arctan2(delta[0], -delta[dim]) + np.pi
        
    elif t < key_point1:
        # transition
        delta = region_center1 - xyz
        angle2 = np.arctan2(delta[0], -delta[dim]) + np.pi

        delta = xyz - xyz_pre
        angle1 = np.arctan2(delta[dim], delta[0])

        angle[2] = (1 - (t - key_point1 + interval1 + 1) / interval1) * angle2 + (t - key_point1 + interval1 + 1) / interval1 * angle1

        # print(f"circle: {angle2}, org: {angle1}, current: {angle}")
    
    elif t < key_point2 - interval2:
        # stage 2

        if xyz_pre is None:
            delta = interp_xyz[t + 1] - interp_xyz[t]
        else:
            delta = xyz - xyz_pre
        
        # delta = xyz - xyz_pre
        angle[2] = np.arctan2(delta[dim], delta[0])

    elif t < key_point2:
        delta = xyz - xyz_pre
        angle2 = np.arctan2(delta[dim], delta[0])
        angle1 = np.arctan2(-delta[0], delta[dim])
        
        angle[2] = (1 - (t - key_point2 + interval2 + 1) / interval2) * angle2 + (t - key_point2 + interval2 + 1) / interval2 * angle1

    elif t < key_point3 - interval3:
        # stage 2
        delta = xyz - xyz_pre
        angle[2] = np.arctan2(-delta[0], delta[dim])

    elif t < key_point3:
        # transition
        
        delta = xyz - xyz_pre
        angle2 = np.arctan2(-delta[0], delta[dim])

        delta = region_center2 - xyz
        angle1 = np.arctan2(delta[0], -delta[dim]) - np.pi

        angle[2] = (1 - (t - key_point3 + interval3 + 1) / interval3) * angle2 + (t - key_point3 + interval3 + 1) / interval3 * angle1
    
    elif t < key_point4 - interval4:
        # stage 3
        delta = region_center2 - xyz
        angle[2] = np.arctan2(delta[0], -delta[dim]) - np.pi
    
    elif t < key_point4:
        # transition
        delta = region_center2 - xyz
        angle[2] = np.arctan2(delta[0], -delta[dim]) - np.pi

        angle[0] = (1 - (t - key_point4 + interval4 + 1) / interval4) * angle[0] + (t - key_point4 + interval4 + 1) / interval4 * (-np.pi)
    
    else:
        # stage 3
        delta = region_center2 - xyz
        angle[2] = np.arctan2(delta[0], -delta[dim]) - np.pi
        angle[0] = -np.pi

    xyz_list.append(xyz)
    angle_list.append(angle)

    xyz_pre = xyz

angle_list = np_move_avg_v2(np.unwrap(np.array(angle_list), axis=0), 10, mode='same').tolist()

frames = []
for t in tqdm(range(len(xyz_list))):
    xyz = xyz_list[t]
    angle = angle_list[t]

    ax.scatter(xyz[0], xyz[dim], s=100, c='r', marker='x', label='poses')

    if t == 0:
        ax.legend(fontsize=fontsize)
    
    # canvas.draw()
    # buf = canvas.buffer_rgba()
    # bev_map = np.asarray(buf)[..., :3]

    # for matrix city, z_dim=2, otherwise z_dim=1
    viewpoint_cam = loadCamV4(lp, idx, poses[0], 1.0, xyz=xyz, angle=angle)
    img = render(viewpoint_cam, gaussians, pp, background)["render"]
    img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    # img = np.concatenate([img, bev_map], axis=1)
    
    frames.append(img) 

# canvas.draw()
# buf = canvas.buffer_rgba()
# bev_map = np.asarray(buf)[..., :3]
# plt.imshow(bev_map)
# plt.axis('off')


# %%
video = imageio.get_writer(os.path.join(video_path, "video.mp4"), mode="I", fps=30, codec="libx264", bitrate="16M", quality=10)
for frame in frames:
    video.append_data(frame)
video.close()
print(f'Video saved to {video_path}')


