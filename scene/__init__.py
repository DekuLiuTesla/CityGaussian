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

import os
import tqdm
import random
import json
import yaml
import torch
import numpy as np
import imblearn
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, GroupParams
from plyfile import PlyData, PlyElement
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffcameras_extentling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


class FusedScene(Scene):
    def __init__(self, args : ModelParams, pipe : ModelParams,  gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.pretrain_path = args.pretrain_path if hasattr(args, "pretrain_path") else None
        self.balance = False
        self.loaded_iter = None
        self.class_weight = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        
        if hasattr(args, "balance"):
            self.balance = args.balance
            if self.balance == "undersample":
                self.rus = imblearn.under_sampling.RandomUnderSampler(random_state=42)
                print("Use RandomUnderSampler for class balance")
            elif self.balance == "oversample":
                self.rus = imblearn.over_sampling.RandomOverSampler(random_state=42)
                print("Use RandomOverSampler for class balance")
            else:
                assert False, "Unknown class balance method!"

        self.train_cameras = {}
        self.test_cameras = {}
        self.cameras_extent = 0.0
        train_label_fuse = []
        train_cameras_fuse = []
        test_cameras_fuse = []
        xyz = []
        rgb = []

        # scene info is constructed using that from individual blocks
        # if not loaded_iter, the initial gaussians are constructed by fusing individual gaussians
        for idx, sub_path in enumerate(args.sub_paths):
            with open(sub_path) as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
            lp, op, pp = self.parse_cfg(cfg)

            if os.path.exists(os.path.join(lp.source_path, "sparse")):
                scene_info_block = sceneLoadTypeCallbacks["Colmap"](lp.source_path, lp.images, lp.eval)
            elif os.path.exists(os.path.join(lp.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info_block = sceneLoadTypeCallbacks["Blender"](lp.source_path, lp.white_background, lp.eval)
            else:
                assert False, f"Could not recognize block scene type of {sub_path}!"
            
            train_label_fuse += [idx] * len(scene_info_block.train_cameras)
            train_cameras_fuse += scene_info_block.train_cameras
            test_cameras_fuse += scene_info_block.test_cameras
            self.cameras_extent = max(self.cameras_extent, scene_info_block.nerf_normalization["radius"])
            plydata = PlyData.read(scene_info_block.ply_path)
            vertices = plydata['vertex']
            positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
            colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
            xyz.append(positions)
            rgb.append(colors)

            if not self.loaded_iter and not self.pretrain_path:
                config_name = os.path.splitext(os.path.basename(sub_path))[0]
                lp.model_path = os.path.join(os.path.dirname(self.model_path), config_name)
                gaussian_block = GaussianModel(lp.sh_degree)
                gaussian_block.load_ply(os.path.join(lp.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(op.iterations),
                                                    "point_cloud.ply"))
                self.update_gaussians(gaussian_block)
        
        # add class balance weights to train_cameras_fuse
        train_label_fuse = np.array(train_label_fuse)
                    
        if not self.loaded_iter:
            # fuse ply data and store in appointed path
            xyz = np.concatenate(xyz, axis=0)
            rgb = np.concatenate(rgb, axis=0)
            storePly(os.path.join(self.model_path, "input.ply"), xyz, rgb)

            json_cams = []
            camlist = []
            if test_cameras_fuse:
                camlist.extend(test_cameras_fuse)
            if train_cameras_fuse:
                camlist.extend(train_cameras_fuse)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
        
        prune_data = args.prune_data if hasattr(args, "prune_data") else False
        if prune_data:
            from gaussian_renderer import render
            from utils.loss_utils import ssim

            with torch.no_grad():
                print("Pruning Training Cameras")
                loss_list = []
                bg_color = [1,1,1] if args.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                views = cameraList_from_camInfos(train_cameras_fuse, 1.0, args)

                for i in range(len(train_cameras_fuse)):
                    view = views[i]
                    render_pkg_fused = render(view, self.gaussians, pipe, background)
                    image_fused, _, _, _, _ = render_pkg_fused["render"], render_pkg_fused["viewspace_points"], render_pkg_fused["visibility_filter"], render_pkg_fused["radii"], render_pkg_fused["geometry"]
                    image_fused = image_fused
                    gt_image = view.original_image.to("cuda")

                    loss = ssim(image_fused, gt_image)
                    loss_list.append(loss)
                
                loss_list = torch.stack(loss_list, dim=0)
                loss_mean = torch.mean(loss_list, dim=0)
                loss_std = torch.std(loss_list, dim=0)

                mask = [i for i in range(loss_list.shape[0]) if loss_list[i] < loss_mean + loss_std]
                train_cameras_fuse = [train_cameras_fuse[i] for i in mask]

            print(f"Left {len(train_cameras_fuse)} Training Cameras")

        if shuffle:
            pack = list(zip(train_cameras_fuse, train_label_fuse))  
            random.shuffle(pack)  # Multi-res consistent random shuffling
            train_cameras_fuse[:], train_label_fuse[:] = zip(*pack)
            random.shuffle(test_cameras_fuse)  # Multi-res consistent random shuffling
        
        self.train_label_fuse = train_label_fuse

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(train_cameras_fuse, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(test_cameras_fuse, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                "point_cloud",
                                                "iteration_" + str(self.loaded_iter),
                                                "point_cloud.ply"))
        elif self.pretrain_path:
            self.gaussians.load_ply(os.path.join(self.pretrain_path, "point_cloud.ply"))
        else:
            self.gaussians.save_ply(os.path.join(self.model_path,
                                                "point_cloud",
                                                "iteration_0",
                                                "point_cloud.ply"))
            # reload to set all properties to be leaf nodes
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                "point_cloud",
                                                "iteration_0",
                                                "point_cloud.ply"))
    
    def parse_cfg(self, cfg):
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
    
    def update_gaussians(self, new_gaussians):
        if len(self.gaussians._xyz) == 0:
            self.gaussians._xyz = new_gaussians._xyz
            self.gaussians._features_dc = new_gaussians._features_dc
            self.gaussians._features_rest = new_gaussians._features_rest
            self.gaussians._scaling = new_gaussians._scaling
            self.gaussians._rotation = new_gaussians._rotation
            self.gaussians._opacity = new_gaussians._opacity
            self.gaussians.max_radii2D = new_gaussians.max_radii2D
        else:
            self.gaussians._xyz = torch.cat([self.gaussians._xyz, new_gaussians._xyz], dim=0)
            self.gaussians._features_dc = torch.cat([self.gaussians._features_dc, new_gaussians._features_dc], dim=0)
            self.gaussians._features_rest = torch.cat([self.gaussians._features_rest, new_gaussians._features_rest], dim=0)
            self.gaussians._scaling = torch.cat([self.gaussians._scaling, new_gaussians._scaling], dim=0)
            self.gaussians._rotation = torch.cat([self.gaussians._rotation, new_gaussians._rotation], dim=0)
            self.gaussians._opacity = torch.cat([self.gaussians._opacity, new_gaussians._opacity], dim=0)
            self.gaussians.max_radii2D = torch.cat([self.gaussians.max_radii2D, new_gaussians.max_radii2D], dim=0)

    def getTrainCameras(self, scale=1.0):
        if self.balance:
            train_idx = np.arange(len(self.train_label_fuse))[:, None]
            train_label = self.train_label_fuse.copy()
            train_idx, train_label = self.rus.fit_resample(train_idx, train_label)
            sample_mask = train_idx[:, 0].tolist()
            return [self.train_cameras[scale][idx] for idx in sample_mask]
        else:
            return self.train_cameras[scale]
    
class RefinedScene(Scene):

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.pretrain_path = args.pretrain_path if hasattr(args, "pretrain_path") else None
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffcameras_extentling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        elif self.pretrain_path:
            self.gaussians.load_ply(os.path.join(self.pretrain_path, "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

class LargeScene(Scene):
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.pretrain_path = args.pretrain_path if hasattr(args, "pretrain_path") else None

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffcameras_extentling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.train_cameras = scene_info.train_cameras
        self.test_cameras = scene_info.test_cameras

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                "point_cloud",
                                                "iteration_" + str(self.loaded_iter),
                                                "point_cloud.ply"))
        elif self.pretrain_path:
            self.gaussians.load_ply(os.path.join(self.pretrain_path, "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
    
    def save(self, iteration, args=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        if hasattr(args, 'block_id') and args.block_id >= 0:
            xyz_org = self.gaussians.get_xyz
            if len(args.aabb) == 4:
                aabb = [args.aabb[0], args.aabb[1], xyz_org[:, -1].min(), 
                        args.aabb[2], args.aabb[3], xyz_org[:, -1].max()]
            elif len(args.aabb) == 6:
                aabb = args.aabb
            else:
                assert False, "Unknown aabb format!"
            aabb = torch.tensor(aabb, dtype=torch.float32, device=xyz_org.device)
            xyz_contracted = self.contract_to_unisphere(xyz_org, aabb, ord=torch.inf)
            block_id_z = args.block_id // (args.block_dim[0] * args.block_dim[1])
            block_id_y = (args.block_id % (args.block_dim[0] * args.block_dim[1])) // args.block_dim[1]
            block_id_x = (args.block_id % (args.block_dim[0] * args.block_dim[1])) % args.block_dim[1]

            min_x, max_x = float(block_id_x) / args.block_dim[0], float(block_id_x + 1) / args.block_dim[0]
            min_y, max_y = float(block_id_y) / args.block_dim[1], float(block_id_y + 1) / args.block_dim[1]
            min_z, max_z = float(block_id_z) / args.block_dim[2], float(block_id_z + 1) / args.block_dim[2]

            block_mask = (xyz_contracted[:, 0] >= min_x) & (xyz_contracted[:, 0] < max_x)  \
                        & (xyz_contracted[:, 1] >= min_y) & (xyz_contracted[:, 1] < max_y) \
                        & (xyz_contracted[:, 2] >= min_z) & (xyz_contracted[:, 2] < max_z)
            
            sh_degree = self.gaussians.max_sh_degree
            masked_gaussians = GaussianModel(sh_degree)
            masked_gaussians._xyz = self.gaussians._xyz[block_mask]
            masked_gaussians._scaling = self.gaussians._scaling[block_mask]
            masked_gaussians._rotation = self.gaussians._rotation[block_mask]
            masked_gaussians._features_dc = self.gaussians._features_dc[block_mask]
            masked_gaussians._features_rest = self.gaussians._features_rest[block_mask]
            masked_gaussians._opacity = self.gaussians._opacity[block_mask]
            masked_gaussians.max_radii2D = self.gaussians.max_radii2D[block_mask]

            block_point_cloud_path = os.path.join(self.model_path, "point_cloud/blocks/{}/iteration_{}".format(args.block_id, iteration))
            masked_gaussians.save_ply(os.path.join(block_point_cloud_path, "point_cloud.ply"))
    
    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras

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
