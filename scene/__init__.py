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
import random
import json
import yaml
import torch
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from scene.gaussian_model import GaussianModel, GaussianModelGrad
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
        self.cameras_extent = 0.0
        train_cameras_fuse = []
        test_cameras_fuse = []
        xyz = []
        rgb = []

        for sub_path in args.sub_paths:
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
            
            train_cameras_fuse += scene_info_block.train_cameras
            test_cameras_fuse += scene_info_block.test_cameras
            self.cameras_extent = max(self.cameras_extent, scene_info_block.nerf_normalization["radius"])
            plydata = PlyData.read(scene_info_block.ply_path)
            vertices = plydata['vertex']
            positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
            colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
            xyz.append(positions)
            rgb.append(colors)

            if not self.loaded_iter:
                config_name = os.path.splitext(os.path.basename(sub_path))[0]
                lp.model_path = os.path.join(os.path.dirname(self.model_path), config_name)
                gaussian_block = GaussianModel(lp.sh_degree)
                gaussian_block.load_ply(os.path.join(lp.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(op.iterations),
                                                    "point_cloud.ply"))
                self.update_gaussians(gaussian_block)
                    
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

        if shuffle:
            random.shuffle(train_cameras_fuse)  # Multi-res consistent random shuffling
            random.shuffle(test_cameras_fuse)  # Multi-res consistent random shuffling

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
