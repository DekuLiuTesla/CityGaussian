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
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import BasicPointCloud
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly, SceneInfo
from scene.gaussian_model import GaussianModel, GaussianModelLOD, GatheredGaussian
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
    
class LargeScene(Scene):
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, load_vq=False, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_vq = load_vq
        self.gaussians = gaussians
        self.pretrain_path = args.pretrain_path

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if args.block_id >= 0:
            partition = np.load(os.path.join(args.source_path, "data_partitions", f"{args.partition_name}.npy"))[:, args.block_id]
            if args.aabb is None:
                args.aabb = np.load(os.path.join(args.source_path, "data_partitions", f"{args.partition_name}_aabb.npy")).tolist()
            print(f"Using Partition File {args.partition_name}.npy")
        else:
            partition = None

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold, partition=partition)
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

        if self.load_vq:
            self.gaussians.load_vq(self.model_path)
        elif self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                "point_cloud",
                                                "iteration_" + str(self.loaded_iter),
                                                "point_cloud.ply"))
        elif self.pretrain_path:
            self.gaussians.load_ply(os.path.join(self.pretrain_path, "point_cloud.ply"))
            self.gaussians.spatial_lr_scale = self.cameras_extent
        else:
            if args.add_background_sphere:
                import math
                scene_center = -scene_info.nerf_normalization['translate']
                scene_radius = scene_info.nerf_normalization['radius']
                # build unit sphere points
                n_points = args.background_sphere_points
                samples = np.arange(n_points)
                y = 1 - (samples / float(n_points - 1)) * 2  # y goes from 1 to -1
                radius = np.sqrt(1 - y * y)  # radius at y
                phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians
                theta = phi * samples  # golden angle increment
                x = np.cos(theta) * radius
                z = np.sin(theta) * radius
                unit_sphere_points = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1)
                # build background sphere
                background_sphere_point_xyz = (unit_sphere_points * scene_radius * args.background_sphere_radius) + scene_center
                background_sphere_point_rgb = np.asarray(np.random.random(background_sphere_point_xyz.shape), dtype=np.float64)
                # add background sphere to scene
                scene_info = SceneInfo(
                    point_cloud=BasicPointCloud(
                                points=np.concatenate([scene_info.point_cloud.points, background_sphere_point_xyz], axis=0),
                                colors=np.concatenate([scene_info.point_cloud.colors, background_sphere_point_rgb], axis=0),
                                normals=np.zeros_like(background_sphere_point_xyz)),
                    train_cameras=scene_info.train_cameras,
                    test_cameras=scene_info.test_cameras,
                    nerf_normalization=scene_info.nerf_normalization,
                    ply_path=scene_info.ply_path)
                # increase prune extent
                # TODO: resize scene_extent without changing lr
                self.cameras_extent = scene_radius * args.background_sphere_radius * 1.0001

                print("added {} background sphere points, rescale prune extent from {} to {}".format(n_points, scene_radius, self.cameras_extent))

            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
    
    def save(self, iteration, args=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))

        if args.block_id >= 0:
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
            block_id_y = (args.block_id % (args.block_dim[0] * args.block_dim[1])) // args.block_dim[0]
            block_id_x = (args.block_id % (args.block_dim[0] * args.block_dim[1])) % args.block_dim[0]

            min_x, max_x = float(block_id_x) / args.block_dim[0], float(block_id_x + 1) / args.block_dim[0]
            min_y, max_y = float(block_id_y) / args.block_dim[1], float(block_id_y + 1) / args.block_dim[1]
            min_z, max_z = float(block_id_z) / args.block_dim[2], float(block_id_z + 1) / args.block_dim[2]

            block_mask = (xyz_contracted[:, 0] >= min_x) & (xyz_contracted[:, 0] < max_x)  \
                        & (xyz_contracted[:, 1] >= min_y) & (xyz_contracted[:, 1] < max_y) \
                        & (xyz_contracted[:, 2] >= min_z) & (xyz_contracted[:, 2] < max_z)
            
            sh_degree = self.gaussians.max_sh_degree
            masked_gaussians = GaussianModel(sh_degree)
            masked_gaussians._xyz = self.gaussians.get_xyz[block_mask]
            masked_gaussians._scaling = self.gaussians._scaling[block_mask]
            masked_gaussians._rotation = self.gaussians._rotation[block_mask]
            masked_gaussians._features_dc = self.gaussians._features_dc[block_mask]
            masked_gaussians._features_rest = self.gaussians._features_rest[block_mask]
            masked_gaussians._opacity = self.gaussians._opacity[block_mask]
            masked_gaussians.max_radii2D = self.gaussians.max_radii2D[block_mask]

            block_point_cloud_path = os.path.join(self.model_path, "point_cloud_blocks/scale_1.0/iteration_{}".format(iteration))
            masked_gaussians.save_ply(os.path.join(block_point_cloud_path, "point_cloud.ply"))

            if args.save_block_only:
                return
            
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    
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
