# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

# this script requires Open3D python binding
# please follow the intructions in setup.py before running this script.
import numpy as np
import open3d as o3d
import os
import argparse
# import torch

from config import scenes_tau_dict
from registration import (
    trajectory_alignment,
    registration_vol_ds,
    registration_unif,
    read_trajectory,
)
# from help_func import auto_orient_and_center_poses
from trajectory_io import CameraPose
from evaluation import EvaluateHisto
from util import make_dir
from plot import plot_graph


def run_evaluation(scene, dataset_dir, ply_path, transform_path, out_dir, view_crop):

    print("")
    print("===========================")
    print("Evaluating %s" % scene)
    print("===========================")

    # assume that the prediction and GT have been aligned
    trajectory_transform = np.identity(4)

    # this is for groundtruth pointcloud, we can use it
    # TODO: change file name for GT point cloud
    gt_filen = os.path.join(dataset_dir, scene + ".ply")
    # this crop file is also w.r.t the groundtruth pointcloud, we can use it. 
    # Otherwise we have to crop the estimated pointcloud by ourself
    cropfile = os.path.join(dataset_dir, scene + ".json")
    # this is not so necessary
    map_file = None

    make_dir(out_dir)

    # Load reconstruction and according GT
    print(ply_path)
    pcd = o3d.io.read_point_cloud(ply_path)
    # add center points
    import trimesh
    mesh = trimesh.load_mesh(ply_path)
    # add center points
    # sampled_vertices = mesh.vertices[mesh.faces].mean(axis=1)
    # add 4 points based on the face vertices
    # face_vertices = mesh.vertices[mesh.faces]# .mean(axis=1)
    # weights = np.array([[3, 3, 3],
    #                     [4, 4, 1],
    #                     [4, 1, 4],
    #                     [1, 4, 4]],dtype=np.float32) / 9.0
    # sampled_vertices = np.sum(face_vertices.reshape(-1, 1, 3, 3) * weights.reshape(1, 4, 3, 1), axis=2).reshape(-1, 3)
    
    # vertices = np.concatenate([mesh.vertices, sampled_vertices], axis=0)
    vertices = trimesh.sample.sample_surface_even(mesh, mesh.vertices.shape[0]+mesh.faces.shape[0])[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    ### end add center points
    
    print(gt_filen)
    gt_pcd = o3d.io.read_point_cloud(gt_filen)

    if transform_path is not None:
        with open(transform_path, 'r') as f:
            coord_transform = np.loadtxt(f)
        # Rotate pcd and gt_pcd with coord_transform
        pcd.transform(np.linalg.inv(coord_transform))
        gt_pcd.transform(np.linalg.inv(coord_transform))

    dTau = np.mean(gt_pcd.compute_nearest_neighbor_distance()) * 1.5  # around 5e-4 of scene size
    
    # big pointclouds will be downlsampled to this number to speed up alignment
    dist_threshold = dTau
    # Refine alignment by using the actual GT and MVS pointclouds
    vol = o3d.visualization.read_selection_polygon_volume(cropfile)
    
    # Registration refinment in 3 iterations
    r2 = registration_vol_ds(pcd, gt_pcd, trajectory_transform, vol, dTau,
                             dTau * 80, 20)
    r3 = registration_vol_ds(pcd, gt_pcd, r2.transformation, vol, dTau / 2.0,
                             dTau * 20, 20)
    r = registration_unif(pcd, gt_pcd, r3.transformation, vol, 2 * dTau, 20)
    trajectory_transform = r.transformation
    
    # Histogramms and P/R/F1
    plot_stretch = 5
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = EvaluateHisto(
        pcd,
        gt_pcd,
        trajectory_transform, # r.transformation,
        vol,
        dTau / 2.0,
        dTau,
        out_dir,
        plot_stretch,
        scene,
        view_crop
    )
    eva = [precision, recall, fscore]
    print("==============================")
    print("evaluation result : %s" % scene)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % eva[0])
    print("recall : %.4f" % eva[1])
    print("f-score : %.4f" % eva[2])
    print("==============================")

    # Plotting
    plot_graph(
        scene,
        fscore,
        dist_threshold,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        plot_stretch,
        out_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="path to a dataset/scene directory containing X.json, X.ply, ...",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="name of the scene, use Block_all by default",
    )
    parser.add_argument(
        "--ply-path",
        type=str,
        required=True,
        help="path to reconstruction ply file",
    )
    parser.add_argument(
        "--transform-path",
        type=str,
        default=None,
        help="path to transformation txt file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=
        "output directory, default: an evaluation directory is created in the directory of the ply file",
    )
    parser.add_argument(
        "--view-crop",
        type=int,
        default=0,
        help="whether view the crop pointcloud after aligned",
    )
    args = parser.parse_args()

    args.view_crop = False #  (args.view_crop > 0)
    if args.out_dir.strip() == "":
        args.out_dir = os.path.join(os.path.dirname(args.ply_path),
                                    "evaluation")

    run_evaluation(
        scene=args.scene,
        dataset_dir=args.dataset_dir,
        ply_path=args.ply_path,
        transform_path=args.transform_path,
        out_dir=args.out_dir,
        view_crop=args.view_crop
    )

    print(f"Evaluation for {args.ply_path} is done.")
