import os
import sys
import add_pypath

import numpy as np
import open3d as o3d
from argparse import ArgumentParser

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--file_dir', '-f', type=str, help='path to target point cloud', required=True)
    parser.add_argument('--vox_size', '-v', type=float, help='downsampling voxel size', default=0.008)
    parser.add_argument('--scaling_factor', '-s', type=float, help='position scaling factor', default=1.0)
    args = parser.parse_args(sys.argv[1:])
    
    pcd = o3d.io.read_point_cloud(args.file_dir)
    print(f"{args.file_dir} has {len(pcd.points)} points")

    ds_pcd = pcd.voxel_down_sample(voxel_size=args.vox_size).scale(args.scaling_factor, center=(0, 0, 0))
    print("Downsampled has", len(ds_pcd.points), "points")
    print("Average distance of downsampled point cloud: ", np.mean(ds_pcd.compute_nearest_neighbor_distance()))

    save_dir = args.file_dir.replace(".ply", "_ds.ply")
    print("Downsampled point cloud is saved to ", save_dir)
    o3d.io.write_point_cloud(save_dir, ds_pcd)