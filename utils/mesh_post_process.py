import add_pypath
import os
import argparse
import open3d as o3d
from internal.utils.gs2d_mesh_utils import post_process_mesh

parser = argparse.ArgumentParser()
parser.add_argument("ply", type=str)
parser.add_argument("num_cluster", type=int)
args = parser.parse_args()

mesh = o3d.io.read_triangle_mesh(args.ply)
mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)

filename = os.path.basename(args.ply)
filename = "{}-post_{}.ply".format(filename[:-4], args.num_cluster)
output_path = os.path.join(
    os.path.dirname(args.ply),
    filename,
)
o3d.io.write_triangle_mesh(output_path, mesh_post)
print("saved to '{}'".format(output_path))
