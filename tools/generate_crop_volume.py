import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import alphashape
import json
from tqdm import tqdm
from torch import nn
from plyfile import PlyData
from argparse import ArgumentParser, Namespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.utils.general_utils import parse
from internal.models.gaussian_model import GaussianModel
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.dataparsers.colmap_dataparser import ColmapParams, ColmapDataParser
from internal.utils.graphics_utils import fetch_ply
from internal.utils.general_utils import inverse_sigmoid

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ply_path', type=str, help='path of config', default='data/GauU_Scene/LFLS/LFLS_ds_35.ply')
    parser.add_argument('--transform_path', type=str, help='path of transformation matrix (txt)', default=None)
    parser.add_argument('--data_path', type=str, help='path of colmap data', default='data/GauU_Scene/LFLS')
    parser.add_argument('--split_mode', type=str, help='split mode of dataset', default='reconstruction')
    parser.add_argument('--eval_image_select_mode', type=str, help='image select mode of evaluation', default='ratio')
    parser.add_argument('--eval_step', type=int, help='ratio of evaluation', default=8)
    parser.add_argument('--eval_ratio', type=float, help='ratio of evaluation', default=0.1)
    parser.add_argument('--down_sample_factor', type=float, help='down sample factor of dataset', default=1)
    parser.add_argument('--vis_threshold', type=int, help='emperical threhold for visibility frequency', default=95)

    args = parser.parse_args(sys.argv[1:])

    # load groundtruth point cloud and dataset
    model = GaussianModel(sh_degree=3)
    model.create_from_pcd(fetch_ply(args.ply_path), device='cuda')
    model._opacity = nn.Parameter(inverse_sigmoid(torch.ones((model.get_xyz.shape[0], 1), 
                                                             dtype=torch.float, device="cuda") * 0.3))

    renderer = VanillaRenderer()
    renderer.setup(stage="val")
    renderer = renderer.to("cuda")

    dataparser_outputs = ColmapDataParser(
        os.path.expanduser(args.data_path),
        os.path.abspath(""),
        global_rank=0,
        params=ColmapParams(
            split_mode=args.split_mode,
            eval_image_select_mode=args.eval_image_select_mode,
            eval_step=args.eval_step,
            eval_ratio=args.eval_ratio,
            down_sample_factor=args.down_sample_factor,
        ),
    ).get_outputs()

    # count visibility frequency
    dataset = dataparser_outputs.train_set
    bg_color=torch.tensor([0, 0, 0], dtype=torch.float, device="cuda")
    with torch.no_grad():
        visible_cnt = torch.zeros(model.get_xyz.shape[0], dtype=torch.long, device="cuda")
        for idx in tqdm(range(0, len(dataset.cameras))):
            camera = dataset.cameras[idx].to_device("cuda")
            output = renderer(camera, model, bg_color=bg_color)
            visible_cnt[output['visibility_filter']] += 1
        xyz = np.float64(model.get_xyz.cpu().numpy())
    
    # load transformation matrix (align z axis to vertical direction)
    if args.transform_path is not None:
        with open(args.transform_path, 'r') as f:
            transform = np.loadtxt(f)
        xyz_homo = np.concatenate([xyz[:, :3], np.ones_like(xyz[:, :1])], axis=-1)
        xyz = (xyz_homo @ np.linalg.inv(transform).T)[:, :3]

    # generate alpha shape
    mask = visible_cnt.cpu().numpy() > args.vis_threshold
    hull = alphashape.alphashape(xyz[mask][::50, :2], alpha=2.0)
    if (hull.geom_type == 'MultiPolygon'):
        # if MultiPolygon, take the smallest convex Polygon containing all the points in the object
        hull = hull.convex_hull
    x, y = hull.exterior.xy
    x = np.array(x)
    y = np.array(y)
    bounding_polygon = np.stack([x, y, np.zeros_like(x)], axis=-1)

    # write json file
    content = {
        "axis_max": np.max(xyz[:, 2]),
        "axis_min": np.min(xyz[:, 2]),
        "bounding_polygon": bounding_polygon.tolist(),
        "class_name": "SelectionPolygonVolume", 
        "orthogonal_axis": "Z", 
        "version_major": 1, 
        "version_minor": 0
    }

    save_path = args.ply_path.replace('.ply', '.json')
    data = json.dumps(content, indent=1)
    with open(save_path, "w", newline='\n') as f:
        f.write(data)
    print(f'Crop Volume has been written to {save_path}')
