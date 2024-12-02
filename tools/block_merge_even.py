import os
import sys
import yaml
import torch
import copy
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.utils.ssim import ssim
from internal.utils.blocking import contract_to_unisphere
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.general_utils import focus_point_fn
from internal.utils.general_utils import parse
from internal.dataparsers.colmap_block_dataparser import ColmapBlockDataParser
from internal.dataparsers.estimated_depth_colmap_block_dataparser import EstimatedDepthColmapDataParser
from internal.renderers.vanilla_trim_renderer import VanillaTrimRenderer
from internal.models.simplified_gaussian_model_manager import SimplifiedGaussianModelManager

def block_merging(coarse_model, 
                  ckpt_path,
                  output_path,
                  block_dim, 
                  xyz_quantile,
                  flatten_gs):

        block_num = block_dim[0] * block_dim[1] * block_dim[2]
        block_model_list = []

        x_quantile = xyz_quantile["x"]
        y_quantile = xyz_quantile["y"]
        z_quantile = xyz_quantile["z"]

        for block_id in range(block_num):
            block_id_z = block_id // (block_dim[0] * block_dim[1])
            block_id_y = (block_id % (block_dim[0] * block_dim[1])) // block_dim[0]
            block_id_x = (block_id % (block_dim[0] * block_dim[1])) % block_dim[0]

            min_x, max_x = x_quantile[block_id_x].item(), x_quantile[block_id_x + 1].item()
            min_y, max_y = y_quantile[block_id_y].item(), y_quantile[block_id_y + 1].item()
            min_z, max_z = z_quantile[block_id_z].item(), z_quantile[block_id_z + 1].item()

            try:
                block_path = os.path.join(output_path, "blocks", "block_{}".format(block_id))
                model_path = GaussianModelLoader.search_load_file(block_path)
                model, _ = GaussianModelLoader.search_and_load(
                    block_path,
                    sh_degree=3,
                    device="cuda",
                )
            except:
                print(f"Block {block_id} not trained. Using coarse Global Model")
                model_path = ckpt_path
                model = copy.deepcopy(coarse_model)
                
            xyz_block = model.get_xyz
            mask_preserved = (xyz_block[:, 0] >= min_x) & (xyz_block[:, 0] < max_x)  \
                            & (xyz_block[:, 1] >= min_y) & (xyz_block[:, 1] < max_y) \
                            & (xyz_block[:, 2] >= min_z) & (xyz_block[:, 2] < max_z)
            model.delete_gaussians(~mask_preserved)
            block_model_list.append(model)
            print(f"Merged block {block_id} with {model.get_xyz.shape[0]} gaussians from {model_path}.")

        merged_model = SimplifiedGaussianModelManager(block_model_list, enable_transform=False, device="cuda")
        if flatten_gs:
            merged_model._scaling = merged_model._scaling[:, :2]
        print(f"Total {merged_model.get_xyz.shape[0]} gaussians to be saved.")

        save_path = os.path.join(output_path, "checkpoints")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        checkpoint = torch.load(ckpt_path)
        merged_model = merged_model.to_parameter_structure()
        checkpoint["state_dict"]["gaussian_model._xyz"] = merged_model.xyz
        checkpoint["state_dict"]["gaussian_model._opacity"] = merged_model.opacities
        checkpoint["state_dict"]["gaussian_model._features_dc"] = merged_model.features_dc
        checkpoint["state_dict"]["gaussian_model._features_rest"] = merged_model.features_rest
        checkpoint["state_dict"]["gaussian_model._scaling"] = merged_model.scales
        checkpoint["state_dict"]["gaussian_model._rotation"] = merged_model.rotations
        checkpoint["state_dict"]["gaussian_model._features_extra"] = merged_model.real_features_extra
        
        ckpt_name = ckpt_path.split('/')[-1].split('.')[0]
        torch.save(checkpoint, os.path.join(save_path, f"{ckpt_name}.ckpt"))

        # save_path = os.path.join(output_path, "point_cloud/iteration_30000")
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # merged_model.to_ply_structure().save_to_ply(os.path.join(save_path, "point_cloud.ply"))
                    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config_path', type=str, help='path of finetuned model', required=True)
    parser.add_argument("--block_dim", type=int, nargs="+", default=None)
    parser.add_argument("--output", type=str, default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "outputs",
        ), help="the base directory of the output")
    args = parser.parse_args(sys.argv[1:])

    with open(args.config_path, 'r') as f:
        config = parse(yaml.load(f, Loader=yaml.FullLoader))
        config.name = os.path.basename(args.config_path).split(".")[0]
        if config.data.type == "estimated_depth_colmap_block":
            params = config.data.params.estimated_depth_colmap_block
        else:
            params = config.data.params.colmap_block
        args.xyz_quantile = torch.load(os.path.join(params.image_list, "quantiles.pt"))
        args.block_dim = params.block_dim if args.block_dim is None else args.block_dim
    
    coarse_model, renderer = GaussianModelLoader.search_and_load(
        config.model.init_from,
        sh_degree=3,
        device="cuda",
    )
    flatten_gs = True if isinstance(renderer, VanillaTrimRenderer) else False

    txt_dict = {}
    image_list = params.image_list
    for block_id in range(args.block_dim[0] * args.block_dim[1] * args.block_dim[2]):
        with open(os.path.join(image_list, f"block_{block_id}.txt"), 'r') as f:
            txt_dict[block_id] = f.readlines()
    
    output_path = os.path.join(args.output, config.name)
    ckpt_path = GaussianModelLoader.search_load_file(config.model.init_from.split("/point_cloud/")[0])

    block_merging(coarse_model, ckpt_path, output_path, args.block_dim, args.xyz_quantile, flatten_gs)

    # All done
    print("Merging complete.")