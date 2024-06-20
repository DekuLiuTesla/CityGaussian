import os
import sys
import yaml
import torch
import copy
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.utils.ssim import ssim
from internal.utils.blocking import contract_to_unisphere
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.dataparsers.colmap_dataparser import ColmapDataParser
from internal.models.simplified_gaussian_model_manager import SimplifiedGaussianModelManager

def parse(data):
    data = Namespace(**data)
    for arg in vars(data):
        if isinstance(getattr(data, arg), dict):
            setattr(data, arg, parse(getattr(data, arg)))
    return data

def block_merging(coarse_model, 
                  ckpt_path,
                  output_path,
                  block_dim, 
                  aabb):

        xyz_coarse = coarse_model.get_xyz
        block_num = block_dim[0] * block_dim[1] * block_dim[2]
        block_model_list = []

        if aabb is None:
            with torch.no_grad():
                sorted_x = torch.sort(xyz_coarse[::100, 0], descending=True)[0]
                sorted_y = torch.sort(xyz_coarse[::100, 1], descending=True)[0]
                sorted_z = torch.sort(xyz_coarse[::100, 2], descending=True)[0]

                ratio = 0.999
                x_max = torch.quantile(sorted_x, ratio)
                x_min = torch.quantile(sorted_x, 1-ratio)
                y_max = torch.quantile(sorted_y, ratio)
                y_min = torch.quantile(sorted_y, 1-ratio)
                z_max = torch.quantile(sorted_z, ratio)
                z_min = torch.quantile(sorted_z, 1-ratio)

                xyz_min = torch.stack([x_min, y_min, z_min])
                xyz_max = torch.stack([x_max, y_max, z_max])
                central_min = xyz_min + (xyz_max - xyz_min) / 3
                central_max = xyz_max - (xyz_max - xyz_min) / 3

                aabb = torch.cat([central_min, central_max], dim=-1)
        else:
            if len(aabb) == 4:
                aabb = [aabb[0], aabb[1], xyz_coarse[:, -1].min(), 
                        aabb[2], aabb[3], xyz_coarse[:, -1].max()]
            elif len(aabb) == 6:
                aabb = aabb
            else:
                assert False, "Unknown aabb format!"
            aabb = torch.tensor(aabb, dtype=torch.float32, device=xyz_coarse.device)
        
        for block_id in range(block_num):
            block_id_z = block_id // (block_dim[0] * block_dim[1])
            block_id_y = (block_id % (block_dim[0] * block_dim[1])) // block_dim[0]
            block_id_x = (block_id % (block_dim[0] * block_dim[1])) % block_dim[0]

            min_x, max_x = float(block_id_x) / block_dim[0], float(block_id_x + 1) / block_dim[0]
            min_y, max_y = float(block_id_y) / block_dim[1], float(block_id_y + 1) / block_dim[1]
            min_z, max_z = float(block_id_z) / block_dim[2], float(block_id_z + 1) / block_dim[2]

            block_path = os.path.join(output_path, "blocks", "block_{}".format(block_id))
            try:
                model, _ = GaussianModelLoader.search_and_load(
                    block_path,
                    sh_degree=3,
                    device="cuda",
                )
            except:
                print(f"Block {block_id} not found. Using coarse Global Model")
                model = copy.deepcopy(coarse_model)
            xyz_block = contract_to_unisphere(model.get_xyz, aabb, ord=torch.inf)
            mask_preserved = (xyz_block[:, 0] >= min_x) & (xyz_block[:, 0] < max_x)  \
                            & (xyz_block[:, 1] >= min_y) & (xyz_block[:, 1] < max_y) \
                            & (xyz_block[:, 2] >= min_z) & (xyz_block[:, 2] < max_z)
            model.delete_gaussians(~mask_preserved)
            block_model_list.append(model)
            print(f"Merged block {block_id} with {model.get_xyz.shape[0]} gaussians.")

        merged_model = SimplifiedGaussianModelManager(block_model_list, enable_transform=False, device="cuda")
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
        torch.save(checkpoint, os.path.join(save_path, "merged.ckpt"))

        # save_path = os.path.join(output_path, "point_cloud/iteration_30000")
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # merged_model.to_ply_structure().save_to_ply(os.path.join(save_path, "point_cloud.ply"))
                    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config_path', type=str, help='path of finetuned model', required=True)
    parser.add_argument("--block_dim", type=int, nargs="+", default=None)
    parser.add_argument("--aabb", type=float, nargs="+", default=None)
    parser.add_argument("--output", type=str, default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "outputs",
        ), help="the base directory of the output")
    args = parser.parse_args(sys.argv[1:])

    with open(args.config_path, 'r') as f:
        config = parse(yaml.load(f, Loader=yaml.FullLoader))
        config.name = os.path.basename(args.config_path).split(".")[0]
        args.block_dim = config.data.params.colmap_block.block_dim if args.block_dim is None else args.block_dim
        args.aabb = config.data.params.colmap_block.aabb if args.aabb is None else args.aabb
    
    coarse_model, _ = GaussianModelLoader.search_and_load(
        config.model.init_from,
        sh_degree=3,
        device="cuda",
    )
    
    output_path = os.path.join(args.output, config.name)
    ckpt_path = GaussianModelLoader.search_load_file(config.model.init_from.split("/point_cloud/")[0])

    block_merging(coarse_model, ckpt_path, output_path, args.block_dim, args.aabb)

    # All done
    print("Merging complete.")