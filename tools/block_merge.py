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
from internal.utils.mesh_utils import focus_point_fn
from internal.dataparsers.colmap_block_dataparser import ColmapBlockDataParser
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
                  aabb,
                  corr_matrix):

        xyz_coarse = coarse_model.get_xyz
        block_num = block_dim[0] * block_dim[1] * block_dim[2]
        block_model_list = []
        trained_block_mask = torch.tensor([True if -corr_matrix[block_id, block_id] >= 50 else False for block_id in range(block_num)], dtype=torch.bool)
        trained_block_idx = torch.where(trained_block_mask)[0]

        if aabb is None:
            coarse_config_path = os.path.join(ckpt_path.split('checkpoints')[0], "config.yaml")
            with open(coarse_config_path, 'r') as f:
                coarse_config = parse(yaml.load(f, Loader=yaml.FullLoader))

            # TODO: support other data parser
            dataset = ColmapBlockDataParser(
                os.path.expanduser(coarse_config.data.path),
                os.path.abspath(""),
                global_rank=0,
                params=coarse_config.data.params.colmap_block,
            ).get_outputs().train_set

            torch.cuda.empty_cache()
            c2ws = np.array([np.linalg.inv(np.asarray((cam.world_to_camera.T).cpu().numpy())) for cam in dataset.cameras])
            poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
            center = (focus_point_fn(poses))
            radius = torch.tensor(np.median(np.abs(c2ws[:,:3,3] - center), axis=0), device=xyz_coarse.device)
            center = torch.from_numpy(center).float().to(xyz_coarse.device)
            if radius.min() / radius.max() < 0.02:
                # If the radius is too small, we don't contract in this dimension
                radius[torch.argmin(radius)] = 0.5 * (xyz_coarse[:, torch.argmin(radius)].max() - xyz_coarse[:, torch.argmin(radius)].min())
            aabb = torch.zeros(6, device=xyz_coarse.device)
            aabb[:3] = center - radius
            aabb[3:] = center + radius
        else:
            assert len(aabb) == 6, "Unknown aabb format!"
            aabb = torch.tensor(aabb, dtype=torch.float32, device=xyz_coarse.device)
        
        for block_id in range(block_num):
            block_id_z = block_id // (block_dim[0] * block_dim[1])
            block_id_y = (block_id % (block_dim[0] * block_dim[1])) // block_dim[0]
            block_id_x = (block_id % (block_dim[0] * block_dim[1])) % block_dim[0]

            min_x, max_x = float(block_id_x) / block_dim[0], float(block_id_x + 1) / block_dim[0]
            min_y, max_y = float(block_id_y) / block_dim[1], float(block_id_y + 1) / block_dim[1]
            min_z, max_z = float(block_id_z) / block_dim[2], float(block_id_z + 1) / block_dim[2]

            try:
                block_path = os.path.join(output_path, "blocks", "block_{}".format(block_id))
                model, _ = GaussianModelLoader.search_and_load(
                    block_path,
                    sh_degree=3,
                    device="cuda",
                )
            except:
                if -corr_matrix[block_id, block_id] < 50:
                    correlated_block = trained_block_idx[torch.argmax(corr_matrix[block_id, trained_block_mask])]
                    if corr_matrix[block_id, correlated_block] > 0:
                        print(f"Block {block_id} has no enough training data. Merging from block {correlated_block}")
                        block_path = os.path.join(output_path, "blocks", "block_{}".format(correlated_block))
                        model, _ = GaussianModelLoader.search_and_load(
                            block_path,
                            sh_degree=3,
                            device="cuda",
                        )
                    else:
                        print(f"Block {block_id} not trained. Using coarse Global Model")
                        model = copy.deepcopy(coarse_model)
                else:
                    raise FileNotFoundError
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
        if args.aabb is None and hasattr(config.data.params.colmap_block, "aabb"):
            args.aabb = config.data.params.colmap_block.aabb
    
    coarse_model, _ = GaussianModelLoader.search_and_load(
        config.model.init_from,
        sh_degree=3,
        device="cuda",
    )


    txt_dict = {}
    image_list = config.data.params.colmap_block.image_list
    for block_id in range(args.block_dim[0] * args.block_dim[1] * args.block_dim[2]):
        with open(os.path.join(image_list, f"block_{block_id}.txt"), 'r') as f:
            txt_dict[block_id] = f.readlines()
    
    corr_matrix = torch.zeros((args.block_dim[0] * args.block_dim[1] * args.block_dim[2], 
                                args.block_dim[0] * args.block_dim[1] * args.block_dim[2]), dtype=torch.int32)
    for i in range(args.block_dim[0] * args.block_dim[1] * args.block_dim[2]):
        for j in range(i+1, args.block_dim[0] * args.block_dim[1] * args.block_dim[2]):
            corr_matrix[i, j] = len(set(txt_dict[i]) & set(txt_dict[j]))
            corr_matrix[j, i] = corr_matrix[i, j]
        corr_matrix[i, i] = -len(txt_dict[i])
    
    output_path = os.path.join(args.output, config.name)
    ckpt_path = GaussianModelLoader.search_load_file(config.model.init_from.split("/point_cloud/")[0])

    block_merging(coarse_model, ckpt_path, output_path, args.block_dim, args.aabb, corr_matrix)

    # All done
    print("Merging complete.")