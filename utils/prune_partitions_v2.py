import add_pypath
import os
import argparse
import gc
import torch
import json
from tqdm.auto import tqdm
from internal.cameras.cameras import Cameras
from internal.utils.light_gaussian import get_count_and_score, calculate_v_imp_score, get_prune_mask
from trained_partition_utils import get_trained_partitions, split_partition_gaussians
from distibuted_tasks import configure_arg_parser_v2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("partition_dir")
    parser.add_argument("--project_name", "-p", type=str, required=True, help="Project name")
    parser.add_argument("--min-images", type=int, default=32)
    parser.add_argument("--prune-percent", type=float, default=0.6)
    configure_arg_parser_v2(parser)
    return parser.parse_args()


def parse_cameras_json(path: str):
    with open(path, "r") as f:
        cameras = json.load(f)

    c2w_list = []
    width_list = []
    height_list = []
    fx_list = []
    fy_list = []
    cx_list = []
    cy_list = []

    for i in cameras:
        c2w = torch.eye(4)
        c2w[:3, :3] = torch.tensor(i["rotation"])
        c2w[:3, 3] = torch.tensor(i["position"])
        c2w_list.append(c2w)

        width_list.append(i["width"])
        height_list.append(i["height"])
        fx_list.append(i["fx"])
        fy_list.append(i["fy"])
        cx_list.append(i.get("cx", i["width"] / 2))
        cy_list.append(i.get("cy", i["height"] / 2))

    w2c = torch.linalg.inv(torch.stack(c2w_list))

    return Cameras(
        R=w2c[..., :3, :3],
        T=w2c[..., :3, 3],
        fx=torch.tensor(fx_list),
        fy=torch.tensor(fy_list),
        cx=torch.tensor(cx_list),
        cy=torch.tensor(cy_list),
        width=torch.tensor(width_list, dtype=torch.int),
        height=torch.tensor(height_list, dtype=torch.int),
        appearance_id=torch.zeros(w2c.shape[0], dtype=torch.int),
        normalized_appearance_id=torch.zeros(w2c.shape[0], dtype=torch.float),
        distortion_params=torch.zeros((w2c.shape[0], 4), dtype=torch.float),
        camera_type=torch.zeros(w2c.shape[0], dtype=torch.int),
    )


def main():
    args = parse_args()

    torch.autograd.set_grad_enabled(False)

    _, trained_partitions, orientation_transformation = get_trained_partitions(
        partition_dir=args.partition_dir,
        project_name=args.project_name,
        min_images=args.min_images,
        n_processes=args.n_processes,
        process_id=args.process_id,
    )

    n_before_pruning = 0
    n_after_pruning = 0

    with tqdm(trained_partitions) as t:
        for partition_idx, partition_id_str, ckpt_file, bounding_box in t:
            t.set_description(partition_id_str)
            t.set_postfix_str("Loading checkpoint...")
            ckpt = torch.load(ckpt_file, map_location="cpu")

            t.set_postfix_str("Splitting..")
            gaussian_model, outside_part, is_inside = split_partition_gaussians(
                ckpt,
                bounding_box,
                orientation_transformation,
            )
            n_before_pruning += gaussian_model.n_gaussians

            cameras = parse_cameras_json(os.path.join(
                os.path.dirname(os.path.dirname(ckpt_file)),
                "cameras.json",
            ))

            gaussian_model.to("cuda")

            # ===
            t.set_postfix_str("Pruning...")
            _, opacity_score_total, _, visibility_score_total = get_count_and_score(
                gaussian_model,
                tqdm(cameras, leave=False),
                anti_aliased=True,
            )
            # prune with zero visibilities
            nonzero_visibility_mask = ~torch.isclose(visibility_score_total, torch.tensor(0., device=visibility_score_total.device))
            gaussian_model.properties = {k: v[nonzero_visibility_mask] for k, v in gaussian_model.properties.items()}
            opacity_score_total = opacity_score_total[nonzero_visibility_mask]

            # prune by opacity
            v_imp_score = calculate_v_imp_score(gaussian_model.get_scaling, opacity_score_total, 0.1)
            high_opacity_score_mask = ~get_prune_mask(args.prune_percent, v_imp_score)
            gaussian_model.properties = {k: v[high_opacity_score_mask] for k, v in gaussian_model.properties.items()}

            n_after_pruning += gaussian_model.n_gaussians

            # ===

            # update gaussian states of ckpt
            for k, v in gaussian_model.state_dict().items():
                state_dict_full_key = "gaussian_model.{}".format(k)
                assert state_dict_full_key in ckpt["state_dict"]
                ckpt["state_dict"][state_dict_full_key] = v

            # move outside part to frozen states
            for k, v in outside_part.items():
                frozen_key = "frozen_gaussians.{}".format(k)

                # concat existing frozen gaussians
                if frozen_key in ckpt["state_dict"]:
                    v = torch.concat([ckpt["state_dict"][frozen_key], v], dim=0)

                ckpt["state_dict"][frozen_key] = v

            # prune optimizer state
            optimizer_prunable_states = ["exp_avg", "exp_avg_sq"]
            property_names = list(gaussian_model.property_names)
            for optimizer_state in ckpt["optimizer_states"]:
                if len(property_names) == 0:
                    break

                for param_group_idx, param_group in enumerate(optimizer_state["param_groups"]):
                    # whether a gaussian param_group
                    param_group_name = param_group["name"]
                    if param_group_name not in property_names:
                        continue

                    inside_gaussian_states = {
                        k: optimizer_state["state"][param_group_idx][k][is_inside][nonzero_visibility_mask.cpu()][high_opacity_score_mask.cpu()]
                        for k in optimizer_prunable_states
                    }

                    # replace optimizer states
                    for k in optimizer_prunable_states:
                        optimizer_state["state"][param_group_idx][k] = inside_gaussian_states[k]

                    property_names.remove(param_group_name)

            # prune density controller state_dict by simply replacing with zeros
            for i in ckpt["state_dict"]:
                if i.startswith("density_controller."):
                    ckpt["state_dict"][i] = torch.zeros((gaussian_model.n_gaussians, *ckpt["state_dict"][i].shape[1:]))

            # save checkpoint
            checkpoint_save_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt_file)), "pruned_checkpoints")
            os.makedirs(checkpoint_save_dir, exist_ok=True)
            checkpoint_save_path = os.path.join(checkpoint_save_dir, f"latest-opacity_pruned-{args.prune_percent}.ckpt")
            t.set_postfix_str("Saving...")
            torch.save(ckpt, checkpoint_save_path)

            del ckpt
            del gaussian_model
            del inside_gaussian_states
            del outside_part
            gc.collect()
            torch.cuda.empty_cache()

    print("{}/{}".format(n_after_pruning, n_before_pruning))


if __name__ == "__main__":
    main()
