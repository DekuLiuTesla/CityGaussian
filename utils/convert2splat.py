import add_pypath
import argparse
import numpy as np
import lightning
from internal.utils.gaussian_utils import Gaussian
from internal.utils.sh_utils import eval_sh


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to ckpt or ply file")
    parser.add_argument("--output", "-o", required=False, default=None, help="Path to output splat file")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input + ".splat"
    assert args.input != args.output

    return args


def main():
    args = getArgs()

    if args.input.endswith(".ply"):
        gaussians = Gaussian.load_from_ply(args.input)
    else:
        import torch
        ckpt = torch.load(args.input)
        gaussians = Gaussian.load_from_state_dict(ckpt["hyper_parameters"]["gaussian"].sh_degree, ckpt["state_dict"]).to_ply_format()

    # sort
    sorted_indices = np.argsort(-np.exp(gaussians.scales.sum(axis=-1)) / (1 + np.exp(-gaussians.opacities.squeeze(-1))))
    xyz_sorted = gaussians.xyz[sorted_indices].astype(np.float32)
    scales_sorted = gaussians.scales[sorted_indices].astype(np.float32)
    rot_sorted = gaussians.rotations[sorted_indices]
    features_dc = gaussians.features_dc[sorted_indices]
    opacities_sorted = gaussians.opacities[sorted_indices]
    # attribute preprocess
    scales_sorted_activated = np.exp(scales_sorted).astype(np.float32)
    rot_sorted_processed = ((rot_sorted / np.linalg.norm(rot_sorted, axis=-1, keepdims=True)) * 128 + 128).clip(0, 255)
    rgbs = eval_sh(0, features_dc, None) + 0.5
    alphas = 1. / (1 + np.exp(-opacities_sorted))
    rgbas = (np.concatenate([rgbs, alphas], axis=-1) * 255).clip(0, 255)
    # define splat file structure
    dtype = np.dtype([
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("s1", np.float32),
        ("s2", np.float32),
        ("s3", np.float32),
        ("red", np.uint8),
        ("green", np.uint8),
        ("blue", np.uint8),
        ("alpha", np.uint8),
        ("r1", np.uint8),
        ("r2", np.uint8),
        ("r3", np.uint8),
        ("r4", np.uint8),
    ])
    # rearrange attributes
    attributes = np.concatenate([xyz_sorted, scales_sorted_activated, rgbas, rot_sorted_processed], axis=-1)
    elements = np.empty(attributes.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, attributes))
    # save
    with open(args.output, "wb") as f:
        f.write(elements.tobytes())

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
