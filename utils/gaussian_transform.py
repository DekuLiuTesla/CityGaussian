import add_pypath
import os
import argparse
import json
import numpy as np
import torch
from internal.utils.rotation import rotation_matrix
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gaussian_model_editor import MultipleGaussianModelEditor
from internal.utils.gaussian_utils import GaussianPlyUtils
from viser import transforms as vt


class RotationUtils:
    @staticmethod
    def rx(theta):
        return np.matrix([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])

    @staticmethod
    def ry(theta):
        return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])

    @staticmethod
    def rz(theta):
        return np.matrix([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")

    # translation
    parser.add_argument("--tx", type=float, default=0)
    parser.add_argument("--ty", type=float, default=0)
    parser.add_argument("--tz", type=float, default=0)

    # rotation in euler angeles
    parser.add_argument("--rx", type=float, default=0, help="in radians")
    parser.add_argument("--ry", type=float, default=0, help="in radians")
    parser.add_argument("--rz", type=float, default=0, help="in radians")

    # scale
    parser.add_argument("--scale", type=float, default=1)

    parser.add_argument("--sh-factor", type=float, default=1.0)

    # auto reorient
    parser.add_argument("--auto-reorient", action="store_true", default=False)
    parser.add_argument("--cameras-json", type=str, default=None)

    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    args.input = os.path.expanduser(args.input)
    args.output = os.path.expanduser(args.output)

    return args


@torch.no_grad()
def main():
    args = parse_args()
    assert args.input != args.output
    assert args.scale > 0
    assert os.path.exists(args.input)

    # load model
    device = torch.device(args.device)
    gaussian_model, _ = GaussianModelLoader.search_and_load(args.input, device=device, eval_mode=True, pre_activate=False)
    gaussian_model_editor = MultipleGaussianModelEditor([gaussian_model], device=device)

    # calculate rotation matrix
    if args.auto_reorient is True:
        cameras_json_path = args.cameras_json
        if cameras_json_path is None:
            cameras_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.input))), "cameras.json")
        with open(cameras_json_path, "r") as f:
            cameras = json.load(f)
        up = np.zeros(3)
        for i in cameras:
            up += np.asarray(i["rotation"])[:3, 1]
        up = -up / np.linalg.norm(up)
        print("up vector = {}".format(up))

        rotation = rotation_matrix(torch.tensor(up, dtype=torch.float), torch.Tensor([0, 0, 1])).numpy()

        print("rotation matrix = {}".format(rotation))

        rot_mat = rotation
    else:
        rot_mat = RotationUtils.rx(args.rx) @ RotationUtils.ry(args.ry) @ RotationUtils.rz(args.rz)

    # transform
    print("transforming...")
    gaussian_model_editor.transform_with_vectors(
        idx=0,
        scale=args.scale,
        r_wxyz=vt.SO3.from_matrix(rot_mat).wxyz,
        t_xyz=np.asarray([args.tx, args.ty, args.tz]),
    )

    # rescale SHs
    if args.sh_factor != 1.:
        print("rescaling SHs...")
        gaussian_model.shs_dc *= args.sh_factor
        gaussian_model.shs_rest *= args.sh_factor

    # save to ply file
    print("saving...")
    GaussianPlyUtils.load_from_model(gaussian_model).to_ply_format().save_to_ply(args.output)

    print(args.output)


main()
