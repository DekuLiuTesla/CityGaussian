#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
from arguments import GroupParams
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_symmetric(uncertainty):
    L = torch.zeros((uncertainty.shape[0], 3, 3), dtype=torch.float, device="cuda")

    L[:, 0, 0] = uncertainty[:, 0]
    L[:, 0, 1] = uncertainty[:, 1]
    L[:, 0, 2] = uncertainty[:, 2]
    L[:, 1, 1] = uncertainty[:, 3]
    L[:, 1, 2] = uncertainty[:, 4]
    L[:, 2, 2] = uncertainty[:, 5]

    L[:, 1, 0] = L[:, 0, 1]
    L[:, 2, 0] = L[:, 0, 2]
    L[:, 2, 1] = L[:, 1, 2]
    return L

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def get_default_lp():
    lp = GroupParams()
    lp.config = None
    lp.sh_degree = 3
    lp.source_path = ""
    lp.model_path = ""
    lp.images = "images"
    lp.resolution = -1
    lp.white_background = False
    lp.data_device = "cuda"
    lp.eval = False
    lp.llffhold = 8
    # data partitioning
    lp.pretrain_path = None  # path to coarse global model
    lp.num_threshold = 25_000  # threshold of point number
    lp.ssim_threshold = 0.08  # threshold of ssim difference
    # finetuning
    lp.partition_name = ""  # filename of .npy partition file
    lp.block_dim = None  # block dimensions
    lp.block_id = -1  # block id
    lp.aabb = None  # foreground area in contraction
    lp.save_block_only = True  # whether to only store gaussians in blocks
    # lod rendering
    lp.lod_configs = None  # list of paths to different detail levels, used only in LoD rendering
    # others
    lp.add_background_sphere = False
    lp.logger_config = None

    return lp

def get_default_op():
    op = GroupParams()
    op.iterations = 30_000
    op.position_lr_init = 0.00016
    op.position_lr_final = 0.0000016
    op.position_lr_delay_mult = 0.01
    op.position_lr_max_steps = 30_000
    op.feature_lr = 0.0025
    op.opacity_lr = 0.05
    op.scaling_lr = 0.005
    op.rotation_lr = 0.001
    op.percent_dense = 0.01
    op.lambda_dssim = 0.2
    op.densification_interval = 100
    op.opacity_reset_interval = 3000
    op.densify_from_iter = 500
    op.densify_until_iter = 15_000
    op.densify_grad_threshold = 0.0002
    op.max_cache_num = 512

    return op

def get_default_pp():
    pp = GroupParams()
    pp.convert_SHs_python = False
    pp.compute_cov3D_python = False
    pp.debug = False

    return pp

def extract_args(params, cfg, args=None):
    for arg in cfg.items():
        setattr(params, arg[0], arg[1])

    if args is not None:
        for arg in vars(args).items():
            if arg[0] in vars(params):
                setattr(params, arg[0], arg[1])


def parse_cfg(cfg, args):
    lp = get_default_lp()
    op = get_default_op()
    pp = get_default_pp()

    extract_args(lp, cfg['model_params'], args)
    extract_args(op, cfg['optim_params'], args)
    extract_args(pp, cfg['pipeline_params'], args)

    return lp, op, pp

