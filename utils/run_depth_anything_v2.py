import add_pypath
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from internal.utils.visualizers import Visualizers
from common import find_files, AsyncNDArraySaver, AsyncImageSaver, AsyncImageReader
from distibuted_tasks import configure_arg_parser, get_task_list_with_args
from utils.distibuted_tasks import get_task_list_with_args

parser = argparse.ArgumentParser()
parser.add_argument("image_dir")
parser.add_argument("--input_size", "-s", type=int, default=518)
parser.add_argument("--output", "-o", default=None)
parser.add_argument("--downsample_factor", "-d", type=float, default=1)
parser.add_argument("--encoder", default="vitl")
parser.add_argument("--extensions", "-e", default=["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"])
parser.add_argument("--preview", "-p", action="store_true", default=False)
parser.add_argument("--colormap", type=str, default="default")
parser.add_argument("--da2_path", type=str, default=os.path.join(os.path.dirname(__file__), "Depth-Anything-V2"))
configure_arg_parser(parser)
args = parser.parse_args()

sys.path.insert(0, args.da2_path)
from depth_anything_v2.dpt import DepthAnythingV2

if args.output is None:
    args.output = os.path.join(os.path.dirname(args.image_dir), "estimated_depths")

images = get_task_list_with_args(args, find_files(args.image_dir, args.extensions, as_relative_path=False))
assert len(images) > 0, "not an image with extension name '{}' can be found in '{}'".format(args.extensions, args.image_dir)

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

depth_anything = DepthAnythingV2(**model_configs[args.encoder])
depth_anything.load_state_dict(torch.load(f'{args.da2_path}/checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()


def apply_color_map(normalized_depth):
    colored_depth = Visualizers.float_colormap(torch.from_numpy(normalized_depth).unsqueeze(0), colormap=args.colormap)
    colored_depth = (colored_depth.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    return colored_depth


ndarray_saver = AsyncNDArraySaver()
image_reader = AsyncImageReader(image_list=images)
image_saver = AsyncImageSaver(is_rgb=True)
try:
    with torch.no_grad(), tqdm(range(len(images))) as t:
        for _ in t:
            image_path, raw_image = image_reader.get()
            image_name = image_path[len(args.image_dir):].lstrip(os.path.sep)

            depth = depth_anything.infer_image(raw_image, args.input_size)
            normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())

            if args.downsample_factor != 1:
                height, width = normalized_depth.shape
                resized_height, resized_width = round(height / args.downsample_factor), round(width / args.downsample_factor)
                normalized_depth = np.array(Image.fromarray(normalized_depth).resize((resized_width, resized_height)))
            output_filename = os.path.join(args.output, "{}.npy".format(image_name))
            ndarray_saver.save(normalized_depth, output_filename)

            if args.preview is True:
                image_saver.save(normalized_depth, os.path.join(args.output, "{}.png".format(image_name)), processor=apply_color_map)
finally:
    ndarray_saver.stop()
    image_reader.stop()
    image_saver.stop()
