import add_pypath
import os
import gc
import argparse
from tqdm import tqdm
from glob import glob
import cv2
import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from common import AsyncTensorSaver, AsyncImageReader, AsyncImageSaver
from distibuted_tasks import configure_arg_parser, get_task_list_with_args

parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, default=None)
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--sam_ckpt", "-c", type=str, default="sam_vit_h_4b8939.pth")
parser.add_argument("--sam_arch", type=str, default="vit_h")
parser.add_argument("--preview", action="store_true", default=False)
parser.add_argument("--ext", "-e", nargs="+", default=["jpg", "jpeg", "JPG", "JPEG"])
configure_arg_parser(parser)
args = parser.parse_args()

MODEL_DEVICE = "cuda"

# build paths
image_path = args.image_path
output_path = args.output
if output_path is None:
    output_path = os.path.join(os.path.dirname(image_path.rstrip("/")))
print(f"output_path={os.path.join(output_path, 'semantic')}")

# build output dirs
mask_dir = os.path.join(output_path, "semantic", "masks")
mask_preview_dir = os.path.join(output_path, "semantic", "masks_preview")
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(mask_preview_dir, exist_ok=True)

# initialize SAM
print("Initializing SAM...")
model_type = args.sam_arch
sam = sam_model_registry[model_type](checkpoint=args.sam_ckpt).to('cuda')
# predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    box_nms_thresh=0.7,
    stability_score_thresh=0.95,
    crop_n_layers=0,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=100,
)

print("Finding image files...")
image_name_set = {}
for e in args.ext:
    for i in glob(os.path.join(image_path, f"**/*.{e}"), recursive=True):
        image_name_set[i] = True
image_list = list(image_name_set.keys())
n_found_images = len(image_list)
assert n_found_images > 0, "Not a image can be found"
image_list.sort()

image_list = get_task_list_with_args(args, image_list)
print("extract masks from {} of {} images".format(len(image_list), n_found_images))

image_reader = AsyncImageReader(image_list)
image_saver = AsyncImageSaver()
tensor_saver = AsyncTensorSaver()
try:
    with tqdm(range(len(image_list))) as t:
        for _ in t:
            # get image information
            image_full_path, img = image_reader.queue.get()
            image_name = image_full_path[len(image_path):].lstrip("/")

            t.set_description(f"Generating masks: {image_name}")
            semantic_file_name = f"{image_name}.pt"

            # read image
            # img = cv2.imread(image_path)
            # img = image_queue.get()

            # extract masks
            masks = mask_generator.generate(img)
            if args.preview is True:
                img_tensor = torch.tensor(img, dtype=torch.float, device="cuda")
            mask_list = []
            for m in masks:
                # TODO: resize
                # discard if all/none pixels are masked
                if np.all(m['segmentation']) or np.all(np.logical_not(m['segmentation'])):
                    continue
                mask_list.append(torch.from_numpy(m['segmentation']))

                if args.preview is True:
                    # preview masks
                    color_mask = torch.rand((1, 1, 3), dtype=torch.float, device=img_tensor.device) * 255
                    transparency = (torch.tensor(m['segmentation'], dtype=torch.float, device=img_tensor.device) * 0.5)[..., None]
                    img_tensor = img_tensor * (1 - transparency) + transparency * color_mask

                    image_saver.save(img_tensor.to(torch.uint8).cpu().numpy(), os.path.join(mask_preview_dir, f"{image_name}.png"))

            tensor_saver.save(torch.stack(mask_list, dim=0), os.path.join(mask_dir, semantic_file_name))
finally:
    image_reader.stop()
    image_saver.stop()
    tensor_saver.stop()
