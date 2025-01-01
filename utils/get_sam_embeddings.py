import add_pypath
import os
import gc
import argparse
from tqdm import tqdm
from glob import glob
import cv2
import torch
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from internal.utils.seganygs import SegAnyGSUtils
from common import AsyncTensorSaver, AsyncImageReader, AsyncImageSaver
from distibuted_tasks import configure_arg_parser, get_task_list_with_args

parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, default=None)
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--sam_ckpt", "-c", type=str, default="sam_vit_h_4b8939.pth")
parser.add_argument("--sam_arch", type=str, default="vit_h")
parser.add_argument("--preview", action="store_true", default=True)
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
feature_dir = os.path.join(output_path, "semantic", "sam_features")
feature_preview_dir = os.path.join(output_path, "semantic", "sam_feature_preview")
os.makedirs(feature_dir, exist_ok=True)
os.makedirs(feature_preview_dir, exist_ok=True)

# initialize SAM
print("Initializing SAM...")
model_type = args.sam_arch
sam = sam_model_registry[model_type](checkpoint=args.sam_ckpt).to(MODEL_DEVICE)
predictor = SamPredictor(sam)

print("Finding image files...")
image_name_set = {}
for e in args.ext:
    for i in glob(os.path.join(image_path, f"**/*.{e}"), recursive=True):
        image_name_set[i] = True
image_list = list(image_name_set.keys())
assert len(image_list) > 0, "Not a image can be found"
print(f"{len(image_list)} images found")
image_list.sort()

image_list = get_task_list_with_args(args, image_list)

image_reader = AsyncImageReader(image_list)
image_saver = AsyncImageSaver()
tensor_saver = AsyncTensorSaver()
try:
    with tqdm(range(len(image_list))) as t:
        for _ in t:
            # get image information
            image_full_path, img = image_reader.queue.get()
            image_name = image_full_path[len(image_path):].lstrip("/")

            t.set_description(f"{image_name}")
            semantic_file_name = f"{image_name}.pt"

            # predict image embedding
            min_size = min(*img.shape[:2])
            img = cv2.resize(img, dsize=(min_size, min_size), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            predictor.set_image(img)
            image_embedding = predictor.get_image_embedding().squeeze()  # [C, H, W]

            # save
            tensor_saver.save(image_embedding, os.path.join(feature_dir, semantic_file_name))

            # preview
            if args.preview is True:
                image_embedding_flatten_normalized = torch.nn.functional.normalize(image_embedding.permute(1, 2, 0).reshape((-1, image_embedding.shape[0])), dim=-1)
                pca_projection_matrix = SegAnyGSUtils.get_pca_projection_matrix(semantic_features=image_embedding_flatten_normalized)
                pca_color = SegAnyGSUtils.get_pca_projected_colors(image_embedding_flatten_normalized, pca_projection_matrix)
                feature_preview = pca_color.reshape((image_embedding.shape[1], image_embedding.shape[2], -1))
                feature_preview = (feature_preview * 255).to(torch.uint8).cpu().numpy()
                image_saver.save(feature_preview, os.path.join(feature_preview_dir, f"{image_name}.png"))
finally:
    image_reader.stop()
    image_saver.stop()
    tensor_saver.stop()
