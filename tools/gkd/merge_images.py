import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Convert each panorama image to 6 perspective images")
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    source_path = args.source_path
    if args.target_path is None:
        target_path = source_path
    else:
        target_path = args.target_path

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    image_folder_list = os.listdir(source_path)

    for image_folder in image_folder_list:
        print(f"Processing {image_folder}...")
        image_list = os.listdir(os.path.join(source_path, image_folder))
        for image_name in tqdm(image_list):
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                os.system(f"cp {os.path.join(source_path, image_folder, image_name)} {os.path.join(target_path, image_folder + '_' + image_name)}")
        
    print("Done!")

    