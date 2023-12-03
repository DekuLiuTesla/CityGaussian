import os
import sys
import shutil
import json
import numpy as np
from PIL import Image
from argparse import ArgumentParser

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Convert transforms.json from UE to 3DGS acceptable format")
    parser.add_argument("--dense_path", type=str, required=True)
    parser.add_argument("--sparse_path", type=str, required=True)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--skip_image", action="store_true", help="Skip images or not.")
    args = parser.parse_args(sys.argv[1:])

    interval = args.interval
    dense_path = args.dense_path
    dense_json_path = os.path.join(args.dense_path, 'transforms_train.json')
    sparse_path = args.sparse_path
    transforms_path_out = os.path.join(sparse_path, 'transforms_train.json')
    out_contents = {}

    if not os.path.exists(sparse_path):
        os.mkdir(sparse_path)
    if not os.path.exists(os.path.join(sparse_path, 'input')):
        os.mkdir(os.path.join(sparse_path, 'input'))
    # TODO: add directory cleaning

    with open(dense_json_path) as json_file:
        contents = json.load(json_file)

    # initialize out_contents with the same keys as ref_contents
    for key in contents.keys():
        if key == 'frames':
            out_contents[key] = []
        else:
            out_contents[key] = contents[key]

    for idx, frame in enumerate(contents['frames']):
        if (idx // 5) % interval == 0:
            source_path = os.path.join(dense_path, frame['file_path']+'.jpg')
            target_path = os.path.join(sparse_path, frame['file_path']+'.jpg')
            
            out_contents['frames'].append({'file_path': frame['file_path'], 
                                           'transform_matrix': frame['transform_matrix']})
            if not args.skip_image:
                shutil.copy(source_path, target_path)

    with open(transforms_path_out, 'w') as json_file:
        json.dump(out_contents, json_file, indent=4)
    
    print(f"Saved {transforms_path_out}")