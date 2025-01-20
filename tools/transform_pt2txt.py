import os
import sys
import json
import numpy as np
import torch
import add_pypath
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from internal.utils.colmap import rotmat2qvec

RDF_TO_DRB = torch.FloatTensor([[0, 1, 0],
                                [1, 0, 0],
                                [0, 0, -1]])

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Convert transforms_train.json to txt required by COLMAP")
    parser.add_argument("--source_path", type=str, required=True)
    args = parser.parse_args(sys.argv[1:])

    train_idx, val_idx = 0, 0

    source_path = Path(args.source_path)
    coordinates = torch.load(source_path / 'coordinates.pt')
    with (source_path / 'mappings.txt').open() as f:
        for line in tqdm(f):
            image_name, metadata_name = line.strip().split(',')
            metadata_path = source_path / 'train' / 'metadata' / metadata_name
            sparse_path = source_path / 'train' / 'sparse'
            if not metadata_path.exists():
                metadata_path = source_path / 'val' / 'metadata' / metadata_name
                sparse_path = source_path / 'val' / 'sparse'
                assert metadata_path.exists()
                val_idx += 1
                idx = val_idx
            else:
                train_idx += 1
                idx = train_idx
            
            if not sparse_path.exists():
                sparse_path.mkdir()

            metadata = torch.load(metadata_path, map_location='cpu')
            intrinsics = metadata['intrinsics']
            c2w = metadata['c2w']
            
            if not (sparse_path / 'cameras.txt').exists():
                # create cameras.txt under sparse_path, write only once
                with open(sparse_path / 'cameras.txt', 'w') as f:
                    f.write('# Camera list with one line of data per camera:\n')
                    f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
                    f.write('# Number of cameras: 1\n')
                    WIDTH = metadata['W']
                    HEIGHT = metadata['H']
                    f.write('1 PINHOLE {} {} {} {} {} {}\n'.format(WIDTH, HEIGHT, intrinsics[0],  intrinsics[1], 
                                                                   intrinsics[2], intrinsics[3]))
            
            if not (sparse_path / 'points3D.txt').exists():
                # create empty points3D.txt under sparse_path, write only once
                with open(sparse_path / 'points3D.txt', 'w') as f:
                    f.write('')
            
            if not (sparse_path / 'images.txt').exists():
                # create images.txt under sparse_path, write only once
                with open(sparse_path / 'images.txt', 'w') as f:
                    f.write('# Image list with two lines of data per image:\n')
                    f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
                    f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
                    f.write('# Number of images: {}\n'.format(len(os.listdir(sparse_path.parent / 'input'))))
            
            with open(sparse_path / 'images.txt', 'a') as f:
                # change from MegaNeRF camera axes (Y right, Z back) to COLMAP (Y down, Z forward)
                c2w_4x4 = torch.eye(4)
                c2w = torch.cat([-c2w[:, 1:2], c2w[:, 0:1], c2w[:, 2:4]], -1)
                c2w[:, 3] = c2w[:, 3] * coordinates['pose_scale_factor'] + coordinates['origin_drb']
                c2w_4x4[:3, :3] = torch.inverse(RDF_TO_DRB) @ c2w[:3, :3] @ RDF_TO_DRB
                c2w_4x4[:3, 3:] = torch.inverse(RDF_TO_DRB) @ c2w[:3, 3:]

                # c2w_4x4[:3,3] /= 100  # postion scale down for convenient visualization
                
                w2c = np.linalg.inv(c2w_4x4.numpy())
                qw, qx, qy, qz = rotmat2qvec(w2c[:3, :3])
                tx, ty, tz = w2c[:3, 3]
                filename = '{}.{}'.format(metadata_path.stem, image_name.split('.')[-1])
                f.write('{} {} {} {} {} {} {} {} 1 {}\n'.format(idx, qw, qx, qy, qz, tx, ty, tz, filename))
                f.write('\n')  # make sure every other line is empty
    
    print(f"Saved {sparse_path}")