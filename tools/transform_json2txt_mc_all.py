import os
import sys
import json
import numpy as np
from argparse import ArgumentParser
from transforms3d.quaternions import mat2quat

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Convert transforms_train.json to txt required by COLMAP")
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--intrinsic_path", type=str, default="data/matrix_city/aerial/pose/block_all/transforms_train.json")
    args = parser.parse_args(sys.argv[1:])

    source_path = args.source_path
    intrinsic_path = args.intrinsic_path
    sparse_path = os.path.join(source_path, 'sparse')
    if not os.path.exists(sparse_path):
        os.mkdir(sparse_path)
    transforms_path = os.path.join(source_path, 'transforms.json')
    with open(transforms_path) as json_file:
        contents = json.load(json_file)
    with open(intrinsic_path) as json_file:
        intrinsic = json.load(json_file)

    # create cameras.txt under sparse_path
    cameras_path = os.path.join(sparse_path, 'cameras.txt')
    with open(cameras_path, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write('# Number of cameras: 1\n')
        WIDTH = intrinsic['w']
        HEIGHT = intrinsic['h']
        f.write('1 PINHOLE {} {} {} {} {} {}\n'.format(WIDTH, HEIGHT, intrinsic['fl_x'], intrinsic['fl_y'], 
                                                    intrinsic['cx'], intrinsic['cy']))
    # create images.txt under sparse_path
    images_path = os.path.join(sparse_path, 'images.txt')
    with open(images_path, 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write('# Number of images: {}\n'.format(len(contents['frames'])))

        parent_dir = os.path.dirname(source_path)
        for idx, frame in enumerate(contents['frames']):
            file_path = '{:0>4d}'.format(idx) +'.png'
            if not os.path.exists(os.path.join(source_path, 'input', file_path)):
                data_path = os.path.join(parent_dir, frame['file_path'].split('/')[-2])
                data_file_path = frame['file_path'].split('/')[-1]
                os.system("cp {} {}".format(os.path.join(data_path, 'input', data_file_path), os.path.join(source_path, 'input', file_path)))
            # NeRF 'rot_mat' is a camera-to-world transform
            c2w = np.array(frame['transform_matrix'])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            qw, qx, qy, qz = mat2quat(w2c[:3, :3])
            tx, ty, tz = w2c[:3, 3]
            f.write('{} {} {} {} {} {} {} {} 1 {}\n'.format(idx+1, qw, qx, qy, qz, tx, ty, tz, file_path))
            f.write('\n')  # make sure every other line is empty

    # create empty points3D.txt under sparse_path
    points3D_path = os.path.join(sparse_path, 'points3D.txt')
    with open(points3D_path, 'w') as f:
        f.write('')
    
    print(f"Saved {sparse_path}")