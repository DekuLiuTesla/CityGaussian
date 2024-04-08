import os
import sys
import json
import numpy as np
from argparse import ArgumentParser
from transforms3d.quaternions import mat2quat

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Convert transforms_train.json to txt required by COLMAP")
    parser.add_argument("--transforms_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, required=True)
    args = parser.parse_args(sys.argv[1:])

    transforms_path = args.transforms_path
    target_path = args.target_path
    mode = 'train' if 'train' in transforms_path else 'test'

    sparse_path = os.path.join(target_path, 'fuse', mode, 'sparse')
    input_path = os.path.join(target_path, 'fuse', mode, 'input')
    if not os.path.exists(sparse_path):
        os.mkdir(sparse_path)
    if not os.path.exists(input_path):
        os.mkdir(input_path)
    with open(transforms_path) as json_file:
        contents = json.load(json_file)

    cameras = []
    for idx, frame in enumerate(contents['frames']):
        camera = {
            'camera_angle_x': frame['camera_angle_x'],
            'fl_x': frame['fl_x'],
            'fl_y': frame['fl_y'],
            'cx': frame['cx'],
            'cy': frame['cy'],
            'w': frame['w'],
            'h': frame['h']
        }
        if camera not in cameras:
            cameras.append(camera)

    # create cameras.txt under sparse_path
    cameras_path = os.path.join(sparse_path, 'cameras.txt')
    with open(cameras_path, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write(f'# Number of cameras: {len(cameras)}\n')
        for idx, intrinsic in enumerate(cameras):
            f.write('{} PINHOLE {} {} {} {} {} {}\n'.format(idx+1, intrinsic['w'], intrinsic['h'], intrinsic['fl_x'], intrinsic['fl_y'], 
                                                    intrinsic['cx'], intrinsic['cy']))
    
    # create images.txt under sparse_path
    images_path = os.path.join(sparse_path, 'images.txt')
    frames = []
    for idx, frame in enumerate(contents['frames']):
        camera = {
            'camera_angle_x': frame['camera_angle_x'],
            'fl_x': frame['fl_x'],
            'fl_y': frame['fl_y'],
            'cx': frame['cx'],
            'cy': frame['cy'],
            'w': frame['w'],
            'h': frame['h']
        }
        cam_id = cameras.index(camera) + 1
        frame['camera_id'] = cam_id
        frames.append(frame)

    with open(images_path, 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write('# Number of images: {}\n'.format(len(frames)))
        for idx, frame in enumerate(frames):
            if 'street' in frame['file_path']:
                file_path = os.path.join("../", os.path.dirname(frame['file_path']), 'input', os.path.basename(frame['file_path']))
            else:
                file_path = os.path.join("../", os.path.dirname(frame['file_path']), mode, 'input', os.path.basename(frame['file_path']))
            data_path = os.path.join(target_path, file_path.replace('../', ''))
            image_path = os.path.join(input_path, "%04d.png" % idx)
            
            if not os.path.exists(data_path):
                continue

            # copy image from data_path to input_path
            os.system(f"cp {data_path} {image_path}")

            # NeRF 'rot_mat' is a camera-to-world transform
            c2w = np.array(frame['transform_matrix'])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            qw, qx, qy, qz = mat2quat(w2c[:3, :3])
            tx, ty, tz = w2c[:3, 3]
            f.write('{} {} {} {} {} {} {} {} {} {}\n'.format(idx+1, qw, qx, qy, qz, tx, ty, tz, frame['camera_id'], "%04d.png" % idx))
            f.write('\n')  # make sure every other line is empty

    # create empty points3D.txt under sparse_path
    points3D_path = os.path.join(sparse_path, 'points3D.txt')
    with open(points3D_path, 'w') as f:
        f.write('')
    
    print(f"Saved {sparse_path}")