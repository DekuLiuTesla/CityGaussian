import os
import sys
import json
import numpy as np
from argparse import ArgumentParser

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Convert transforms.json from UE to 3DGS acceptable format")
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("--train", action="store_true", help="Train or not.")
    args = parser.parse_args(sys.argv[1:])

    transforms_path_ref = args.ref_path
    source_path = args.source_path
    transforms_path = os.path.join(source_path, 'input', 'transforms.json')
    json_name = 'transforms_train.json' if args.train else 'transforms_test.json'
    transforms_path_out = os.path.join(os.path.dirname(args.source_path), json_name)
    out_contents = {}

    with open(transforms_path_ref) as json_file:
        ref_contents = json.load(json_file)
    with open(transforms_path, 'r', encoding='utf-16') as json_file:
        contents = json.load(json_file)
        contents['Sensor_Width'] = 23.76  # mm
        contents['Sensor_Height'] = 13.365  # mm
        contents['Current_Focal_Length'] = 11.8  # mm
        contents['camera_angle_x'] = 90.387123 * np.pi / 180  # degree to rad

    # initialize out_contents with the same keys as ref_contents
    for key in ref_contents.keys():
        if key == 'frames':
            out_contents[key] = []
        else:
            out_contents[key] = 0

    out_contents['camera_angle_x'] = contents['camera_angle_x']
    out_contents['cx'] = contents['cx']
    out_contents['cy'] = contents['cy']
    out_contents['w'] = contents['w']
    out_contents['h'] = contents['h']
    out_contents['fl_x'] = contents['w'] / contents['Sensor_Width'] * contents['Current_Focal_Length']
    out_contents['fl_y'] = contents['h'] / contents['Sensor_Height'] * contents['Current_Focal_Length']
    camera_angle_x = np.arctan(0.5 * contents['Sensor_Width'] / contents['Current_Focal_Length']) * 2

    frames = contents["Frames"]
    for idx, frame in enumerate(frames):
        c2w = frame['transform_matrix']
        c2w = c2w.replace('[', '').replace(']', '').replace(',', '').split(' ')
        c2w = [float(i) for i in c2w if i != '']
        c2w = np.array(c2w).reshape(4, 4).T
        c2w[:3, 3] = c2w[:3, 3] / 100  # cm to m

        c2w_gs = np.zeros((4, 4))
        c2w_gs[:, 3] = c2w[:, 3]
        c2w_gs[:3, 0] = c2w[:3, 1]
        c2w_gs[:3, 1] = c2w[:3, 2]
        c2w_gs[:3, 2] = -c2w[:3, 0]
        file_path = os.path.join(os.path.split(args.source_path)[-1], 'input', os.path.split(frame['file_path'])[-1])
        file_path = file_path.replace('jpg', 'png')
        out_contents['frames'].append({'file_path': file_path, 'transform_matrix': c2w_gs.tolist()})

    with open(transforms_path_out, 'w') as json_file:
        json.dump(out_contents, json_file, indent=4)
    
    print(f"Saved {transforms_path_out}")