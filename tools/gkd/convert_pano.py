import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out 

class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        #cp = self._img.copy()  
        #w = self._width
        #self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        #self._img[:, w/8:, :] = cp[:, :7*w/8, :]
    

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
        K_inv = np.linalg.inv(K)
        
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz) 
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Convert each panorama image to 6 perspective images")
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, default=None)
    parser.add_argument("--image_prefix", type=str, default='')
    parser.add_argument("--FoV", type=int, default=90)
    parser.add_argument("--angle_interval", type=int, default=60)
    parser.add_argument("--image_width", type=int, default=1080)
    parser.add_argument("--image_height", type=int, default=1080)
    args = parser.parse_args(sys.argv[1:])

    source_path = args.source_path
    if args.target_path is None:
        target_path = os.path.join(source_path, "input")
    else:
        target_path = args.target_path
    
    image_prefix = args.image_prefix
    if image_prefix != '':
        image_prefix = image_prefix + '_'
    FoV = args.FoV
    angle_interval = args.angle_interval
    image_width = args.image_width
    image_height = args.image_height

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    pano_list = os.listdir(os.path.join(source_path, "pano"))
    for pano_name in tqdm(pano_list):
        if pano_name.endswith(".list") or pano_name.endswith(".csv"):
            continue
        pano_path = os.path.join(source_path, "pano", pano_name)
        equ = Equirectangular(pano_path)
        
        for yaw in range(0, 360, angle_interval):
            persp_img = equ.GetPerspective(FoV, yaw, 0, image_height, image_width)
            suffix = os.path.splitext(pano_name)[-1]
            persp_name = pano_name.replace(suffix, "_{}".format(yaw) + suffix)
            persp_path = os.path.join(target_path, image_prefix+persp_name)
            cv2.imwrite(persp_path, persp_img)
        
    print("Done!")

    