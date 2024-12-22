import os
import sys
sys.path.append(".")
import cv2
import numpy as np
import json
import torch
import xml.etree.ElementTree as ET
from tqdm import tqdm
from argparse import ArgumentParser
from defisheye import Defisheye
from transforms3d.quaternions import mat2quat
from internal.utils.gaussian_projection import build_rotation_matrix
from numpy import ndarray, hypot, arctan, pi, sin, tan, sqrt, arange, meshgrid

def parse_ocam_model(xml_file, cam_idx=0):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    ocam_model = root.find(f".//CameraModel[SensorName='cam{cam_idx}']/OCamModel")
    c = float(ocam_model.find('c').text)
    d = float(ocam_model.find('d').text)
    e = float(ocam_model.find('e').text)
    cx = float(ocam_model.find('cx').text)
    cy = float(ocam_model.find('cy').text)
    world2cam = [float(coeff.text) for coeff in ocam_model.findall('world2cam/coeff')]
    cam2world = [float(coeff.text) for coeff in ocam_model.findall('cam2world/coeff')]

    return c, d, e, cx, cy, world2cam, cam2world

class DeNavfisheye(Defisheye):
    def __init__(self, infile, **kwargs):
        vkwargs = {"fov": 180,
                   "pfov": 120,
                   "xcenter": None,
                   "ycenter": None,
                   "radius": None,
                   "pad": 0,
                   "angle": 0,
                   "c": 0,
                   "d": 0,
                   "e": 0,
                   "world2cam": list,
                   "format": "fullframe"
                   }
        self._start_att(vkwargs, kwargs)

        if type(infile) == str:
            _image = cv2.imread(infile)
        elif type(infile) == ndarray:
            _image = infile
        else:
            raise Exception("Image format not recognized")

        if self._pad > 0:
            _image = cv2.copyMakeBorder(
                _image, self._pad, self._pad, self._pad, self._pad, cv2.BORDER_CONSTANT)

        width = _image.shape[1]
        height = _image.shape[0]
        xcenter = width // 2
        ycenter = height // 2

        dim = min(width, height)
        x0 = xcenter - dim // 2
        xf = xcenter + dim // 2
        y0 = ycenter - dim // 2
        yf = ycenter + dim // 2

        self._image = _image

        self._width = _image[y0:yf, x0:xf, :].shape[1]
        self._height = _image[y0:yf, x0:xf, :].shape[0]

        if self._xcenter is None:
            self._xcenter = (self._width - 1) // 2

        if self._ycenter is None:
            self._ycenter = (self._height - 1) // 2
    
    def _map(self, i, j, ofoc, dim):

        xd = i - (self._width - 1) // 2
        yd = j - (self._height - 1) // 2

        rd = hypot(xd, yd)
        phiang = arctan(-ofoc / rd)
        
        # Apply the world2cam polynomial
        rho = 0
        theta_i = 1
        for coeff in self._world2cam:
            rho += coeff * theta_i
            theta_i *= phiang

        rdmask = rd != 0
        xs = xd.copy()
        ys = yd.copy()

        x = (rho[rdmask] / rd[rdmask]) * xd[rdmask]
        y = (rho[rdmask] / rd[rdmask]) * yd[rdmask]
        xs[rdmask] = x * self._c + y * self._d + self._xcenter
        ys[rdmask] = x * self._e + y + self._ycenter

        xs[~rdmask] = 0
        ys[~rdmask] = 0

        xs = xs.astype(int)
        ys = ys.astype(int)
        return xs, ys
    
    def convert(self, outfile=None):
        if self._format == "circular":
            dim = min(self._width, self._height)
        elif self._format == "fullframe":
            dim = sqrt(self._width ** 2.0 + self._height ** 2.0)

        if self._radius is not None:
            dim = 2 * self._radius

        # compute output (perspective) focal length and its inverse from ofov
        # phi=fov/2; r=N/2
        # r/f=tan(phi);
        # f=r/tan(phi);
        # f= (N/2)/tan((fov/2)*(pi/180)) = N/(2*tan(fov*pi/360))

        ofoc = dim / (2 * tan(self._pfov * pi / 360))

        i = arange(self._width)
        j = arange(self._height)
        i, j = meshgrid(i, j)

        xs, ys, = self._map(i, j, ofoc, dim)
        img = np.zeros((self._height, self._width, 3), dtype=self._image.dtype)

        img[j, i, :] = self._image[ys, xs, :]
        if outfile is not None:
            cv2.imwrite(outfile, img)
        return img, ofoc

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Convert each panorama image to 6 perspective images")
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, default=None)
    parser.add_argument("--image_prefix", type=str, default='')
    parser.add_argument("--pfov", type=int, default=135)
    parser.add_argument("--format", type=str, default='circular')
    args = parser.parse_args(sys.argv[1:])

    source_path = args.source_path
    if args.target_path is None:
        target_path = os.path.join(source_path, "perspective")
    else:
        target_path = args.target_path
    
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    sparse_path = os.path.join(target_path, 'sparse')
    cameras_path = os.path.join(sparse_path, 'cameras.txt')
    images_path = os.path.join(sparse_path, 'images.txt')
    points3D_path = os.path.join(sparse_path, 'points3D.txt')
    if not os.path.exists(sparse_path):
        os.mkdir(sparse_path)
    input_path = os.path.join(target_path, 'input')
    if not os.path.exists(input_path):
        os.mkdir(input_path)

    intrinsics_path = os.path.join(source_path, "sensor_frame.xml")
    
    image_prefix = args.image_prefix
    if image_prefix != '':
        image_prefix = image_prefix + '-'
    pfov = args.pfov
    format = args.format

    cam_list = os.listdir(os.path.join(source_path, "cam"))
    with open(images_path, 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write('# Number of images: {}\n'.format(len(cam_list)))
        for idx, cam_name in enumerate(tqdm(cam_list)):
            image_idx = int(cam_name.split('-')[0])
            cam_idx = int(cam_name[9])
            cam_path = os.path.join(source_path, "cam", cam_name)
            info_path = os.path.join(source_path, "info", f"{image_idx:05d}-info.json")
            with open(info_path, "r") as info_f:
                info = json.loads(info_f.read())

            cam = info[f"cam{cam_idx}"]
            cam_p = cam['position']
            cam_q = cam['quaternion']
            c2w = np.eye(4)
            c2w[:3, :3] = np.array(build_rotation_matrix(torch.tensor(cam_q)[None, :]))[0]
            c2w[:3, 3] = np.array(cam_p)
            if cam_idx == 0:
                c2w[:, 0] *= -1  # cam 0
            else:
                c2w[:, 1] *= -1  # cam 1, 2, 3
            c2w = c2w[:, [1, 0, 2, 3]]
            w2c = np.linalg.inv(c2w)

            # save extrinsics and intrinsics in COLMAP format (Y down, Z forward)
            qw, qx, qy, qz = mat2quat(w2c[:3, :3])
            tx, ty, tz = w2c[:3, 3]
            f.write('{} {} {} {} {} {} {} {} 1 {}\n'.format(idx+1, qw, qx, qy, qz, tx, ty, tz, f"{image_prefix}{cam_name}"))
            f.write('\n')  # make sure every other line is empty
            
            # project image from fisheye to perspective view
            img = cv2.imread(cam_path)
            c, d, e, cx, cy, world2cam, cam2world = parse_ocam_model(intrinsics_path, cam_idx)
            obj = DeNavfisheye(img, format=format, pfov=pfov, c=c, d=d, e=e, 
                            xcenter=cx, ycenter=cy, world2cam=world2cam)
            new_image, f_p = obj.convert();
            cv2.imwrite(os.path.join(input_path, f"{image_prefix}{cam_name}"), new_image)
    
    # create cameras.txt under sparse_path
    with open(cameras_path, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write('# Number of cameras: 1\n')
        H, W = new_image.shape[:2]
        cx_p = (W - 1) // 2
        cy_p = (H - 1) // 2
        K = np.array([[f_p, 0, cx_p], [0, f_p, cy_p], [0, 0, 1]])
        f.write('1 PINHOLE {} {} {} {} {} {}\n'.format(W, H, f_p, f_p, cx_p, cy_p))
    
    # create empty points3D.txt under sparse_path
    with open(points3D_path, 'w') as f:
        f.write('')

    print("Done!")

    