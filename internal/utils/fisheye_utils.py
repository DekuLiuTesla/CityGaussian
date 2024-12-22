import cv2
import numpy as np
import xml.etree.ElementTree as ET
from defisheye import Defisheye
from numpy import ndarray, hypot, arctan, pi, sin, cos, tan, sqrt, arange, meshgrid

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

def project_points_to_image(points, c, d, e, cx, cy, world2cam):
    # Normalize the points
    norm = np.linalg.norm(points[:, :2], axis=1)[:, None]
    theta = np.arctan2(-points[:, [2]], norm)
    
    # Apply the world2cam polynomial
    rho = 0
    theta_i = 1
    for coeff in world2cam:
        rho += coeff * theta_i
        theta_i *= theta
    
    # Compute image coordinates
    x = points[:, [0]] * rho / norm
    y = points[:, [1]] * rho / norm
    x_img = x * c + y * d + cx
    y_img = x * e + y + cy
    
    return np.concatenate([x_img, y_img, rho], axis=1)

class DeNavfisheye(Defisheye):
    def __init__(self, infile, **kwargs):
        vkwargs = {"fov": 180,
                   "pfov": 120,
                   "xcenter": None,
                   "ycenter": None,
                   "radius": None,
                   "pad": 0,
                   "c": 0,
                   "d": 0,
                   "e": 0,
                   "cam2world": list,
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
    
    def _map(self, i, j, ofoc, angle):

        xd = i - (self._width - 1) // 2
        yd = j - (self._height - 1) // 2

        angle_ = angle * np.pi / 180
        coords = np.stack([xd, yd, ofoc * np.ones_like(xd)], axis=-1)
        rot_mat = np.array([[cos(angle_), 0, sin(angle_)], [0, 1, 0], [-sin(angle_), 0, cos(angle_)]])
        coords = np.dot(coords, rot_mat)
        xd, yd, zd = coords[..., 0], coords[..., 1], coords[..., 2]

        rd = hypot(xd, yd)
        phiang = arctan(-zd / rd);
        
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

        rdmask[rdmask] = (xs[rdmask] >= 0) & (xs[rdmask] < self._image.shape[1]) & \
                         (ys[rdmask] >= 0) & (ys[rdmask] < self._image.shape[0]) 

        xs[~rdmask] = 0
        ys[~rdmask] = 0

        xs = xs.astype(int)
        ys = ys.astype(int)
        return xs, ys
    
    def convert(self, outfile=None, angle=0.0):
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

        xs, ys, = self._map(i, j, ofoc, angle)
        img = np.zeros((self._height, self._width, 3), dtype=self._image.dtype)

        img[j, i, :] = self._image[ys, xs, :]
        if outfile is not None:
            cv2.imwrite(outfile, img)
        return img, ofoc