import numpy as np
from PIL import Image
import torch
import torchvision


def read_image_as_tensor(path: str) -> torch.Tensor:
    return torchvision.io.read_image(path)  # [C, H, W]


def save_tensor_image(path: str, image: torch.Tensor):
    torchvision.utils.save_image(image, path)


def read_image(path: str) -> np.ndarray:
    pil_image = Image.open(path)
    return np.array(pil_image)  # [H, W, C]


def save_image(path: str, image: np.ndarray):
    pil_image = Image.fromarray(image)
    pil_image.save(path, subsampling=0, quality=100)


def rgba2rgb(rgba: np.ndarray, background: np.ndarray = None) -> np.ndarray:
    if background is None:
        background = np.array([0, 0, 0], dtype=np.float64)
    normalized_rgba = rgba / 255.
    rgb = normalized_rgba[:, :, :3] * normalized_rgba[:, :, 3:4] + background * (1 - normalized_rgba[:, :, 3:4])
    return np.asarray(rgb * 255, dtype=np.uint8)
