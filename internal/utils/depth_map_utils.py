from typing import Union, Optional
import torch
import numpy as np
import cv2


def build_homogenous_coordinates(
        fx: Union[float, torch.Tensor],
        fy: Union[float, torch.Tensor],
        cx: Union[float, torch.Tensor],
        cy: Union[float, torch.Tensor],
        width: int,
        height: int,
        dtype=torch.float,
        device=None,
):
    # build intrinsics matrix
    K = torch.eye(3, device=device, dtype=dtype)
    K[0, 2] = cx
    K[1, 2] = cy
    K[0, 0] = fx
    K[1, 1] = fy

    # build pixel coordination
    u, v = torch.meshgrid(
        torch.arange(width, device=device, dtype=dtype),
        torch.arange(height, device=device, dtype=dtype),
        indexing="xy",
    )
    u_flatten = u.flatten()
    v_flatten = v.flatten()
    p2d = torch.stack([u_flatten, v_flatten, torch.ones_like(u_flatten)], dim=-1)
    homogenous_coordinates = torch.matmul(p2d, torch.linalg.inv(K).T)

    return homogenous_coordinates


def depth_map_to_points(
        depth_map: torch.Tensor,  # [H, W]
        fx: Union[float, torch.Tensor],
        fy: Union[float, torch.Tensor],
        cx: Union[float, torch.Tensor],
        cy: Union[float, torch.Tensor],
        c2w: torch.Tensor,  # [4, 4]
        valid_pixel_mask: torch.Tensor = None,  # [H, W]
):
    homogenous_coordinates = build_homogenous_coordinates(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=depth_map.shape[1],
        height=depth_map.shape[0],
        dtype=depth_map.dtype,
        device=depth_map.device,
    )

    depths = depth_map.flatten()
    if valid_pixel_mask is not None:
        valid_pixel_mask = valid_pixel_mask.flatten()
        depths = depths[valid_pixel_mask]
        homogenous_coordinates = homogenous_coordinates[valid_pixel_mask]

    points_3d_in_camera = homogenous_coordinates * depths[:, None]
    points_3d_in_world = torch.matmul(points_3d_in_camera, c2w[:3, :3].T) + c2w[:3, 3]

    return points_3d_in_world


def depth_map_to_colored_points_with_down_sample(
        depth_map: torch.Tensor,  # [H, W]
        rgb: Optional[np.ndarray],  # [H, W, 3]
        fx: Union[float, torch.Tensor],
        fy: Union[float, torch.Tensor],
        cx: Union[float, torch.Tensor],
        cy: Union[float, torch.Tensor],
        c2w: torch.Tensor,  # [4, 4]
        down_sample_factor: int = 1,
        valid_pixel_mask: torch.Tensor = None,  # [H, W]
):
    if down_sample_factor > 1:
        fx = fx / down_sample_factor
        fy = fy / down_sample_factor
        cx = cx / down_sample_factor
        cy = cy / down_sample_factor
        height, width = depth_map.shape
        height = height // down_sample_factor
        width = width // down_sample_factor

        depth_map = torch.nn.functional.interpolate(
            depth_map[None, None, ...],
            size=(height, width),
            mode="nearest",
        )[0, 0]
        if valid_pixel_mask is not None:
            valid_pixel_mask = torch.nn.functional.interpolate(
                valid_pixel_mask[None, None, ...].to(torch.uint8),
                size=(height, width),
                mode="nearest",
            )[0, 0] == 1

    if rgb is not None:
        if down_sample_factor > 1:
            rgb = cv2.resize(rgb, dsize=(width, height))
            assert rgb.shape[:2] == depth_map.shape[:2]
        return depth_map_to_colored_points(
            depth_map=depth_map,
            rgb=rgb,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            c2w=c2w,
            valid_pixel_mask=valid_pixel_mask,
        )
    return depth_map_to_points(
        depth_map,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        c2w=c2w,
        valid_pixel_mask=valid_pixel_mask,
    )


def depth_map_to_colored_points(
        depth_map: torch.Tensor,  # [H, W]
        rgb: np.ndarray,  # [H, W, 3]
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        c2w: torch.Tensor,  # [4, 4]
        valid_pixel_mask: torch.Tensor = None,  # [H, W]
):
    points_3d = depth_map_to_points(
        depth_map=depth_map,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        c2w=c2w,
        valid_pixel_mask=valid_pixel_mask,
    )

    if valid_pixel_mask is None:
        rgb = rgb.reshape((-1, 3))
    else:
        rgb = rgb[valid_pixel_mask.cpu().numpy()]

    return points_3d, rgb


def enable_exr():
    import os
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


def read_depth(path: str) -> np.ndarray:
    return cv2.imread(
        path,
        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
    )[..., 0]


def read_depth_as_tensor(path: str) -> torch.Tensor:
    return torch.from_numpy(read_depth(path))


def test_build_homogenous_coordinates():
    w, h = 1920, 1080
    cx, cy = w / 2, h / 2

    coordinates = build_homogenous_coordinates(
        fx=1.,
        fy=1.,
        cx=cx,
        cy=cy,
        width=w,
        height=h,
        dtype=torch.float64,
    ).reshape((h, w, -1))
    # x right
    assert torch.all(coordinates[..., 0] == torch.arange(w).unsqueeze(0) - cx)
    # y down
    assert torch.all(coordinates[..., 1] == torch.arange(h).unsqueeze(1) - cy)
    # z front
    assert torch.all(coordinates[..., -1] == 1.)
