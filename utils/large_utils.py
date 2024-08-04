import torch
import numpy as np
from utils.camera_utils import loadCam_woImage

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    ord: float = 2,
    eps: float = 1e-6,
    derivative: bool = False,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x

def block_filtering(block_id, xyz_org, aabb, block_dim, scale=1.0, mask_only=True):

    if len(aabb) == 4:
        aabb = [aabb[0], aabb[1], xyz_org[:, -1].min(), 
                aabb[2], aabb[3], xyz_org[:, -1].max()]
    elif len(aabb) == 6:
        aabb = aabb
    else:
        assert False, "Unknown aabb format!"

    xyz_tensor = torch.tensor(xyz_org)
    aabb = torch.tensor(aabb, dtype=torch.float32, device=xyz_tensor.device)
    block_id_z = block_id // (block_dim[0] * block_dim[1]);
    block_id_y = (block_id % (block_dim[0] * block_dim[1])) // block_dim[0];
    block_id_x = (block_id % (block_dim[0] * block_dim[1])) % block_dim[0];

    xyz = contract_to_unisphere(xyz_tensor, aabb, ord=torch.inf)
    min_x, max_x = float(block_id_x) / block_dim[0], float(block_id_x + 1) / block_dim[0]
    min_y, max_y = float(block_id_y) / block_dim[1], float(block_id_y + 1) / block_dim[1]
    min_z, max_z = float(block_id_z) / block_dim[2], float(block_id_z + 1) / block_dim[2]

    delta_x = (max_x - min_x) * (scale - 1.0)
    delta_y = (max_y - min_y) * (scale - 1.0)
    delta_z = (max_z - min_z) * (scale - 1.0)

    min_x -= delta_x / 2
    max_x += delta_x / 2
    min_y -= delta_y / 2
    max_y += delta_y / 2
    min_z -= delta_z / 2
    
    block_mask = (xyz[:, 0] >= min_x) & (xyz[:, 0] < max_x)  \
                    & (xyz[:, 1] >= min_y) & (xyz[:, 1] < max_y) \
                    & (xyz[:, 2] >= min_z) & (xyz[:, 2] < max_z)

    if mask_only:
        return block_mask
    else:
        return mask_only, xyz, [min_x, max_x, min_y, max_y, min_z, max_z]

def which_block(xyz_org, aabb, block_dim):

    if len(aabb) == 4:
        aabb = [aabb[0], aabb[1], xyz_org[:, -1].min(), 
                aabb[2], aabb[3], xyz_org[:, -1].max()]
    elif len(aabb) == 6:
        aabb = aabb
    else:
        assert False, "Unknown aabb format!"

    xyz_tensor = torch.tensor(xyz_org)
    aabb = torch.tensor(aabb, dtype=torch.float32, device=xyz_tensor.device)

    xyz = contract_to_unisphere(xyz_tensor, aabb, ord=torch.inf)

    block_id_x = torch.floor((xyz[:, 0] * block_dim[0]).clamp(0, block_dim[0] - 1)).long()
    block_id_y = torch.floor((xyz[:, 1] * block_dim[1]).clamp(0, block_dim[1] - 1)).long()
    block_id_z = torch.floor((xyz[:, 2] * block_dim[2]).clamp(0, block_dim[2] - 1)).long()

    block_id = block_id_z * block_dim[0] * block_dim[1] + block_id_y * block_dim[0] + block_id_x

    return block_id

def in_frustum(viewpoint_cam, cell_corners, aabb, block_dim):
    num_cell = cell_corners.shape[0]
    device = cell_corners.device

    cell_corners = torch.cat([cell_corners, torch.ones_like(cell_corners[..., [0]])], dim=-1)
    full_proj_transform = viewpoint_cam.full_proj_transform.repeat(num_cell, 1, 1)
    viewmatrix = viewpoint_cam.world_view_transform.repeat(num_cell, 1, 1)
    cell_corners_screen = cell_corners.bmm(full_proj_transform)
    cell_corners_screen = cell_corners_screen / cell_corners_screen[..., [-1]]
    cell_corners_screen = cell_corners_screen[..., :-1].reshape(-1, 3)

    cell_corners_cam = cell_corners.bmm(viewmatrix)
    dist = torch.norm(cell_corners_cam[:, :, :3], dim=-1)
    dist_min = torch.min(dist, dim=-1)[0]
    cam_center_id = torch.argmin(dist_min)
    mask = (cell_corners_cam[..., 2] > 0.2)

    mask_ = mask.reshape(-1)
    cell_corners_screen_ = cell_corners_screen.clone().reshape(-1, 3)
    cell_corners_screen_[~mask_] = torch.inf
    cell_corners_screen_min = cell_corners_screen_.reshape(num_cell, -1, 3).min(dim=1).values
    cell_corners_screen_min[cell_corners_screen_min==torch.inf] = 0.0

    cell_corners_screen_ = cell_corners_screen.clone().reshape(-1, 3)
    cell_corners_screen_[~mask_] = -torch.inf
    cell_corners_screen_max = cell_corners_screen_.reshape(num_cell, -1, 3).max(dim=1).values
    cell_corners_screen_max[cell_corners_screen_max==-torch.inf] = 0.0

    box_a = torch.cat([cell_corners_screen_min[:, :2], cell_corners_screen_max[:, :2]], dim=1)
    box_b = torch.tensor([[-1, -1, 1, 1]], dtype=torch.float32, device=device)
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                    box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                    box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    mask = (inter[:, 0, 0] * inter[:, 0, 1]) > 0
    mask[cam_center_id] = True
    
    return mask, dist_min[mask]

def get_default_aabb(args, cameras, xyz_org, scale=1.0):
    
    torch.cuda.empty_cache()
    c2ws = np.array([np.linalg.inv(np.asarray((loadCam_woImage(args, idx, cam, scale).world_view_transform.T).cpu().numpy())) for idx, cam in enumerate(cameras)])
    poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
    center = (focus_point_fn(poses))
    radius = torch.tensor(np.median(np.abs(c2ws[:,:3,3] - center), axis=0), device=xyz_org.device)
    center = torch.from_numpy(center).float().to(xyz_org.device)
    if radius.min() / radius.max() < 0.02:
        # If the radius is too small, we don't contract in this dimension
        radius[torch.argmin(radius)] = 0.5 * (xyz_org[:, torch.argmin(radius)].max() - xyz_org[:, torch.argmin(radius)].min())
    aabb = torch.zeros(6, device=xyz_org.device)
    aabb[:3] = center - radius
    aabb[3:] = center + radius

    return aabb