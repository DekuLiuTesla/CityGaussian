import torch

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
    block_id = torch.zeros(xyz.shape[0], dtype=torch.int32, device=xyz.device)
    block_id_x = torch.floor(xyz[:, 0] * block_dim[0]).clamp(0, block_dim[0] - 1).int()
    block_id_y = torch.floor(xyz[:, 1] * block_dim[1]).clamp(0, block_dim[1] - 1).int()
    block_id_z = torch.floor(xyz[:, 2] * block_dim[2]).clamp(0, block_dim[2] - 1).int()

    block_id = block_id_z * block_dim[0] * block_dim[1] + block_id_y * block_dim[0] + block_id_x

    return block_id

def in_frustum(cam_center, full_proj_transform, world_view_transform, cell_corners, aabb, block_dim):
    num_cell = cell_corners.shape[0]
    device = cell_corners.device

    cell_corners = torch.cat([cell_corners, torch.ones_like(cell_corners[..., [0]])], dim=-1)
    full_proj_transform = full_proj_transform.repeat(num_cell, 1, 1)
    viewmatrix = world_view_transform.repeat(num_cell, 1, 1)
    cell_corners_screen = cell_corners.bmm(full_proj_transform)
    cell_corners_screen = cell_corners_screen / cell_corners_screen[..., [-1]]
    cell_corners_screen = cell_corners_screen[..., :-1]

    cell_corners_cam = cell_corners.bmm(viewmatrix)
    mask = (cell_corners_cam[..., 2] > 0.2)

    cell_corners_screen_min = torch.zeros((num_cell, 3), dtype=torch.float32, device=device)
    cell_corners_screen_max = torch.zeros((num_cell, 3), dtype=torch.float32, device=device)

    for i in range(num_cell):
        if mask[i].sum() > 0:
            cell_corners_screen_min[i] = cell_corners_screen[i][mask[i]].min(dim=0).values
            cell_corners_screen_max[i] = cell_corners_screen[i][mask[i]].max(dim=0).values

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

    cam_center_id = which_block(cam_center[None, :], aabb, block_dim)[0]
    mask[cam_center_id] = True

    return mask
