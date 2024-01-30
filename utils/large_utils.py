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

    aabb = torch.tensor(aabb, dtype=torch.float32)
    block_id_z = block_id // (block_dim[0] * block_dim[1]);
    block_id_y = (block_id % (block_dim[0] * block_dim[1])) // block_dim[0];
    block_id_x = (block_id % (block_dim[0] * block_dim[1])) % block_dim[0];

    xyz = contract_to_unisphere(torch.tensor(xyz_org), aabb, ord=torch.inf)
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