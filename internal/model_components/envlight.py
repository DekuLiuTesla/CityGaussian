import torch
import nvdiffrast.torch as dr


class EnvLight(torch.nn.Module):
    def __init__(self, resolution=1024):
        super().__init__()
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True),
        )

    def forward(self, l):
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape)
        l = l.contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])

        light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        light = light.view(*prefix, -1)

        return light
