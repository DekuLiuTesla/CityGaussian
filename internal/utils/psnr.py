import torch

def color_correct(img, ref, num_iters=5, eps=0.5 / 255):
    """Warp `img` to match the colors in `ref_img`."""
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f'img\'s {img.shape[-1]} and ref\'s {ref.shape[-1]} channels must match'
        )
    
    num_channels = img.shape[-1]
    img_mat = img.reshape(-1, num_channels)
    ref_mat = ref.reshape(-1, num_channels)
    
    def is_unclipped(z):
        return (z >= eps) & (z <= (1 - eps))  # z ∈ [eps, 1-eps].
    
    mask0 = is_unclipped(img_mat)
    
    for _ in range(num_iters):
        a_mat = []
        for c in range(num_channels):
            a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])  # Quadratic term.
        a_mat.append(img_mat)  # Linear term.
        a_mat.append(torch.ones_like(img_mat[:, :1]))  # Bias term.
        a_mat = torch.cat(a_mat, dim=-1)
        
        warp = []
        for c in range(num_channels):
            b = ref_mat[:, c]
            mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
            ma_mat = torch.where(mask[:, None], a_mat, torch.zeros_like(a_mat))
            mb = torch.where(mask, b, torch.zeros_like(b))
            w = torch.linalg.lstsq(ma_mat, mb, rcond=-1).solution  # Solve the linear system.
            assert torch.all(torch.isfinite(w))
            warp.append(w.squeeze())
        
        warp = torch.stack(warp, dim=-1)
        img_mat = torch.clamp(torch.matmul(a_mat, warp), 0, 1)
    
    corrected_img = img_mat.reshape(img.shape)
    return corrected_img