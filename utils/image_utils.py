#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import jax
import jax.numpy as jnp
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def color_correct(img, ref, num_iters=5, eps=0.5 / 255):
  """Warp `img` to match the colors in `ref_img`."""
  if img.shape[-1] != ref.shape[-1]:
    raise ValueError(
        f'img\'s {img.shape[-1]} and ref\'s {ref.shape[-1]} channels must match'
    )
  num_channels = img.shape[-1]
  img_mat = img.reshape([-1, num_channels])
  ref_mat = ref.reshape([-1, num_channels])
  is_unclipped = lambda z: (z >= eps) & (z <= (1 - eps))  # z \in [eps, 1-eps].
  mask0 = is_unclipped(img_mat)
  # Because the set of saturated pixels may change after solving for a
  # transformation, we repeatedly solve a system `num_iters` times and update
  # our estimate of which pixels are saturated.
  for _ in range(num_iters):
    # Construct the left hand side of a linear system that contains a quadratic
    # expansion of each pixel of `img`.
    a_mat = []
    for c in range(num_channels):
      a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])  # Quadratic term.
    a_mat.append(img_mat)  # Linear term.
    a_mat.append(jnp.ones_like(img_mat[:, :1]))  # Bias term.
    a_mat = jnp.concatenate(a_mat, axis=-1)
    warp = []
    for c in range(num_channels):
      # Construct the right hand side of a linear system containing each color
      # of `ref`.
      b = ref_mat[:, c]
      # Ignore rows of the linear system that were saturated in the input or are
      # saturated in the current corrected color estimate.
      mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
      ma_mat = jnp.where(mask[:, None], a_mat, 0)
      mb = jnp.where(mask, b, 0)
      # Solve the linear system. We're using the np.lstsq instead of jnp because
      # it's significantly more stable in this case, for some reason.
      w = np.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
      assert jnp.all(jnp.isfinite(w))
      warp.append(w)
    warp = jnp.stack(warp, axis=-1)
    # Apply the warp to update img_mat.
    img_mat = jnp.clip(
        jnp.matmul(a_mat, warp, precision=jax.lax.Precision.HIGHEST), 0, 1)
  corrected_img = jnp.reshape(img_mat, img.shape)
  return corrected_img