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
# import jax
# import jax.numpy as np
import numpy as np

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
    a_mat.append(np.ones_like(img_mat[:, :1]))  # Bias term.
    a_mat = np.concatenate(a_mat, axis=-1)
    warp = []
    for c in range(num_channels):
      # Construct the right hand side of a linear system containing each color
      # of `ref`.
      b = ref_mat[:, c]
      # Ignore rows of the linear system that were saturated in the input or are
      # saturated in the current corrected color estimate.
      mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
      ma_mat = np.where(mask[:, None], a_mat, 0)
      mb = np.where(mask, b, 0)
      # Solve the linear system. We're using the np.lstsq instead of np because
      # it's significantly more stable in this case, for some reason.
      w = np.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
      assert np.all(np.isfinite(w))
      warp.append(w)
    warp = np.stack(warp, axis=-1)
    # Apply the warp to update img_mat.
    img_mat = np.clip(
        np.matmul(a_mat, warp), 0, 1)
  corrected_img = np.reshape(img_mat, img.shape)
  return corrected_img