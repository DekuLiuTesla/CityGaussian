"""
Most codes are copied from: https://github.com/Jumpat/SegAnyGAussians/blob/v2/train_contrastive_feature.py

# TODO: re-implemented by adding `metrics` class
"""

import os.path
from collections import namedtuple
from typing import Any, Optional, Union, Dict
from typing_extensions import Self

import lightning
import torch.nn
import pytorch3d.ops
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from internal.configs.segany_splatting import Optimization as OptimizationConfig
from internal.renderers.gsplat_contrastive_feature_renderer import GSplatContrastiveFeatureRenderer
# from internal.renderers.contrastive_feature_renderer import ContrastiveFeatureRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader


class SegAnySplatting(lightning.LightningModule):
    def __init__(
            self,
            initialize_from: str,
            optimization: OptimizationConfig,
            n_feature_dims: int = 32,
            scale_aware_dim: int = -1,
            ray_sample_rate: int = 0,
            num_sampled_rays: int = 1000,
            smooth_K: int = 16,
            smooth_dropout: float = 0.5,
            rfn: float = 1.,
            output_path: str = None,
            # TODO: remove unnecessary parameters
            sh_degree: int = -1,
            save_iterations: list = None,
            web_viewer: bool = False,
            save_val_output: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.n_feature_dims = n_feature_dims

        # avoid storing state_dict
        self.models = namedtuple("Models", ["gaussian"])

        # self.renderer = ContrastiveFeatureRenderer()
        self.renderer = GSplatContrastiveFeatureRenderer()

        self.feature_smooth_map = None

        # hyper parameters
        self.scale_aware_dim = scale_aware_dim
        self.ray_sample_rate = ray_sample_rate
        self.num_sampled_rays = num_sampled_rays
        self.smooth_K = smooth_K
        self.smooth_dropout = smooth_dropout
        self.rfn = rfn

    def setup(self, stage: str) -> None:
        super().setup(stage)

        initialize_from = self.hparams["initialize_from"]
        sh_degree = self.hparams["sh_degree"]

        gaussian_model, _ = GaussianModelLoader.search_and_load(initialize_from, self.device)
        gaussian_model.freeze()
        self.models.gaussian = gaussian_model

        self.setup_parameters(n_gaussians=gaussian_model.get_xyz.shape[0])

        if stage == "fit":
            if self.trainer.global_rank == 0:
                # save initialization model path
                with open(os.path.join(self.hparams["output_path"], "seganygs"), "w") as f:
                    f.write(self.hparams["initialize_from"])

            self.gather_scales()

    def setup_parameters(self, n_gaussians: int):
        # multiply with 1e-2 is the key to reduce noisy
        self.gaussian_semantic_features = torch.nn.Parameter(torch.randn(
            (n_gaussians, self.n_feature_dims),
            dtype=torch.float,
        ) * 1e-2, requires_grad=True)

        self.scale_gate = torch.nn.Sequential(
            torch.nn.Linear(1, self.n_feature_dims, bias=True),
            torch.nn.Sigmoid()
        )

        self.bg_color = torch.nn.Parameter(torch.zeros((self.n_feature_dims,), dtype=torch.float, device=self.device), requires_grad=False)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["feature_smooth_map"] = self.feature_smooth_map

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        self.feature_smooth_map = checkpoint["feature_smooth_map"]

    @staticmethod
    def get_quantile_func(scales: torch.Tensor, distribution="normal"):
        from sklearn.preprocessing import QuantileTransformer

        """
        Use 3D scale statistics to normalize scales -- use quantile transformer.
        """
        scales = scales.flatten()

        scales = scales.detach().cpu().numpy()

        # Calculate quantile transformer
        quantile_transformer = QuantileTransformer(output_distribution=distribution)
        quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

        def quantile_transformer_func(scales):
            # This function acts as a wrapper for QuantileTransformer.
            # QuantileTransformer expects a numpy array, while we have a torch tensor.
            scales = scales.reshape(-1, 1)
            return torch.Tensor(
                quantile_transformer.transform(scales.detach().cpu().numpy())
            ).to(scales.device)

        return quantile_transformer_func

    def gather_scales(self) -> None:
        scale_aware_dim = self.scale_aware_dim
        # gather scales
        print("Finding upper_bound_scale")
        all_scales = []

        from tqdm import tqdm
        for image in tqdm(self.trainer.datamodule.dataparser_outputs.train_set):
            scale_file_path = image[4][1]
            all_scales.append(torch.load(scale_file_path, map_location=self.device))
        all_scales = torch.cat(all_scales)

        self.upper_bound_scale = all_scales.max().item()
        print(f"upper_bound_scale={self.upper_bound_scale}")

        if scale_aware_dim <= 0 or scale_aware_dim >= 32:
            print("Using adaptive scale gate.")
            self.q_trans = self.get_quantile_func(all_scales, "uniform")
        else:
            self.q_trans = self.get_quantile_func(all_scales, "uniform")
            self.fixed_scale_gate = torch.tensor([[1 for j in range(32 - scale_aware_dim + i)] + [0 for k in range(scale_aware_dim - i)] for i in range(scale_aware_dim + 1)]).cuda()

    def mask_preprocess(self, sam_masks, mask_scales):
        ray_sample_rate = self.ray_sample_rate
        num_sampled_rays = self.num_sampled_rays

        with torch.no_grad():
            upper_bound_scale = self.upper_bound_scale

            # build Scale-Aware Pixel Identity Vector mentioned in the paper

            # sort scales
            mask_scales, sort_indices = torch.sort(mask_scales, descending=True)
            # reorder masks according to the sorted scales
            sam_masks = sam_masks.float()[sort_indices, :, :]

            # pick `num_sampled_scales` scales randomly
            num_sampled_scales = 8
            sampled_scale_index = torch.randperm(len(mask_scales))[:num_sampled_scales]

            tmp = torch.zeros(num_sampled_scales + 2)
            tmp[1:len(sampled_scale_index) + 1] = sampled_scale_index
            tmp[-1] = len(mask_scales) - 1
            tmp[0] = -1  # attach a bigger scale
            sampled_scale_index = tmp.long()  # = [-1, the indices of num_sampled_scales..., N_scales - 1]

            sampled_scales = mask_scales[sampled_scale_index]

            second_big_scale = mask_scales[mask_scales < upper_bound_scale].max()

            # pick some pixels to sample
            ray_sample_rate = ray_sample_rate if ray_sample_rate > 0 else num_sampled_rays / (sam_masks.shape[-1] * sam_masks.shape[-2])

            sampled_ray = torch.rand(sam_masks.shape[-2], sam_masks.shape[-1]).cuda() < ray_sample_rate
            non_mask_region = sam_masks.sum(dim=0) == 0

            # do not sample pixels that not being masked by any masks
            sampled_ray = torch.logical_and(sampled_ray, ~non_mask_region)

            # Appendix A.1: Re-weighting

            # H W
            # foreach mask, set the value of the masked pixel to the `number of total mask pixels`
            per_pixel_mask_size = sam_masks * sam_masks.sum(-1).sum(-1)[:, None, None]  # [N_masks, H, W]

            # `per_pixel_mask_size.sum(dim=0)` is the total number of the masked pixels of all masks, in [H, W]
            # `sam_masks.sum(dim=0)` is the total masked times of each pixel, in [H, W]
            # `per_pixel_mean_mask_size` is the average number of masked pixels, in [H, W]
            per_pixel_mean_mask_size = per_pixel_mask_size.sum(dim=0) / (sam_masks.sum(dim=0) + 1e-9)

            # pick those pixels selected to be sampled
            per_pixel_mean_mask_size = per_pixel_mean_mask_size[sampled_ray]  # [N_sampled_rays]

            # [1, N_sampled_rays] * [N_sampled_rays, 1] = [N_sampled_rays, N_sampled_rays]
            # the mean mask size multiplication results of all pixel paris
            pixel_to_pixel_mask_size = per_pixel_mean_mask_size.unsqueeze(0) * per_pixel_mean_mask_size.unsqueeze(1)
            # weights min-max normalization
            ptp_max_size = pixel_to_pixel_mask_size.max()
            pixel_to_pixel_mask_size[pixel_to_pixel_mask_size == 0] = 1e10
            per_pixel_weight = torch.clamp(ptp_max_size / pixel_to_pixel_mask_size, 1.0, None)  # smaller one has bigger weight
            # first normalize the weight to [0, 1], then enlarge to [1, 10]
            per_pixel_weight = (per_pixel_weight - per_pixel_weight.min()) / (per_pixel_weight.max() - per_pixel_weight.min()) * 9. + 1.

            sam_masks_sampled_ray = sam_masks[:, sampled_ray]  # [N_masks, N_sampled_rays]

            gt_corrs = []

            # set the first value of `sampled_scales` to a scale bigger than upper one a bit, which will trigger operation when `upper_bound=True` below
            sampled_scales[0] = upper_bound_scale + upper_bound_scale * torch.rand(1)[0]
            for idx, si in enumerate(sampled_scale_index):
                upper_bound = sampled_scales[idx] >= upper_bound_scale

                if si != len(mask_scales) - 1 and not upper_bound:  # not the last scale, and not upper_bound
                    # make it a little smaller, according to the diff to the later one
                    sampled_scales[idx] -= (sampled_scales[idx] - mask_scales[si + 1]) * torch.rand(1)[0]
                elif upper_bound:
                    # make it a little smaller, according to the diff to the second_big_scale
                    sampled_scales[idx] -= (sampled_scales[idx] - second_big_scale) * torch.rand(1)[0]
                else:
                    sampled_scales[idx] -= sampled_scales[idx] * torch.rand(1)[0]

                if not upper_bound:
                    gt_vec = torch.zeros_like(sam_masks_sampled_ray)  # V(s, p), [N_masks, N_sampled_rays (or pixels)]
                    gt_vec[:si + 1, :] = sam_masks_sampled_ray[:si + 1, :]  # add bigger and equal scale masks
                    for j in range(si, -1, -1):  # from si to 0
                        # `gt_vec[j + 1:, :].any(dim=0)`: in all smaller masks, is pixel masked?, [N_smaller_masks, N_sampled_pixels]
                        #    torch.logical_not() mark the unmasked pixel to True
                        # if the pixel is not masked in all smaller mask, and masked at current scale, mark it to True ("the i-th entry of V(s, p) equals to 1 only if ...")
                        gt_vec[j, :] = torch.logical_and(
                            torch.logical_not(gt_vec[j + 1:, :].any(dim=0)), gt_vec[j, :]
                        )
                    # add smaller scale masks ("when ... < s, the i-th entry of V(s, p) is set to ...")
                    gt_vec[si + 1:, :] = sam_masks_sampled_ray[si + 1:, :]
                else:
                    gt_vec = sam_masks_sampled_ray

                # gt_vec.T multiply with gt_vec
                # gt_vec is [N_masks (or N_scales), N_sampled_rays]
                gt_corr = torch.einsum('nh,nj->hj', gt_vec, gt_vec)  # for each pair of pixels, how many common masks do they have?
                gt_corr[gt_corr != 0] = 1  # mark those pairs having any common masks (or masked in the same mask; "(i.e., V(s, p1) · V(s, p2) > 0), they should have similar features at scale s")
                gt_corrs.append(gt_corr)

            # N_scale S C_clip
            # gt_clip_features = torch.stack(gt_clip_features, dim = 0)
            # N_scale S S
            gt_corrs = torch.stack(gt_corrs, dim=0)  # [N_sample_scales, N_sampled_rays, N_sampled_rays], marks indicating whether pixel paris have common masks
            # map the distribution of scales to Gaussian distribution
            sampled_scales = self.q_trans(sampled_scales).squeeze()
            sampled_scales = sampled_scales.squeeze()  # [N_sampled_scales]

        return sampled_ray, per_pixel_weight, gt_corrs, sampled_scales

    def get_smoothed_point_features(self, K=16, dropout=0.5):
        # Local Feature Smoothing

        if K <= 1:
            return self.gaussian_semantic_features

        assert dropout < 0 or int(K * dropout) >= 1

        with torch.no_grad():
            if self.feature_smooth_map is None or self.feature_smooth_map["K"] != K:
                xyz = self.models.gaussian.get_xyz
                nearest_k_idx = pytorch3d.ops.knn_points(
                    xyz.unsqueeze(0),
                    xyz.unsqueeze(0),
                    K=K,
                ).idx.squeeze()
                self.feature_smooth_map = {"K": K, "m": nearest_k_idx}

        normed_features = torch.nn.functional.normalize(self.gaussian_semantic_features, dim=-1, p=2)

        if dropout > 0 and dropout < 1:
            # discard some points randomly
            select_point = torch.randperm(K)[: int(K * dropout)]

            select_idx = self.feature_smooth_map["m"][:, select_point]  # [N_gaussians, N_selected_points]
            # `normed_features[select_idx, :]`: [N_gaussians, N_selected_points, N_semantic_feature_dims]
            ret = normed_features[select_idx, :].mean(dim=1)  # [N_gaussians, N_semantic_feature_dims]
        else:
            ret = normed_features[self.feature_smooth_map["m"], :].mean(dim=1)

        return ret

    def renderer_forward(self, camera, semantic_features):
        return self.renderer(
            viewpoint_camera=camera,
            pc=self.models.gaussian,
            bg_color=self.bg_color,
            semantic_features=semantic_features,
        )

    def get_processed_semantic_features(self):
        # dropout only apply during training
        smooth_dropout = self.smooth_dropout
        if self.training is False:
            smooth_dropout = -1
        semantic_features = self.get_smoothed_point_features(K=self.smooth_K, dropout=smooth_dropout)
        semantic_features = semantic_features / (semantic_features.norm(dim=-1, keepdim=True) + 1e-9)
        return semantic_features

    def forward(self, camera) -> Any:
        return self.renderer_forward(
            camera,
            self.get_processed_semantic_features(),
        )

    def forward_with_loss_calculation(self, camera, image_info, semantic):
        outputs = self(camera)
        rendered_features = outputs["render"]

        masks, scales = semantic
        scale_aware_dim = self.scale_aware_dim

        sampled_ray, per_pixel_weight, gt_corrs, sampled_scales = self.mask_preprocess(sam_masks=masks, mask_scales=scales)

        rendered_feature_norm = rendered_features.norm(dim=0, p=2).mean()
        rendered_feature_norm_reg = (1 - rendered_feature_norm) ** 2

        rendered_features = torch.nn.functional.interpolate(rendered_features.unsqueeze(0), masks.shape[-2:], mode='bilinear').squeeze(0)

        # 3.3.1. section of the paper: projects scale scalars to soft gate vectors
        # N_sampled_scales 32
        if scale_aware_dim <= 0 or scale_aware_dim >= 32:
            gates = self.scale_gate(sampled_scales.unsqueeze(-1))  # [N_scales, N_semantic_feature_dims]
        else:
            int_sampled_scales = ((1 - sampled_scales.squeeze()) * scale_aware_dim).long()
            gates = self.fixed_scale_gate[int_sampled_scales].detach()

        # N_sampled_scales C H W
        feature_with_scale = rendered_features.unsqueeze(0).repeat([sampled_scales.shape[0], 1, 1, 1])  # [N_sampled_scales, N_semantic_feature_dims (C), H, W]
        # multiply rendered vectors with gate vectors
        feature_with_scale = feature_with_scale * gates.unsqueeze(-1).unsqueeze(-1)  # eq. 5 of the paper, [N_sampled_scales, N_semantic_feature_dims (C), H, W]

        # pick selected rays (pixels)
        sampled_feature_with_scale = feature_with_scale[:, :, sampled_ray]  # [N_sampled_scales, N_semantic_feature_dims (C), N_sampled_rays]

        scale_conditioned_features_sam = sampled_feature_with_scale.permute([0, 2, 1])  # [N_sampled_scales, N_sampled_rays, N_semantic_feature_dims (C)]

        # normalize the semantic features
        scale_conditioned_features_sam = torch.nn.functional.normalize(scale_conditioned_features_sam, dim=-1, p=2)
        # equivalent to batch matrix multiplication: torch.bmm(scale_conditioned_features_sam, scale_conditioned_features_sam.transpose(1, 2))
        # calculate the cos(theta) between feature vectors
        corr = torch.einsum('nhc,njc->nhj', scale_conditioned_features_sam, scale_conditioned_features_sam)  # [N_sampled_scales, N_rays, N_rays]

        diag_mask = torch.eye(corr.shape[1], dtype=torch.bool, device=corr.device)

        # Appendix A.1: Resampling
        # the `gt_corrs` is in [N_sample_scales, N_sampled_rays, N_sampled_rays], the value indicates whether a pair of pixels shares a common mask at a given scale
        sum_0 = gt_corrs.sum(dim=0)  # how many common masks the pixel paris have in all scales, [N_sampled_rays, N_sampled_rays]
        consistent_negative = sum_0 == 0  # where no common masks
        consistent_positive = sum_0 == len(gt_corrs)  # where have common masks in all scales
        inconsistent = torch.logical_not(torch.logical_or(consistent_negative, consistent_positive))  # find those pixel paris not belong to above classes
        inconsistent_num = inconsistent.count_nonzero()
        sampled_num = inconsistent_num / 2

        rand_num = torch.rand_like(sum_0)  # [N_sample_pixels, N_sample_pixels]

        # randomly select `sampled_num` pixels from `positive` and `negative` respectively
        sampled_positive = torch.logical_and(consistent_positive, rand_num < sampled_num / consistent_positive.count_nonzero())
        sampled_negative = torch.logical_and(consistent_negative, rand_num < sampled_num / consistent_negative.count_nonzero())

        """
        To tackle the hard samples in training, we also 
          add pixel pairs in Qneg 
            which have feature correspondences larger than 0.5 
          and pixel pairs in Qpos 
            which have feature correspondence smaller than 0.75
        into loss calculation.
        """
        # `torch.logical_and(corr < 0.75, gt_corrs == 1)`: a pair of pixels sharing common masks, but has low cos(theta) value (large theta)
        sampled_mask_positive = torch.logical_or(
            torch.logical_or(
                sampled_positive, torch.any(torch.logical_and(corr < 0.75, gt_corrs == 1), dim=0)
            ),
            inconsistent
        )
        sampled_mask_positive = torch.logical_and(sampled_mask_positive, ~diag_mask)  # exclude elements on diagonal
        sampled_mask_positive = torch.triu(sampled_mask_positive, diagonal=0)  # all elements on and above the main diagonal
        sampled_mask_positive = sampled_mask_positive.bool()

        # `torch.any(torch.logical_and(corr > 0.5, gt_corrs == 0)`: high cos(theta) but does not share any common masks
        sampled_mask_negative = torch.logical_or(
            torch.logical_or(
                sampled_negative, torch.any(torch.logical_and(corr > 0.5, gt_corrs == 0), dim=0)
            ),
            inconsistent
        )
        sampled_mask_negative = torch.logical_and(sampled_mask_negative, ~diag_mask)
        sampled_mask_negative = torch.triu(sampled_mask_negative, diagonal=0)
        sampled_mask_negative = sampled_mask_negative.bool()

        per_pixel_weight = per_pixel_weight.unsqueeze(0)
        #  `sampled_mask_positive` contains those pixel pair having common mask and low cos(theta). The expectation is increasing cos(theta).
        #    `gt_corrs[:, sampled_mask_positive]`, in [N_masks (or scales), N_sampled_pixels], indicating whether the pixels of a pixel pair are both masked at a give mask (or scale)
        #    `corr[:, sampled_mask_positive]`, cos(theta) between two pixels,
        #    since a negative sign exists, the loss term here will increase cos(theta).
        #  `sampled_mask_negative` contains those without common mask but with high cos(theta) which should be decreased.
        #    `1 - gt_corrs[:, sampled_mask_negative]`: at a give mask (or scale), the pixels of a pixel pair are not both masked,
        #    `torch.relu(corr[:, sampled_mask_negative])`: max(corr[...], 0),
        #    decrease cos(theta) of those pixel pairs to minimum this loss term.
        loss = (- per_pixel_weight[:, sampled_mask_positive] * gt_corrs[:, sampled_mask_positive] * corr[:, sampled_mask_positive]).mean() \
               + (per_pixel_weight[:, sampled_mask_negative] * (1 - gt_corrs[:, sampled_mask_negative]) * torch.relu(corr[:, sampled_mask_negative])).mean() \
               + self.rfn * rendered_feature_norm_reg

        with torch.no_grad():
            cosine_pos = corr[gt_corrs == 1].mean()  # higher is better
            cosine_neg = corr[gt_corrs == 0].mean()  # lower is better

        return rendered_features, loss, cosine_pos, cosine_neg, rendered_feature_norm

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        camera, image_info, semantic = batch
        _, loss, cosine_pos, cosine_neg, rendered_feature_norm = self.forward_with_loss_calculation(camera, image_info, semantic)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=1)
        self.log("val/cosine_pos", cosine_pos, prog_bar=True, on_epoch=True, batch_size=1)
        self.log("val/cosine_neg", cosine_neg, prog_bar=True, on_epoch=True, batch_size=1)
        self.log("val/RFN", rendered_feature_norm, prog_bar=True, on_epoch=True, batch_size=1)
        return

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        camera, image_info, semantic = batch
        _, loss, cosine_pos, cosine_neg, rendered_feature_norm = self.forward_with_loss_calculation(camera, image_info, semantic)

        self.log("train/loss", loss, prog_bar=True, on_step=True, batch_size=1)
        self.log("train/cosine_pos", cosine_pos, prog_bar=True, on_step=True, batch_size=1)
        self.log("train/cosine_neg", cosine_neg, prog_bar=True, on_step=True, batch_size=1)
        self.log("train/RFN", rendered_feature_norm, prog_bar=True, on_step=True, batch_size=1)

        if self.trainer.global_step % 100 == 0:
            metrics_to_log = {}
            for opt_idx, opt in enumerate([self.optimizer]):
                if opt is None:
                    continue
                for idx, param_group in enumerate(opt.param_groups):
                    param_group_name = param_group["name"] if "name" in param_group else str(idx)
                    metrics_to_log["lr/{}_{}".format(opt_idx, param_group_name)] = param_group["lr"]
            self.logger.log_metrics(
                metrics_to_log,
                step=self.trainer.global_step,
            )

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        lr_final_factor = self.hparams["optimization"].lr_final_factor

        self.optimizer = torch.optim.Adam(
            params=[
                {
                    "params": [self.gaussian_semantic_features],
                    "name": "gaussian_semantic_features",
                    "lr": self.hparams["optimization"].lr,
                },
                {
                    "params": self.scale_gate.parameters(),
                    "name": "scale_gate",
                    "lr": self.hparams["optimization"].lr,
                }
            ],
            lr=0.,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer=self.optimizer,
                    lr_lambda=lambda iter: lr_final_factor ** min(iter / self.trainer.max_steps, 1),
                ),
                "interval": "step",
                "frequency": 1,
            },
        }

    def to(self, *args: Any, **kwargs: Any) -> Self:
        self.models.gaussian = self.models.gaussian.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def cuda(self, device: Optional[Union[torch.device, int]] = None) -> Self:
        return_value = super().cuda(device)
        self.models.gaussian = self.models.gaussian.to(device=return_value.device)
        return return_value
