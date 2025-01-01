import traceback
from typing import Literal, Any, Tuple, Optional, Union, List, Dict

import lightning
import torch
from .renderer import RendererOutputTypes, RendererOutputInfo, Renderer
from .gsplat_renderer import GSPlatRenderer
from gsplat.sh import spherical_harmonics
import sklearn
import sklearn.decomposition
import numpy as np
from ..cameras import Camera
from ..models.gaussian import GaussianModel


class NoFeatureDecoder(torch.nn.Module):
    def forward(self, i):
        return i


class CNNDecoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_dim, output_dim, kernel_size=1).cuda()

    def forward(self, x):
        x = self.conv(x)
        return x


class Feature3DGSRenderer(Renderer):
    def __init__(
            self,
            speedup: bool,
            n_feature_dims: int,
            feature_lr: float = 0.001,
            feature_decoder_lr: float = 0.0001,
            rasterize_batch: int = 32,
    ):
        super().__init__()
        self.speedup = speedup
        self.n_feature_dims = n_feature_dims
        self.feature_lr = feature_lr
        self.feature_decoder_lr = feature_decoder_lr
        self.rasterize_batch = rasterize_batch

        # update this when feature updated
        self.pca_projected_color = None

        self.edit_mask = None
        self.edit_mask_2d = None

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        n_actual_feature_dims = self.n_feature_dims

        # setup feature decoder
        self.feature_decoder = NoFeatureDecoder()
        if self.speedup is True:
            n_actual_feature_dims = n_actual_feature_dims // 2
            self.feature_decoder = CNNDecoder(n_actual_feature_dims, self.n_feature_dims)

        self.n_actual_feature_dims = n_actual_feature_dims

        # initialize features
        feature_tensor = torch.zeros(
            (kwargs["lightning_module"].gaussian_model.n_gaussians, n_actual_feature_dims),
            dtype=torch.float,
            device=kwargs["lightning_module"].device,
            requires_grad=True,
        )
        self.features = torch.nn.Parameter(feature_tensor)

    def training_setup(self, module: lightning.LightningModule) -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        optimizer = torch.optim.Adam(
            params=[
                {"name": "features", "params": [self.features], "lr": self.feature_lr},
                {"name": "feature_decoder", "params": self.feature_decoder.parameters(), "lr": self.feature_decoder_lr},
            ]
        )

        return optimizer, None

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
            **kwargs,
    ):
        if render_types is None:
            render_types = ["features"]

        with torch.no_grad():
            project_results = GSPlatRenderer.project(
                means3D=pc.get_xyz,
                scales=pc.get_scaling,
                rotations=pc.get_rotation,
                viewpoint_camera=viewpoint_camera,
                scaling_modifier=scaling_modifier,
            )
            # anti-aliased
            comp = project_results[4]
            opacities = pc.get_opacity * comp[:, None]

        outputs = {}

        if "rgb" in render_types:
            with torch.no_grad():
                viewdirs = pc.get_xyz.detach() - viewpoint_camera.camera_center  # (N, 3)
                rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
                rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
                outputs["render"] = GSPlatRenderer.rasterize_simplified(
                    project_results,
                    viewpoint_camera=viewpoint_camera,
                    colors=rgbs,
                    bg_color=bg_color,
                    opacities=opacities,
                    anti_aliased=False,
                )
                if getattr(self, "edit_mask_2d", None) is not None:
                    interpolated_mask = torch.nn.functional.interpolate(
                        self.edit_mask_2d.unsqueeze(0).unsqueeze(0).float(),
                        size=(outputs["render"].shape[1], outputs["render"].shape[2]),
                        mode='bilinear',
                        align_corners=True,
                    ).squeeze(0)
                    outputs["render"] = outputs["render"] * (interpolated_mask > 0.5)
        if "features" in render_types or "features_vanilla_pca_2d" in render_types:
            rendered_features_list = []
            feature_bg_color = torch.zeros((self.rasterize_batch,), dtype=torch.float, device=self.features.device)
            for i in range(self.n_actual_feature_dims // self.rasterize_batch):
                start = i * self.rasterize_batch
                rendered_features_list.append(GSPlatRenderer.rasterize_simplified(
                    project_results,
                    viewpoint_camera=viewpoint_camera,
                    colors=self.features[..., start:start + self.rasterize_batch],
                    bg_color=feature_bg_color,
                    opacities=opacities,
                    anti_aliased=False,
                ))
            raw_features = torch.concat(rendered_features_list, dim=0)
            outputs["raw_features"] = raw_features
            outputs["features"] = self.feature_decoder(raw_features)
        if "features_vanilla_pca_2d" in render_types:
            outputs["features_vanilla_pca_2d"] = self.feature_visualize(outputs["features"])
        if "features_pca_3d" in render_types:
            if getattr(self, "pca_projected_color", None) is None:
                from internal.utils.seganygs import SegAnyGSUtils
                normalized_features = torch.nn.functional.normalize(self.features, dim=-1)
                normalized_features[torch.isnan(normalized_features)] = 0.
                self.pca_projected_color = SegAnyGSUtils.get_pca_projected_colors(
                    semantic_features=normalized_features,
                    pca_projection_matrix=SegAnyGSUtils.get_pca_projection_matrix(semantic_features=normalized_features),
                )
            features_pca_3d = GSPlatRenderer.rasterize_simplified(
                project_results,
                viewpoint_camera=viewpoint_camera,
                colors=self.pca_projected_color,
                bg_color=bg_color,
                opacities=opacities,
                anti_aliased=False,
            )
            # view_shape = (3, -1)
            # features_pca_3d = features_pca_3d - torch.min(features_pca_3d.view(view_shape), dim=1, keepdim=True).values.unsqueeze(-1)
            # features_pca_3d = features_pca_3d / (torch.max(features_pca_3d.view(view_shape), dim=1, keepdim=True).values.unsqueeze(-1) + 1e-9)
            outputs["features_pca_3d"] = features_pca_3d
        if "edited" in render_types:
            edited_opacities = opacities
            if getattr(self, "edit_mask", None) is not None:
                edited_opacities = edited_opacities * self.edit_mask.unsqueeze(-1)

            viewdirs = pc.get_xyz.detach() - viewpoint_camera.camera_center  # (N, 3)
            rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

            outputs["edited"] = GSPlatRenderer.rasterize_simplified(
                project_results,
                viewpoint_camera=viewpoint_camera,
                colors=rgbs,
                bg_color=bg_color,
                opacities=edited_opacities,
                anti_aliased=False,
            )

        return outputs

    def training_forward(self, step: int, module: lightning.LightningModule, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        return self(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            render_types=["features"],
            **kwargs,
        )

    def get_available_outputs(self) -> Dict:
        return {
            "rgb": RendererOutputInfo(key="render"),
            "features": RendererOutputInfo(key="features", type=RendererOutputTypes.FEATURE_MAP),
            "features_vanilla_pca_2d": RendererOutputInfo(key="features_vanilla_pca_2d"),
            "features_pca_3d": RendererOutputInfo(key="features_pca_3d"),
            "edited": RendererOutputInfo(key="edited"),
        }

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        self.viewer_options = ViewerOptions(self, viewer, server, tabs)

    # @staticmethod
    # def feature_visualize(image: torch.Tensor):
    #     from internal.utils.seganygs import SegAnyGSUtils
    #
    #     feature_map = image.permute(1, 2, 0).view((-1, image.shape[0]))
    #     normalized_feature_map = torch.nn.functional.normalize(feature_map, dim=-1)
    #     pca_projection_matrix = SegAnyGSUtils.get_pca_projection_matrix(normalized_feature_map)
    #     pca_projected_colors = SegAnyGSUtils.get_pca_projected_colors(normalized_feature_map, pca_projection_matrix)
    #     pca_feature_map = pca_projected_colors.view(*image.shape[1:], pca_projected_colors.shape[-1]).permute(2, 0, 1)
    #     return pca_feature_map

    @staticmethod
    def feature_visualize(feature):
        fmap = feature[None, :, :, :]  # torch.Size([1, C, H, W])
        fmap = torch.nn.functional.normalize(fmap, dim=1)
        pca = sklearn.decomposition.PCA(3, random_state=42)
        f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
        transformed = pca.fit_transform(f_samples)
        feature_pca_mean = torch.tensor(f_samples.mean(0), dtype=torch.float, device=feature.device)
        feature_pca_components = torch.tensor(pca.components_, dtype=torch.float, device=feature.device)
        q1, q99 = np.percentile(transformed, [1, 99])
        feature_pca_postprocess_sub = q1
        feature_pca_postprocess_div = (q99 - q1)
        del f_samples
        vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
        vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
        vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).permute(2, 0, 1)
        return vis_feature


class ViewerOptions:
    def __init__(self, renderer: Feature3DGSRenderer, viewer, server, tabs):
        from viser import ViserServer

        self.renderer = renderer
        self.viewer = viewer
        self.server: ViserServer = server
        self.tabs = tabs

        self._setup()

    def _setup(self):
        with self.tabs.add_tab("Semantic"):
            # with self.server.gui.add_folder(
            #         label="Load Model",
            # ) as f:
            #     type_dropdown = self.server.gui.add_dropdown(
            #         label="Type",
            #         options=["SAM"],
            #     )
            #     path_text = self.server.gui.add_text(
            #         label="Path",
            #         initial_value="sam_vit_h_4b8939.pth",
            #     )
            #     load_model_button = self.server.gui.add_button(
            #         label="Load",
            #     )
            #
            #     @load_model_button.on_click
            #     def _(_):
            #         load_model_button.disabled = True
            #         with self.server.atomic():
            #             try:
            #                 self._load_sam_model(path_text.value)
            #                 message = "Loaded successfully"
            #                 model_loaded = True
            #             except Exception as e:
            #                 model_loaded = False
            #                 message = repr(e)
            #                 traceback.print_exc()
            #         if model_loaded is True:
            #             f.remove()
            #         else:
            #             load_model_button.disabled = False
            #         self.viewer.show_message(message)

            if self.renderer.n_feature_dims == 512:
                self._setup_lseg_options()
            else:
                self.server.gui.add_markdown("No option for SAM")

    def _setup_lseg_options(self):
        objects = ["car", "tree", "building", "sidewalk", "road"]
        clip_editor = self._get_clip_editor()
        text_feature = clip_editor.encode_text([obj.replace("_", " ") for obj in objects])
        del clip_editor
        torch.cuda.empty_cache()

        with self.server.gui.add_folder("LSeg"):
            object_dropdown = self.server.gui.add_dropdown(
                label="Object",
                options=objects,
            )
            # object_text = self.server.gui.add_text(
            #     label="Text",
            #     initial_value="",
            # )
            score_2d_threshold_slider = self.server.gui.add_slider(
                label="Score 2D",
                min=0.,
                max=1.,
                step=0.001,
                initial_value=1. / len(objects),
                visible=self.renderer.n_actual_feature_dims == 256,
            )
            score_3d_threshold_slider = self.server.gui.add_slider(
                label="Score 3D",
                min=0.,
                max=1.,
                step=0.001,
                initial_value=0.95 if self.renderer.n_actual_feature_dims == 256 else 0.2,
            )

            extract_button = self.server.gui.add_button(
                label="Extract",
            )
            reset_button = self.server.gui.add_button(
                label="Reset",
            )

            self.server.gui.add_markdown("<b>[NOTE]</b> Switch to `edited` mode in 'General' panel to visualize the 3D extraction result")

            @extract_button.on_click
            def _(event):
                try:
                    target_idx = objects.index(object_dropdown.value)
                except ValueError:
                    self.viewer.show_message("Object not supported")
                    return

                # if object_text.value == "":
                #     self.viewer.show_message("Empty value")
                #     return
                #
                # text_feature = clip_editor.encode_text([object_text.value.replace("_", "")])

                with torch.no_grad(), self.server.atomic():
                    if self.renderer.n_actual_feature_dims == 256:
                        """
                        `n_actual_feature_dims == 256` means that LSeg with speedup mode enabled.
                        Since the number of LSeg feature dimension is 512, it is impossible to calculate similarity in 3D directly.
                        So render the feature maps first, i.e. the raw feature map (256 dims) and decoded feature map (512 dims).
                        Then calculate similarity in decoded feature map, and get the mask representing pixels having suitable score.
                        Then get features in raw feature map correspond to this mask.
                        Then get the mean feature of these features.
                        Finally use this mean feature to calculate similarity in 3D.
                        
                        Not performs well on finding object outside of current view.
                        """
                        from internal.viewer.client import ClientThread
                        render_outputs = self.renderer.forward(
                            viewpoint_camera=ClientThread.get_camera(
                                event.client.camera,
                                self.viewer.max_res_when_moving.value,
                            ).to_device(self.viewer.device),
                            pc=self.viewer.viewer_renderer.gaussian_model,
                            bg_color=torch.zeros((self.renderer.n_actual_feature_dims,), dtype=torch.float, device=self.viewer.device),
                            render_types=["features"],
                        )

                        feature_map = render_outputs["features"]
                        feature_map_in_hwc = render_outputs["features"].permute(1, 2, 0)
                        feature_map_flatten = feature_map_in_hwc.reshape((-1, self.renderer.n_feature_dims))
                        raw_feature_map_in_hwc = render_outputs["raw_features"].permute(1, 2, 0)

                        scores_2d = self.calculate_selection_score(
                            feature_map_flatten,
                            text_feature,
                            score_2d_threshold_slider.value,
                            positive_ids=[target_idx],
                        )

                        mask_2d = (scores_2d >= 0.5).reshape(feature_map.shape[1:])
                        self.renderer.edit_mask_2d = mask_2d
                        mean_masked_raw_features = raw_feature_map_in_hwc[mask_2d].reshape((-1, self.renderer.n_actual_feature_dims)).mean(dim=0)

                        scores_3d = self.calculate_selection_score(
                            self.renderer.features,
                            mean_masked_raw_features.unsqueeze(0),
                            score_3d_threshold_slider.value,
                            positive_ids=[0],
                        )
                    else:
                        scores_3d = self.calculate_selection_score(
                            self.renderer.features,
                            text_feature,
                            score_3d_threshold_slider.value,
                            positive_ids=[target_idx],
                        )

                    mask = (scores_3d >= 0.5)
                    self.renderer.edit_mask = mask

                self.viewer.rerender_for_all_client()

            @reset_button.on_click
            def _(_):
                self.renderer.edit_mask = None
                self.renderer.edit_mask_2d = None
                self.viewer.rerender_for_all_client()

    @property
    def device(self):
        return self.viewer.device

    def _load_sam_model(self, path, arch="vit_h"):
        assert path.endswith(".ckpt")
        from segment_anything import SamPredictor, sam_model_registry
        self._sam = sam_model_registry[arch](checkpoint=path).to(self.device)
        self._predictor = SamPredictor(self._sam)

    @staticmethod
    def calculate_selection_score(features, query_features, score_threshold=None, positive_ids=[0]):
        features /= features.norm(dim=-1, keepdim=True)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        scores = features.half() @ query_features.T.half()  # (N_points, n_texts)
        if scores.shape[-1] == 1:
            scores = ((scores[:, 0] + 1.) / 2.)  # (N_points,)
            scores = (scores >= score_threshold).float()
        else:
            scores = torch.nn.functional.softmax(scores, dim=-1)  # (N_points, n_texts)
            if score_threshold is not None:
                scores = scores[:, positive_ids].sum(-1)  # (N_points, )
                scores = (scores >= score_threshold).float()
            else:
                scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)  # (N_points, )
                scores = torch.isin(torch.argmax(scores, dim=-1), torch.tensor(positive_ids).cuda()).float()
        return scores

    def _get_clip_editor(self):
        import clip

        class CLIPEditor(object):
            def __init__(self, device):
                super(CLIPEditor, self).__init__()
                self.device = device
                self.model, _preprocess = clip.load("ViT-B/32", device=self.device)
                self.model = self.model.float()

            def encode_text(self, text_list):
                with torch.no_grad():
                    texts = clip.tokenize(text_list).to(self.device)
                    text_features = self.model.encode_text(texts)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features

        clip_editor = CLIPEditor(self.viewer.device)
        return clip_editor
