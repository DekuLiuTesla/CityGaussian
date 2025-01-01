from typing import Tuple, Optional, Any, List
from dataclasses import dataclass, field
import lightning
import torch
import tinycudann as tcnn
from torch import nn
from gsplat.sh import spherical_harmonics
from .renderer import Renderer
from .gsplat_renderer import GSPlatRenderer, DEFAULT_ANTI_ALIASED_STATUS
from internal.utils.network_factory import NetworkFactory
from ..cameras import Camera
from ..models.gaussian import GaussianModel
from internal.encodings.positional_encoding import PositionalEncoding


@dataclass
class UVEncodingConfig:
    n_levels: int = 8
    base_resolution: int = 16
    per_level_scale: float = 1.405


@dataclass
class NetworkConfig:
    n_neurons: int = 64
    n_layers: int = 3
    skip_layers: List[int] = field(default_factory=lambda: [])


@dataclass
class AppearanceNetworkConfig(NetworkConfig):
    pass


@dataclass
class VisibilityNetworkConfig(NetworkConfig):
    pass


@dataclass
class ModelConfig:
    n_images: int = -1

    # appearance
    n_gaussian_feature_dims: int = 64
    n_appearance_embedding_dims: int = 128
    appearance_network: AppearanceNetworkConfig = field(default_factory=lambda: AppearanceNetworkConfig())
    is_view_dependent: bool = False
    n_view_direction_frequencies: int = 4

    # transient
    n_transient_embedding_dims: int = 128
    uv_encoding: UVEncodingConfig = field(default_factory=lambda: UVEncodingConfig())
    visibility_network: VisibilityNetworkConfig = field(default_factory=lambda: VisibilityNetworkConfig())


@dataclass
class LRConfig:
    lr_init: float
    lr_final_factor: float = 0.1


@dataclass
class OptimizationConfig:
    gamma_eps: float = 1e-6

    appearance_embedding: LRConfig = field(default_factory=lambda: LRConfig(lr_init=2e-3))
    appearance_network: LRConfig = field(default_factory=lambda: LRConfig(lr_init=1e-3))

    transient_embedding: LRConfig = field(default_factory=lambda: LRConfig(lr_init=2e-3))
    uv_encoding: LRConfig = field(default_factory=lambda: LRConfig(lr_init=2e-3))
    visibility_network: LRConfig = field(default_factory=lambda: LRConfig(lr_init=1e-3))

    eps: float = 1e-15
    max_steps: int = 30_000
    appearance_warm_up: int = 1000
    transient_warm_up: int = 2000


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._setup()

    def _setup(self):
        network_factory = NetworkFactory(tcnn=False)

        # appearance
        self.appearance_embedding = nn.Embedding(
            num_embeddings=self.config.n_images,
            embedding_dim=self.config.n_appearance_embedding_dims,
        )
        n_appearance_network_input_dims = self.config.n_gaussian_feature_dims + self.config.n_appearance_embedding_dims
        if self.config.is_view_dependent is True:
            self.view_direction_encoding = PositionalEncoding(3, self.config.n_view_direction_frequencies)
            n_appearance_network_input_dims += self.view_direction_encoding.get_output_n_channels()
        self.appearance_network = network_factory.get_network_with_skip_layers(
            n_input_dims=n_appearance_network_input_dims,
            n_output_dims=3,
            n_layers=self.config.appearance_network.n_layers,
            n_neurons=self.config.appearance_network.n_neurons,
            activation="ReLU",
            output_activation="Sigmoid",
            skips=self.config.appearance_network.skip_layers,
        )

        # transient
        self.transient_embedding = nn.Embedding(
            num_embeddings=self.config.n_images,
            embedding_dim=self.config.n_transient_embedding_dims,
        )
        self.uv_encodings = nn.ModuleList()
        for i in range(self.config.n_images):
            self.uv_encodings.append(tcnn.Encoding(
                n_input_dims=2,
                encoding_config={
                    "otype": "DenseGrid",
                    "n_levels": self.config.uv_encoding.n_levels,
                    "base_resolution": self.config.uv_encoding.base_resolution,
                    "per_level_scale": self.config.uv_encoding.per_level_scale,
                },
                seed=i,
                dtype=torch.float,
            ))
        n_visibility_network_input_dims = self.config.n_transient_embedding_dims + self.uv_encodings[0].n_output_dims
        self.visibility_network = network_factory.get_network_with_skip_layers(
            n_input_dims=n_visibility_network_input_dims,
            n_output_dims=1,
            n_layers=self.config.visibility_network.n_layers,
            n_neurons=self.config.visibility_network.n_neurons,
            activation="ReLU",
            output_activation="Sigmoid",
        )

    def appearance_forward(self, gaussian_features, appearance, view_dirs):
        appearance_embeddings = self.appearance_embedding(appearance.reshape((-1,))).repeat(gaussian_features.shape[0], 1)
        input_tensor_list = [gaussian_features, appearance_embeddings]
        if self.config.is_view_dependent is True:
            input_tensor_list.append(self.view_direction_encoding(view_dirs))
        appearance_network_input = torch.concat(input_tensor_list, dim=-1)
        return self.appearance_network(appearance_network_input)

    def visibility_forward(self, width: int, height: int, appearance):
        # build pixel coordinates
        n = width * height
        transient_embeddings = self.transient_embedding(appearance.reshape((-1,))).repeat(n, 1)

        grid_x, grid_y = torch.meshgrid(
            torch.arange(width, dtype=torch.float, device=transient_embeddings.device),
            torch.arange(height, dtype=torch.float, device=transient_embeddings.device),
            indexing="xy",
        )
        grid_normalized = torch.concat([grid_x.unsqueeze(-1) / (width - 1), grid_y.unsqueeze(-1) / (height - 1)], dim=-1)
        uv_encoding_input = grid_normalized.reshape((-1, 2))
        # encoding pixel coordinates
        uv_encoding_output = self.uv_encodings[appearance.item()](uv_encoding_input)
        # concat with transient_embedding
        visibility_network_input = torch.concat([uv_encoding_output, transient_embeddings], dim=-1)
        return self.visibility_network(visibility_network_input).reshape(grid_normalized.shape[:-1])

    def forward(self, width, height, gaussian_features, appearance, view_dirs):
        return self.appearance_forward(gaussian_features, appearance, view_dirs), self.visibility_forward(width=width, height=height, appearance=appearance)


class GSplatAppearanceEmbeddingVisibilityMapRenderer(Renderer):
    def __init__(
            self,
            model: ModelConfig,
            optimization: OptimizationConfig,
    ):
        super().__init__()
        self.anti_aliased = DEFAULT_ANTI_ALIASED_STATUS  # tell absgrad whether AA enabled
        self.model_config = model
        self.optimization_config = optimization

    def setup(self, stage: str, lightning_module, *args: Any, **kwargs: Any) -> Any:
        if self.model_config.n_images <= 0:
            max_input_id = 0
            appearance_group_ids = lightning_module.trainer.datamodule.dataparser_outputs.appearance_group_ids
            if appearance_group_ids is not None:
                for i in appearance_group_ids.values():
                    if i[0] > max_input_id:
                        max_input_id = i[0]
            n_appearances = max_input_id + 1
            self.model_config.n_images = n_appearances

        self.model = Model(self.model_config)
        # print(self.model)
        self.renderer = GSPlatRenderer()

    def training_setup(self, module: lightning.LightningModule):
        # appearance
        appearance_embedding_optimizer, appearance_embedding_scheduler = self._create_optimizer_and_scheduler(
            self.model.appearance_embedding.parameters(),
            "appearance_embedding",
            lr_init=self.optimization_config.appearance_embedding.lr_init,
            lr_final_factor=self.optimization_config.appearance_embedding.lr_final_factor,
            max_steps=self.optimization_config.max_steps,
            eps=self.optimization_config.eps,
            warm_up=self.optimization_config.appearance_warm_up,
        )
        appearance_network_optimizer, appearance_network_scheduler = self._create_optimizer_and_scheduler(
            self.model.appearance_network.parameters(),
            "appearance_network",
            lr_init=self.optimization_config.appearance_network.lr_init,
            lr_final_factor=self.optimization_config.appearance_network.lr_final_factor,
            max_steps=self.optimization_config.max_steps,
            eps=self.optimization_config.eps,
            warm_up=self.optimization_config.appearance_warm_up,
        )

        # transient
        transient_embedding_optimizer, transient_embedding_scheduler = self._create_optimizer_and_scheduler(
            self.model.transient_embedding.parameters(),
            "transient_embedding",
            lr_init=self.optimization_config.transient_embedding.lr_init,
            lr_final_factor=self.optimization_config.transient_embedding.lr_final_factor,
            max_steps=self.optimization_config.max_steps,
            eps=self.optimization_config.eps,
            warm_up=self.optimization_config.transient_warm_up,
        )
        uv_encoding_optimizer, uv_encoding_scheduler = self._create_optimizer_and_scheduler(
            self.model.uv_encodings.parameters(),
            "uv_encoding",
            lr_init=self.optimization_config.uv_encoding.lr_init,
            lr_final_factor=self.optimization_config.uv_encoding.lr_final_factor,
            max_steps=self.optimization_config.max_steps,
            eps=self.optimization_config.eps,
            warm_up=self.optimization_config.transient_warm_up,
        )
        visibility_network_optimizer, visibility_network_scheduler = self._create_optimizer_and_scheduler(
            self.model.visibility_network.parameters(),
            "visibility_network",
            lr_init=self.optimization_config.visibility_network.lr_init,
            lr_final_factor=self.optimization_config.visibility_network.lr_final_factor,
            max_steps=self.optimization_config.max_steps,
            eps=self.optimization_config.eps,
            warm_up=self.optimization_config.transient_warm_up,
        )

        return [
            appearance_embedding_optimizer,
            appearance_network_optimizer,
            transient_embedding_optimizer,
            uv_encoding_optimizer,
            visibility_network_optimizer,
        ], [
            appearance_embedding_scheduler,
            appearance_network_scheduler,
            transient_embedding_scheduler,
            uv_encoding_scheduler,
            visibility_network_scheduler,
        ]

    def forward(self, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, **kwargs):
        projection_results = GSPlatRenderer.project(
            means3D=pc.get_xyz,
            scales=pc.get_scaling,
            rotations=pc.get_rotation,
            viewpoint_camera=viewpoint_camera,
            scaling_modifier=scaling_modifier,
        )

        radii = projection_results[2]
        is_gaussian_visible = radii > 0

        detached_xyz = pc.get_xyz.detach()
        view_directions = detached_xyz[is_gaussian_visible] - viewpoint_camera.camera_center  # (N, 3)
        view_directions = view_directions / view_directions.norm(dim=-1, keepdim=True)
        base_rgb = spherical_harmonics(pc.active_sh_degree, view_directions, pc.get_features[is_gaussian_visible]) + 0.5
        raw_rgb_offset, visibility = self.model(
            viewpoint_camera.width.item(),
            viewpoint_camera.height.item(),
            pc.get_appearance_features()[is_gaussian_visible],
            viewpoint_camera.appearance_id,
            view_directions
        )
        rgb_offset = raw_rgb_offset * 2 - 1.
        rgbs = torch.zeros((radii.shape[0], 3), dtype=projection_results[0].dtype, device=radii.device)
        rgbs[is_gaussian_visible] = torch.clamp(base_rgb + rgb_offset, min=0., max=1.)

        rasterize_outputs = GSPlatRenderer.rasterize(
            opacities=pc.get_opacity,
            rgbs=rgbs,
            bg_color=bg_color,
            project_results=projection_results,
            viewpoint_camera=viewpoint_camera,
        )
        rasterize_outputs["visibility"] = visibility.unsqueeze(0)
        rasterize_outputs["extra_image"] = rasterize_outputs["visibility"]  # for image saving

        return rasterize_outputs

    def training_forward(self, step: int, module: lightning.LightningModule, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, **kwargs):
        if step < self.optimization_config.appearance_warm_up:
            render_outputs = self.renderer(
                viewpoint_camera,
                pc,
                bg_color,
                scaling_modifier,
                **kwargs,
            )
            render_outputs["visibility"] = torch.ones((1, viewpoint_camera.height, viewpoint_camera.width), device=module.device)
            return render_outputs

        render_outputs = self.forward(viewpoint_camera, pc, bg_color, scaling_modifier, **kwargs)
        if step < self.optimization_config.transient_warm_up:
            render_outputs["visibility"] = torch.ones((1, viewpoint_camera.height, viewpoint_camera.width), device=module.device)
        return render_outputs

    @staticmethod
    def _create_optimizer_and_scheduler(
            params,
            name,
            lr_init,
            lr_final_factor,
            max_steps,
            eps,
            warm_up,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.Adam(
            params=[
                {"params": list(params), "name": name}
            ],
            lr=lr_init,
            eps=eps,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: lr_final_factor ** min(max(iter - warm_up, 0) / max_steps, 1),
            verbose=False,
        )

        return optimizer, scheduler
