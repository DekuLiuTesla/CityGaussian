import random
from dataclasses import dataclass
from typing import Any, Dict
import os

import numpy as np
import torch
from tqdm.auto import tqdm

from . import RendererOutputInfo
from .gsplat_v1_renderer import GSplatV1Renderer
from .renderer import RendererConfig, Renderer
from ..cameras import Camera
from ..models.gaussian import GaussianModel


@dataclass
class PartitionLoDRenderer(RendererConfig):
    data: str
    names: list[str]  # from finest to coarsest
    min_images: int = 32
    visibility_filter: bool = False
    freeze: bool = False
    drop_shs_rest: bool = False

    def instantiate(self, *args, **kwargs) -> "PartitionLoDRendererModule":
        return PartitionLoDRendererModule(self)


class PartitionLoDRendererModule(Renderer):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.on_render_hooks = []
        self.on_model_updated_hooks = []

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        super().setup(stage, *args, **kwargs)

        import sys
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "utils",
        ))

        device = torch.device("cuda")

        # load partition data
        from internal.utils.partitioning_utils import PartitionCoordinates
        partitions = torch.load(os.path.join(self.config.data, "partitions.pt"))
        self.orientation_transform = torch.eye(3)
        if partitions["extra_data"] is not None:
            self.orientation_transform = partitions["extra_data"]["rotation_transform"]
        self.orientation_transform = self.orientation_transform.T.to(device=device)[:3, :3]

        partition_size = partitions["scene_config"]["partition_size"]
        self.partition_size = partition_size

        self.partition_coordinates = PartitionCoordinates(**partitions["partition_coordinates"])
        self.partition_bounding_boxes = self.partition_coordinates.get_bounding_boxes(partition_size).to(device=device)

        # load partitions' models
        from internal.utils.gaussian_model_loader import GaussianModelLoader
        from utils.train_partitions import PartitionTraining, PartitionTrainingConfig
        lods = []  # [N_lods, N_partitions]

        for lod in tqdm(self.config.names):
            models = []

            partition_training = PartitionTraining(PartitionTrainingConfig(
                partition_dir=self.config.data,
                project_name=lod,
                min_images=self.config.min_images,
                n_processes=1,
                process_id=1,
                dry_run=False,
                extra_epoches=0,
            ))
            trainable_partition_idx_list = partition_training.get_trainable_partition_idx_list(
                min_images=self.config.min_images,
                n_processes=1,
                process_id=1,
            )

            for partition_idx in tqdm(trainable_partition_idx_list, leave=False):
                ckpt = torch.load(os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "outputs",
                    lod,
                    partition_training.get_experiment_name(partition_idx),
                    "preprocessed.ckpt"
                ), map_location="cpu")
                gaussian_model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, device)
                if self.config.drop_shs_rest:
                    gaussian_model.config.sh_degree = 0
                    gaussian_model.active_sh_degree = 0
                    gaussian_model.shs_rest = torch.empty((gaussian_model.n_gaussians, 0, 3), device=gaussian_model.shs_dc.device)
                gaussian_model.pre_activate_all_properties()
                gaussian_model.freeze()
                gaussian_model.to(device=device)
                models.append(gaussian_model)

            lods.append(models)
        self.lods = lods

        # retain trainable partitions only
        trainable_partition_idx = torch.tensor(trainable_partition_idx_list)
        self.partition_coordinates.id = self.partition_coordinates.id[trainable_partition_idx]
        self.partition_coordinates.xy = self.partition_coordinates.xy[trainable_partition_idx]
        self.partition_bounding_boxes.min = self.partition_bounding_boxes.min[trainable_partition_idx]
        self.partition_bounding_boxes.max = self.partition_bounding_boxes.max[trainable_partition_idx]

        # get partition 3D bounding boxes
        partition_min_max_z = []
        for partition_model in self.lods[0]:
            partition_z_value = partition_model.get_means() @ self.orientation_transform[:, -1:]
            partition_min_max_z.append(torch.stack([partition_z_value.min(), partition_z_value.max()]))
        partition_min_max_z = torch.stack(partition_min_max_z)
        # print(partition_min_max_z)
        partition_full_2d_bounding_box = []
        for partition_xy in self.partition_coordinates.xy:
            partition_xy = partition_xy.to(device=device)
            left_down = partition_xy
            left_top = partition_xy + torch.tensor([0., partition_size], device=device)
            right_down = partition_xy + torch.tensor([partition_size, 0.], device=device)
            right_top = partition_xy + partition_size
            partition_full_2d_bounding_box.append(torch.stack([
                left_down,
                left_top,
                right_down,
                right_top,
            ]))
        partition_full_2d_bounding_box = torch.stack(partition_full_2d_bounding_box)  # [N_partitions, 4, 2]
        partition_min_max_z = partition_min_max_z[:, None, :].repeat(1, 4, 1)
        self.partition_full_3d_bounding_box = torch.concat([
            torch.concat([partition_full_2d_bounding_box, partition_min_max_z[..., 0:1]], dim=-1),
            torch.concat([partition_full_2d_bounding_box, partition_min_max_z[..., 1:2]], dim=-1),
        ], dim=1) @ self.orientation_transform.T  # [N_partitions, 8, 3], in world space
        # print(self.partition_full_3d_bounding_box)

        # set default LoD distance thresholds
        self.lod_thresholds = (torch.arange(1, len(self.lods)) * 0.25 * partition_size).to(device=device)  # [N_lods - 1]

        # initialize partition lod states
        self.n_partitions = len(self.lods[0])
        self.partition_lods = torch.empty(self.n_partitions, dtype=torch.int8, device=device).fill_(127)  # [N_partitions]
        self.is_partition_visible = torch.ones(self.n_partitions, dtype=torch.bool, device=device)

        # setup empty Gaussian Model
        self.gaussian_model = gaussian_model.config.instantiate()
        self.gaussian_model.setup_from_number(0)
        self.gaussian_model.pre_activate_all_properties()
        self.gaussian_model.freeze()
        self.gaussian_model.active_sh_degree = gaussian_model.active_sh_degree
        self.gaussian_model.to(device=device)

        # setup gsplat renderer
        renderer_config = GSplatV1Renderer(
            block_size=getattr(ckpt["hyper_parameters"]["renderer"], "block_size", 16),
            anti_aliased=getattr(ckpt["hyper_parameters"]["renderer"], "anti_aliased", True),
            filter_2d_kernel_size=getattr(ckpt["hyper_parameters"]["renderer"], "filter_2d_kernel_size", 0.3),
            separate_sh=getattr(ckpt["hyper_parameters"]["renderer"], "separate_sh", True),
            tile_based_culling=getattr(ckpt["hyper_parameters"]["renderer"], "tile_based_culling", False),
        )
        print(renderer_config)
        self.gsplat_renderer = renderer_config.instantiate()
        self.gsplat_renderer.setup(stage)

    def get_partition_distances(self, p: torch.Tensor):
        p = (p @ self.orientation_transform)[:2]
        dist_min2p = self.partition_bounding_boxes.min - p
        dist_p2max = p - self.partition_bounding_boxes.max
        dxy = torch.maximum(dist_min2p, dist_p2max)
        return torch.sqrt(torch.pow(dxy.clamp(min=0.), 2).sum(dim=-1))  # [N_partitions]

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
            **kwargs,
    ):
        partition_distances = self.get_partition_distances(viewpoint_camera.camera_center)
        # print((partition_distances == 0.).nonzero())
        self.partition_distances = partition_distances
        partition_lods = torch.ones_like(self.partition_lods).fill_(-1)

        # set lods by distances
        for i in range(len(self.lods) - 2, -1, -1):
            partition_lods[partition_distances < self.lod_thresholds[i]] = i

        # visibility
        # TODO: optimize visibility based filter, it does not correctly know whether a partition partly in front of the camera is invisible
        is_partition_visible = torch.ones_like(self.is_partition_visible)
        if self.config.visibility_filter:
            full_perspective_projection = viewpoint_camera.get_full_perspective_projection()
            partition_corners_in_image_space_in_homogeneous = self.partition_full_3d_bounding_box @ full_perspective_projection[:3, :3] + full_perspective_projection[3, :3]
            partition_corners_in_image_space = partition_corners_in_image_space_in_homogeneous[..., :2] / (torch.abs(partition_corners_in_image_space_in_homogeneous[..., 2:3]) + 1e-6)  # [N_partitions, 8, 2]

            min_pixel_coor = torch.tensor([0., 0.], dtype=torch.float, device=partition_corners_in_image_space.device)
            max_pixel_coor = torch.tensor([viewpoint_camera.width, viewpoint_camera.height], dtype=torch.float, device=partition_corners_in_image_space.device)

            is_behind_camera = partition_corners_in_image_space_in_homogeneous[..., -1:] <= 0.  # [N_partitions, 8, 1]
            is_partition_behind_camera = torch.all(is_behind_camera.squeeze(-1), dim=1)  # [N_partition]

            # move them into the image plane
            # print("partition_corners_in_image_space[0]={}".format(partition_corners_in_image_space[0].cpu().numpy()))
            partition_corners_in_image_space = torch.maximum(partition_corners_in_image_space, min_pixel_coor)
            partition_corners_in_image_space = torch.minimum(partition_corners_in_image_space, max_pixel_coor)
            # print("partition_corners_in_image_space[0]={}".format(partition_corners_in_image_space[0].cpu().numpy()))

            # prevent selecting valid min and max values from corners behind camera by applying the `behind_camera_corner_xy_offset`
            # behind_camera_corner_xy_offset = is_behind_camera * max_pixel_coor.max()
            behind_camera_corner_xy_offset = 0.
            partition_corners_min = torch.min(partition_corners_in_image_space + behind_camera_corner_xy_offset, dim=1).values  # [N_partitions, 2]
            partition_corners_max = torch.max(partition_corners_in_image_space - behind_camera_corner_xy_offset, dim=1).values  # [N_partitions, 2]
            # the results of (partition_corners_max - partition_corners_min) of corners behind camera will <= 0
            partition_projection_area = torch.prod((partition_corners_max - partition_corners_min).clamp(min=0.), dim=-1)  # [N_partitions]
            # print("partition_projection_area[0]={}".format(partition_projection_area[0].cpu().numpy()))

            is_partition_visible = torch.gt(partition_projection_area, torch.tensor(0., device=partition_projection_area.device))
            is_partition_visible = torch.logical_and(is_partition_visible, torch.logical_not(is_partition_behind_camera))
            is_partition_visible[torch.argmin(partition_distances)] = True
            # print("is_partition_visible[0]={}".format(is_partition_visible[0]))

        if not self.config.freeze and (not torch.all(torch.eq(self.partition_lods, partition_lods)) or not torch.all(torch.eq(self.is_partition_visible, is_partition_visible))):
            # update stored lods
            self.partition_lods = partition_lods
            self.is_partition_visible = is_partition_visible
            # update model
            properties = {}
            for partition_idx, lod in enumerate(partition_lods):
                if not is_partition_visible[partition_idx]:
                    continue
                for k, v in self.lods[lod.item()][partition_idx].properties.items():
                    properties.setdefault(k, []).append(v)

            for k in list(properties.keys()):
                properties[k] = torch.concat(properties[k], dim=0).to(device="cuda")

            self.gaussian_model.properties = properties

            for i in self.on_model_updated_hooks:
                i()

        for i in self.on_render_hooks:
            i()

        outputs = self.gsplat_renderer(
            viewpoint_camera,
            self.gaussian_model,
            bg_color,
            scaling_modifier,
            render_types,
            **kwargs,
        )
        # if self.config.visibility_filter:
        #     for idx, corner in enumerate(partition_corners_in_image_space[0]):
        #         red_value = 1.
        #         if partition_corners_in_image_space_in_homogeneous[0, idx, 2] <= 0.:
        #             red_value = 0.
        #         box_min = torch.clamp(corner - 16, min=0).to(torch.int)
        #         box_max = corner + 16
        #         box_max = torch.minimum(box_max, max_pixel_coor).to(torch.int)
        #         outputs["render"][1:3, box_min[1]:box_max[1], box_min[0]:box_max[0]] = (1 - red_value)
        #         outputs["render"][0, box_min[1]:box_max[1], box_min[0]:box_max[0]] = red_value
        return outputs

    def get_available_outputs(self) -> Dict[str, RendererOutputInfo]:
        return self.gsplat_renderer.get_available_outputs()

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        with tabs.add_tab("LoD"):
            self.viewer_options = ViewerOptions(self, viewer, server)


import viser
from viser import GuiMarkdownHandle
from viser import ViserServer


class ViewerOptions:
    def __init__(self, renderer: PartitionLoDRendererModule, viewer, server):
        super().__init__()
        self.renderer = renderer
        self.viewer = viewer
        self.server: ViserServer = server

        self.show_label = False

        self.setup_status_folder()
        self.setup_labels()
        self.setup_settings_folder()

        # self.draw_3d_boxes()

        self.renderer.on_model_updated_hooks.append(self.update_number_of_gaussians)

    def setup_status_folder(self):
        with self.server.gui.add_folder("Status"):
            self.markdown: GuiMarkdownHandle = self.server.gui.add_markdown("")

    def setup_settings_folder(self):
        def get_lod_threshold_updater(idx):
            def updater(v: viser.GuiEvent):
                self.renderer.lod_thresholds[idx] = v.target.value

            return updater

        with self.server.gui.add_folder("Settings"):
            # threshold
            with self.server.gui.add_folder("Max Distance"):
                for idx, i in enumerate(self.renderer.lod_thresholds):
                    lod_threshold_number = self.server.gui.add_number(label="LoD {}".format(idx), initial_value=i.item())
                    lod_threshold_number.on_update(get_lod_threshold_updater(idx))
                    self.viewer.rerender_for_all_client()

            # visibility filter
            visibility_filter_checkbox = self.server.gui.add_checkbox("Visibility Filter", initial_value=self.renderer.config.visibility_filter)

            @visibility_filter_checkbox.on_update
            def _(_):
                self.renderer.config.visibility_filter = visibility_filter_checkbox.value

            # visibility filter
            freeze_checkbox = self.server.gui.add_checkbox("Freeze", initial_value=self.renderer.config.freeze)

            @freeze_checkbox.on_update
            def _(_):
                self.renderer.config.freeze = freeze_checkbox.value

            # labels
            show_label_checkbox = self.server.gui.add_checkbox("Show Labels", initial_value=self.show_label)

            @show_label_checkbox.on_update
            def _(_):
                self.show_label = show_label_checkbox.value
                self.update_labels()

    def setup_labels(self):
        self.label_updaters = []

        def get_label_updater(name, text, position):
            label = None

            def updater(distance, lod):
                nonlocal label
                if label is not None:
                    label.remove()
                    label = None
                if not self.show_label:
                    return
                label = self.server.scene.add_label(
                    name=name,
                    text="{}, {:.2f}, {}".format(text, distance, lod),
                    position=position,
                )

            return updater

        for idx, (id, xy) in enumerate(self.renderer.partition_coordinates):
            self.label_updaters.append(get_label_updater(
                name="part/{}".format(idx),
                text="#{}({},{})".format(idx, *id.tolist()),
                position=(torch.concat([xy + 0.5 * self.renderer.partition_size, torch.tensor([0.])]).cuda() @ self.renderer.orientation_transform.T).cpu().numpy(),
            ))

        self.renderer.on_render_hooks.append(self.update_labels)

    def draw_3d_boxes(self):
        from matplotlib.pyplot import cm
        colors = list(iter(cm.rainbow(np.linspace(0, 1, self.renderer.partition_full_3d_bounding_box.shape[0]))))
        random.shuffle(colors)

        for idx, box in enumerate(self.renderer.partition_full_3d_bounding_box):
            for cornerIdx, corner in enumerate(box):
                self.server.scene.add_icosphere(
                    name="box/{}-{}".format(idx, cornerIdx),
                    radius=0.01 * self.renderer.partition_size,
                    color=colors[idx][:3],
                    position=corner.cpu().numpy(),
                )
            # break

    def update_number_of_gaussians(self):
        self.markdown.content = "Gaussians: {}".format(
            self.renderer.gaussian_model.n_gaussians,
        )

    def update_labels(self):
        for idx, i in enumerate(self.label_updaters):
            i(self.renderer.partition_distances[idx].item(), self.renderer.partition_lods[idx].item())
