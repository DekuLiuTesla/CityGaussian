import os
from typing import Tuple, Callable
import dataclasses
from dataclasses import dataclass

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm


@dataclass
class SceneConfig:
    origin: torch.Tensor

    partition_size: float

    location_based_enlarge: float = None
    """ enlarge bounding box by `partition_size * location_based_enlarge`, used for location based camera assignment """

    visibility_based_distance: float = None
    """ enlarge bounding box by `partition_size * visibility_based_distance`, used for visibility based camera assignment """

    visibility_threshold: float = None

    bounding_box_based_visibility: bool = False


@dataclass
class MinMaxBoundingBox:
    min: torch.Tensor  # [2]
    max: torch.Tensor  # [2]


@dataclass
class MinMaxBoundingBoxes:
    min: torch.Tensor  # [N, 2]
    max: torch.Tensor  # [N, 2]

    def __getitem__(self, item):
        return MinMaxBoundingBox(
            min=self.min[item],
            max=self.max[item],
        )

    def to(self, *args, **kwargs):
        self.min = self.min.to(*args, **kwargs)
        self.max = self.max.to(*args, **kwargs)
        return self


@dataclass
class SceneBoundingBox:
    bounding_box: MinMaxBoundingBox
    n_partitions: torch.Tensor  # = [N_x, N_y]
    origin_partition_offset: torch.Tensor


@dataclass
class PartitionCoordinates:
    id: torch.Tensor  # [N_partitions, 2]
    xy: torch.Tensor  # [N_partitions, 2]

    def __len__(self):
        return self.id.shape[0]

    def __getitem__(self, item):
        return self.id[item], self.xy[item]

    def __iter__(self):
        for idx in range(len(self)):
            yield self.id[idx], self.xy[idx]

    def get_bounding_boxes(self, size: float, enlarge: float = 0.) -> MinMaxBoundingBoxes:
        xy_min = self.xy - (enlarge * size)  # [N_partitions, 2]
        xy_max = self.xy + size + (enlarge * size)  # [N_partitions, 2]
        return MinMaxBoundingBoxes(
            min=xy_min,
            max=xy_max,
        )

    def get_str_id(self, idx: int) -> str:
        return Partitioning.partition_id_to_str(self.id[idx])


@dataclass
class PartitionableScene:
    scene_config: SceneConfig

    camera_centers: torch.Tensor

    camera_center_based_bounding_box: MinMaxBoundingBox = None

    scene_bounding_box: SceneBoundingBox = None

    partition_coordinates: PartitionCoordinates = None

    is_camera_in_partition: torch.Tensor = None  # [N_partitions, N_cameras]

    camera_visibilities: torch.Tensor = None  # [N_partitions, N_cameras]

    is_partitions_visible_to_cameras: torch.Tensor = None  # [N_partitions, N_cameras]

    def get_bounding_box_by_camera_centers(self, enlarge: float = 0.):
        self.camera_center_based_bounding_box = Partitioning.get_bounding_box_by_camera_centers(self.camera_centers, enlarge=enlarge)
        return self.camera_center_based_bounding_box

    def get_scene_bounding_box(self):
        self.scene_bounding_box = Partitioning.align_bounding_box(
            self.camera_center_based_bounding_box,
            origin=self.scene_config.origin,
            size=self.scene_config.partition_size,
        )
        return self.scene_bounding_box

    def build_partition_coordinates(self):
        self.partition_coordinates = Partitioning.build_partition_coordinates(
            self.scene_bounding_box,
            self.scene_config.origin,
            self.scene_config.partition_size,
        )
        return self.partition_coordinates

    def camera_center_based_partition_assignment(self):
        self.is_camera_in_partition = Partitioning.camera_center_based_partition_assignment(
            partition_coordinates=self.partition_coordinates,
            camera_centers=self.camera_centers,
            size=self.scene_config.partition_size,
            enlarge=self.scene_config.location_based_enlarge,
        )
        return self.is_camera_in_partition
    
    def calculate_camera_visibilities(self, *args, **kwargs):
        if self.scene_config.bounding_box_based_visibility:
            return self.calculate_point_bounding_box_based_camera_visibilities(*args, **kwargs)
        return self.calculate_point_based_camera_visibilities(*args, **kwargs)

    def calculate_point_based_camera_visibilities(self, point_getter: Callable, device):
        self.camera_visibilities = Partitioning.cameras_point_based_visibilities_calculation(
            partition_coordinates=self.partition_coordinates,
            size=self.scene_config.partition_size,
            n_cameras=self.camera_centers.shape[0],
            point_getter=point_getter,
            device=device,
        )
        return self.camera_visibilities

    def calculate_point_bounding_box_based_camera_visibilities(self, point_getter: Callable, device):
        self.camera_visibilities = Partitioning.cameras_point_bounding_box_based_visibilities_calculation(
            partition_coordinates=self.partition_coordinates,
            size=self.scene_config.partition_size,
            n_cameras=self.camera_centers.shape[0],
            point_getter=point_getter,
            device=device,
        )
        return self.camera_visibilities

    def visibility_based_partition_assignment(self):
        self.is_partitions_visible_to_cameras = Partitioning.visibility_based_partition_assignment(
            partition_coordinates=self.partition_coordinates,
            camera_centers=self.camera_centers,
            size=self.scene_config.partition_size,
            max_distance=self.scene_config.visibility_based_distance,
            assigned_mask=self.is_camera_in_partition,
            visibilities=self.camera_visibilities,
            visibility_threshold=self.scene_config.visibility_threshold,
        )
        return self.is_partitions_visible_to_cameras

    def build_output_dirname(self):
        return "partitions-size_{}-enlarge_{}-{}_visibility_{}_{}".format(
            self.scene_config.partition_size,
            self.scene_config.location_based_enlarge,
            "bbox" if self.scene_config.bounding_box_based_visibility else "point",
            self.scene_config.visibility_based_distance,
            round(self.scene_config.visibility_threshold, 2),
        )

    def save(self, output_dir: str, extra_data=None):
        return Partitioning.save_partitions(
            output_dir,
            scene_config=self.scene_config,
            scene_bounding_box=self.scene_bounding_box,
            partition_coordinates=self.partition_coordinates,
            visibilities=self.camera_visibilities,
            location_based_assignments=self.is_camera_in_partition,
            visibility_based_assignments=self.is_partitions_visible_to_cameras,
            extra_data=extra_data,
        )

    def set_plot_ax_limit(self, ax, plot_enlarge: float = 0.25):
        ax.set_xlim([
            self.scene_bounding_box.bounding_box.min[0] - plot_enlarge * self.scene_config.partition_size,
            self.scene_bounding_box.bounding_box.max[0] + plot_enlarge * self.scene_config.partition_size,
        ])
        ax.set_ylim([
            self.scene_bounding_box.bounding_box.min[1] - plot_enlarge * self.scene_config.partition_size,
            self.scene_bounding_box.bounding_box.max[1] + plot_enlarge * self.scene_config.partition_size,
        ])

    def plot_scene_bounding_box(self, ax):
        ax.set_aspect('equal', adjustable='box')

        ax.scatter(self.camera_centers[:, 0], self.camera_centers[:, 1], s=0.2)
        ax.add_artist(mpatches.Rectangle(
            self.scene_bounding_box.bounding_box.min.tolist(),
            self.scene_bounding_box.bounding_box.max[0] - self.scene_bounding_box.bounding_box.min[0],
            self.scene_bounding_box.bounding_box.max[1] - self.scene_bounding_box.bounding_box.min[1],
            fill=False,
            color="green",
        ))
        self.set_plot_ax_limit(ax)

    def plot_partitions(self, ax=None, annotate_font_size: int = 5, annotate_position_x: float = 0.125, annotate_position_y: float = 0.25):
        self.set_plot_ax_limit(ax)
        ax.set_aspect('equal', adjustable='box')

        colors = list(iter(cm.rainbow(np.linspace(0, 1, len(self.partition_coordinates)))))
        # random.shuffle(colors)
        ax.scatter(self.camera_centers[:, 0], self.camera_centers[:, 1], s=0.2)

        color_iter = iter(colors)
        idx = 0
        for partition_id, partition_xy in self.partition_coordinates:
            ax.add_artist(mpatches.Rectangle(
                (partition_xy[0], partition_xy[1]),
                self.scene_config.partition_size,
                self.scene_config.partition_size,
                fill=False,
                color=next(color_iter),
            ))
            ax.annotate(
                "#{}\n({}, {})".format(idx, partition_id[0], partition_id[1]),
                xy=(
                    partition_xy[0] + annotate_position_x * self.scene_config.partition_size,
                    partition_xy[1] + annotate_position_y * self.scene_config.partition_size,
                ),
                fontsize=annotate_font_size,
            )
            idx += 1

    def plot_partition_assigned_cameras(self, ax, partition_idx: int, point_xyzs: torch.Tensor = None, point_rgbs: torch.Tensor = None, point_size: float = 0.1, point_sparsify: int = 8):
        location_base_assignment_bounding_boxes = self.partition_coordinates.get_bounding_boxes(
            size=self.scene_config.partition_size,
            enlarge=self.scene_config.location_based_enlarge,
        )
        location_base_assignment_bounding_box_sizes = location_base_assignment_bounding_boxes.max - location_base_assignment_bounding_boxes.min

        visibility_base_assignment_bounding_boxes = self.partition_coordinates.get_bounding_boxes(
            size=self.scene_config.partition_size,
            enlarge=self.scene_config.visibility_based_distance,
        )
        visibility_base_assignment_bounding_box_size = visibility_base_assignment_bounding_boxes.max - visibility_base_assignment_bounding_boxes.min

        location_based_assignment = self.is_camera_in_partition[partition_idx]
        visibility_based_assignment = self.is_partitions_visible_to_cameras[partition_idx]
        location_based_assignment.nonzero().squeeze(-1), visibility_based_assignment.nonzero().squeeze(-1)

        ax.set_aspect('equal', adjustable='box')
        if point_xyzs is not None:
            ax.scatter(point_xyzs[::point_sparsify, 0], point_xyzs[::point_sparsify, 1], c=point_rgbs[::point_sparsify] / 255., s=point_size)

        # plot original partition bounding box
        ax.add_artist(mpatches.Rectangle(
            self.partition_coordinates.xy[partition_idx],
            self.scene_config.partition_size,
            self.scene_config.partition_size,
            fill=False,
            color="green",
        ))

        # plot location based assignment bounding box
        ax.add_artist(mpatches.Rectangle(
            location_base_assignment_bounding_boxes.min[partition_idx],
            location_base_assignment_bounding_box_sizes[partition_idx, 0].item(),
            location_base_assignment_bounding_box_sizes[partition_idx, 1].item(),
            fill=False,
            color="purple",
        ))

        # plot visibility based assignment bounding box
        ax.add_artist(mpatches.Rectangle(
            visibility_base_assignment_bounding_boxes.min[partition_idx],
            visibility_base_assignment_bounding_box_size[partition_idx, 0].item(),
            visibility_base_assignment_bounding_box_size[partition_idx, 1].item(),
            fill=False,
            color="yellow",
        ))

        # plot assigned cameras
        ax.scatter(self.camera_centers[location_based_assignment.numpy(), 0],
                   self.camera_centers[location_based_assignment.numpy(), 1], s=0.2, c="blue")
        ax.scatter(self.camera_centers[visibility_based_assignment.numpy(), 0],
                   self.camera_centers[visibility_based_assignment.numpy(), 1], s=0.2, c="red")

        ax.annotate(
            text="#{} ({}, {})".format(
                partition_idx,
                self.partition_coordinates.id[partition_idx, 0].item(),
                self.partition_coordinates.id[partition_idx, 1].item(),
            ),
            xy=self.partition_coordinates.xy[partition_idx] + 0.05 * self.scene_config.partition_size,
            color="orange",
        )

        self.set_plot_ax_limit(ax)

    def plot(self, func: Callable, *args, **kwargs):
        plt.close()
        fig, ax = plt.subplots()
        func(ax, *args, **kwargs)
        plt.show(fig)

    def save_plot(self, func: Callable, path: str, *args, **kwargs):
        plt.close()
        fig, ax = plt.subplots()
        func(ax, *args, **kwargs)
        plt.savefig(path, dpi=600)
        plt.show(fig)


class Partitioning:
    @staticmethod
    def get_bounding_box_by_camera_centers(camera_centers: torch.Tensor, enlarge: float = 0.) -> MinMaxBoundingBox:
        xyz_min = torch.min(camera_centers, dim=0)[0]
        xyz_min -= xyz_min * enlarge

        xyz_max = torch.max(camera_centers, dim=0)[0]
        xyz_max += xyz_max * enlarge

        return MinMaxBoundingBox(
            min=xyz_min,
            max=xyz_max,
        )

    @staticmethod
    def align_xyz(xyz, origin: torch.Tensor, size: float):
        xyz_radius_factor = (xyz - origin) / size
        n_partitions = torch.ceil(torch.abs(xyz_radius_factor)).to(torch.long)
        xyz_ceil_radius_factor = n_partitions * torch.sign(xyz_radius_factor)
        new_xyz = origin + xyz_ceil_radius_factor * size
        return new_xyz, n_partitions

    @classmethod
    def align_bounding_box(cls, bounding_box: MinMaxBoundingBox, origin: torch.Tensor, size: float):
        # TODO: if center out of bounding box, incorrect result will be produced
        assert torch.all(origin >= bounding_box.min), "center {} out-of-min-bound {}".format(origin, bounding_box.min)
        assert torch.all(origin <= bounding_box.max), "center {} out-of-max-bound {}".format(origin, bounding_box.max)

        new_min, n1 = cls.align_xyz(bounding_box.min, origin, size)
        new_max, n2 = cls.align_xyz(bounding_box.max, origin, size)

        aligned = MinMaxBoundingBox(
            min=new_min,
            max=new_max,
        )

        return SceneBoundingBox(
            bounding_box=aligned,
            n_partitions=n1 + n2,
            origin_partition_offset=-n1,
        )

    @staticmethod
    def build_partition_coordinates(scene_bounding_box: SceneBoundingBox, origin: torch.Tensor, size: float):
        partition_count = scene_bounding_box.n_partitions
        origin_partition_offset = scene_bounding_box.origin_partition_offset
        grid_x, grid_y = torch.meshgrid(
            torch.arange(partition_count[0], dtype=torch.int) + origin_partition_offset[0],
            torch.arange(partition_count[1], dtype=torch.int) + origin_partition_offset[1],
            indexing="xy",
        )

        partition_id = torch.dstack([grid_x, grid_y])
        partition_xy = partition_id * size + origin

        return PartitionCoordinates(
            id=partition_id.reshape(torch.prod(partition_count), 2),
            xy=partition_xy.reshape(torch.prod(partition_count), 2),
        )

    @staticmethod
    def is_in_bounding_boxes(
            bounding_boxes: MinMaxBoundingBoxes,
            coordinates: torch.Tensor,  # [N, 2]
    ):
        xy_min = bounding_boxes.min  # [N_partitions, 2]
        xy_max = bounding_boxes.max  # [N_partitions, 2]

        coordinates = coordinates.unsqueeze(0)  # [1, N_coordinates, 2]
        xy_min = xy_min.unsqueeze(1)  # [N_partitions, 1, 2]
        xy_max = xy_max.unsqueeze(1)  # [N_partitions, 1, 2]

        is_gt_min = torch.prod(torch.ge(coordinates, xy_min), dim=-1)  # [N_partitions, N_coordinates]
        is_le_max = torch.prod(torch.le(coordinates, xy_max), dim=-1)  # [N_partitions, N_coordinates]
        is_in_partition = is_gt_min * is_le_max != 0  # [N_partitions, N_coordinates]

        return is_in_partition

    @classmethod
    def camera_center_based_partition_assignment(
            cls,
            partition_coordinates: PartitionCoordinates,
            camera_centers: torch.Tensor,  # [N_images, 2]
            size: float,
            enlarge: float = 0.1,
    ) -> torch.Tensor:
        assert enlarge >= 0.

        bounding_boxes = partition_coordinates.get_bounding_boxes(
            size=size,
            enlarge=enlarge,
        )
        return cls.is_in_bounding_boxes(
            bounding_boxes=bounding_boxes,
            coordinates=camera_centers,
        )

    @classmethod
    def cameras_point_based_visibilities_calculation(
            cls,
            partition_coordinates: PartitionCoordinates,
            size: float,
            n_cameras,
            point_getter: Callable[[int], torch.Tensor],
            device,
    ):
        partition_bounding_boxes = partition_coordinates.get_bounding_boxes(size, enlarge=0.).to(device=device)
        all_visibilities = torch.ones((n_cameras, len(partition_coordinates)), device="cpu") * -255.  # [N_cameras, N_partitions]

        def calculate_visibilities(camera_idx: int):
            points_3d = point_getter(camera_idx)
            visibilities, _ = Partitioning.calculate_point_based_visibilities(
                partition_bounding_boxes=partition_bounding_boxes,
                points=points_3d[..., :2],
            )  # [N_partitions]
            all_visibilities[camera_idx].copy_(visibilities.to(device=all_visibilities.device))

        from concurrent.futures.thread import ThreadPoolExecutor
        with ThreadPoolExecutor() as tpe:
            for _ in tqdm(
                    tpe.map(calculate_visibilities, range(n_cameras)),
                    total=n_cameras,
            ):
                pass

        assert torch.all(all_visibilities >= 0.)

        return all_visibilities.T  # [N_partitions, N_cameras]

    @classmethod
    def cameras_point_bounding_box_based_visibilities_calculation(
            cls,
            partition_coordinates: PartitionCoordinates,
            size: float,
            n_cameras,
            point_getter: Callable[[int], Tuple[torch.Tensor, torch.Tensor, int]],
            device,
    ):
        partition_bounding_boxes = partition_coordinates.get_bounding_boxes(size, enlarge=0.).to(device=device)
        all_visibilities = torch.ones((n_cameras, len(partition_coordinates)), device="cpu") * -255.  # [N_cameras, N_partitions]

        def calculate_visibilities(camera_idx: int):
            points_2d, points_3d, n_pixels = point_getter(camera_idx)
            visibilities, _, _ = Partitioning.calculate_point_bounding_box_based_visibilities(
                partition_bounding_boxes=partition_bounding_boxes,
                points_2d=points_2d,
                points_3d=points_3d[..., :2],
                n_pixels=n_pixels,
            )  # [N_partitions]
            all_visibilities[camera_idx].copy_(visibilities.to(device=all_visibilities.device))

        from concurrent.futures.thread import ThreadPoolExecutor
        with ThreadPoolExecutor() as tpe:
            for _ in tqdm(
                    tpe.map(calculate_visibilities, range(n_cameras)),
                    total=n_cameras,
            ):
                pass

        assert torch.all(all_visibilities >= 0.)

        return all_visibilities.T  # [N_partitions, N_cameras]

    @classmethod
    def visibility_based_partition_assignment(
            cls,
            partition_coordinates: PartitionCoordinates,
            camera_centers: torch.Tensor,
            size: float,
            max_distance: float,
            assigned_mask: torch.Tensor,  # the Tensor produced by `camera_center_based_partition_assignment` above
            visibilities: torch.Tensor,  # [N_partitions, N_cameras]
            visibility_threshold: float,
    ):
        is_in_range = cls.camera_center_based_partition_assignment(
            partition_coordinates,
            camera_centers,
            size,
            max_distance,
        )  # [N_partitions, N_cameras]
        # exclude assigned
        is_not_assigned = torch.logical_and(is_in_range, torch.logical_not(assigned_mask))

        is_visible = torch.ge(visibilities, visibility_threshold)  # [N_partitions, N_cameras]

        return torch.logical_and(is_visible, is_not_assigned)

    @classmethod
    def calculate_point_based_visibilities(
            cls,
            partition_bounding_boxes: MinMaxBoundingBoxes,
            points: torch.Tensor,  # [N_points, 2 or 3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if points.shape[0] == 0:
            n_points_in_partitions = torch.zeros((partition_bounding_boxes.min.shape[0],), dtype=torch.long)
            visibilities = torch.zeros((partition_bounding_boxes.min.shape[0],), dtype=torch.float)
        else:
            is_in_bounding_boxes = cls.is_in_bounding_boxes(
                bounding_boxes=partition_bounding_boxes,
                coordinates=points,
            )  # [N_partitions, N_points]
            n_points_in_partitions = is_in_bounding_boxes.sum(dim=-1)  # [N_partitions]
            visibilities = n_points_in_partitions / points.shape[0]  # [N_partitions]

        return visibilities, n_points_in_partitions

    @classmethod
    def calculate_point_bounding_box_based_visibilities(
            cls,
            partition_bounding_boxes: MinMaxBoundingBoxes,
            points_2d: torch.Tensor,  # [N_points, 2]
            points_3d: torch.Tensor,  # [N_points, 2 or 3]
            n_pixels: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a bounding based on the feature points inside the partition, 
        and then calculate its ratio to the image.
        """

        if points_3d.shape[0] == 0:
            bbox_area = torch.zeros((partition_bounding_boxes.min.shape[0],), dtype=torch.long)
            visibilities = torch.zeros((partition_bounding_boxes.min.shape[0],), dtype=torch.float)
            bbox_min = torch.zeros((partition_bounding_boxes.min.shape[0], 2), dtype=torch.long)
            bbox_max = bbox_min
        else:
            is_in_bounding_boxes = cls.is_in_bounding_boxes(
                bounding_boxes=partition_bounding_boxes,
                coordinates=points_3d,
            )  # [N_partitions, N_points]

            # get bounding boxes
            not_in_bounding_boxes = ~is_in_bounding_boxes[:, :, None]  # [N_partitions, N_points, 1]
            not_in_partition_offset = not_in_bounding_boxes * n_pixels  # avoid selecting points not in the partition
            bbox_min = torch.min(points_2d[None, :, :] + not_in_partition_offset, dim=1).values  # [N_partitions, 2]
            bbox_max = torch.max(points_2d[None, :, :] - not_in_partition_offset, dim=1).values  # [N_partitions, 2]

            # calculate bounding box sizes
            bbox_size = bbox_max - bbox_min  # [N_partitions, 2]
            bbox_area = torch.prod(bbox_size, dim=-1)  # [N_partitions]

            visible_partition = is_in_bounding_boxes.sum(dim=-1) > 0
            bbox_area = bbox_area * visible_partition
            bbox_min = bbox_min * visible_partition[:, None]
            bbox_max = bbox_max * visible_partition[:, None]

            visibilities = bbox_area / n_pixels  # [N_partitions]

        return visibilities, bbox_area, (bbox_min, bbox_max)

    @classmethod
    def save_partitions(
            cls,
            dir: str,
            scene_config: SceneConfig,
            scene_bounding_box: SceneBoundingBox,
            partition_coordinates: PartitionCoordinates,
            visibilities: torch.Tensor,  # [N_partitions, N_cameras]
            location_based_assignments: torch.Tensor,  # [N_partitions, N_cameras]
            visibility_based_assignments: torch.Tensor,  # [N_partitions, N_cameras]
            extra_data=None,
    ):
        os.makedirs(dir, exist_ok=True)
        save_to = os.path.join(dir, "partitions.pt")
        torch.save(
            {
                "scene_config": dataclasses.asdict(scene_config),
                "scene_bounding_box": dataclasses.asdict(scene_bounding_box),
                "partition_coordinates": dataclasses.asdict(partition_coordinates),
                "visibilities": visibilities,
                "location_based_assignments": location_based_assignments,
                "visibility_based_assignments": visibility_based_assignments,
                "extra_data": extra_data,
            },
            save_to,
        )
        return save_to

    @classmethod
    def partition_id_to_str(cls, id):
        x, y = id[0], id[1]
        if isinstance(id, torch.Tensor):
            x = x.item()
            y = y.item()

        return "{:03d}_{:03d}".format(x, y)
