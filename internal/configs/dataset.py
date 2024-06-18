from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class ColmapParams:
    """
        Args:
            image_dir: the path to the directory that store images

            mask_dir:
                the path to the directory store mask files;
                the mask file of the image `a/image_name.jpg` is `a/image_name.jpg.png`;
                single channel, 0 is the masked pixel;

            split_mode: reconstruction: train model use all images; experiment: withholding a test set for evaluation

            eval_step: -1: use all images as training set; > 1: pick an image for every eval_step

            reorient: whether reorient the scene

            appearance_groups: filename without extension
    """

    image_dir: str = None

    mask_dir: str = None

    split_mode: Literal["reconstruction", "experiment"] = "reconstruction"

    eval_image_select_mode: Literal["step", "ratio"] = "step"

    eval_step: int = 8

    eval_ratio: float = 0.01

    scene_scale: float = 1.

    reorient: bool = False  # TODO

    appearance_groups: Optional[str] = None

    image_list: Optional[str] = None

    down_sample_factor: int = 1

    down_sample_rounding_model: Literal["floor", "round", "ceil"] = "round"


@dataclass
class BlenderParams:
    white_background: bool = False

    random_point_color: bool = False

    split_mode: Literal["reconstruction", "experiment"] = "experiment"


@dataclass
class NSVFParams(BlenderParams):
    pass


@dataclass
class NerfiesParams:
    down_sample_factor: int = 1

    undistort_image: bool = True

    step: int = 1

    split_mode: Literal["reconstruction", "experiment"] = "experiment"

    eval_image_select_mode: Literal["step"] = "step"

    eval_step: int = 16


@dataclass
class MatrixCityParams:
    train: list[str] = None

    test: list[str] = None

    scale: float = 0.01

    depth_scale: float = 0.01

    max_depth: float = 65_000
    """ Using to remove sky, multiply with scale and depth_scale automatically """

    depth_read_step: int = 1

    max_points: int = 3_840_000


@dataclass
class PhotoTourismParams(ColmapParams):
    pass


@dataclass
class DatasetParams:
    """
        Args:
            train_max_num_images_to_cache: limit the max num images to be load at the same time

            val_max_num_images_to_cache: limit the max num images to be load at the same time
    """

    colmap: ColmapParams

    colmap_joint: ColmapParams

    blender: BlenderParams

    nsvf: NSVFParams

    nerfies: NerfiesParams

    matrix_city: MatrixCityParams

    phototourism: PhotoTourismParams

    image_scale_factor: float = 1.  # TODO

    train_max_num_images_to_cache: int = -1

    val_max_num_images_to_cache: int = 0

    test_max_num_images_to_cache: int = 0

    num_workers: int = 8

    add_background_sphere: bool = False

    use_data_sampler: bool = False

    background_sphere_distance: float = 2.2

    background_sphere_points: int = 204_800
