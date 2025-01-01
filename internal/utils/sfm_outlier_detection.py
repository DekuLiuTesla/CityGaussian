from dataclasses import dataclass
from typing import Any
import os
import shutil
import numpy as np
import torch
from tqdm.auto import tqdm
from internal.utils import colmap


@dataclass
class Poses:
    gps_prior_path: str


@dataclass
class ColmapImages:
    path: str
    colmap_image_data: dict
    # colmap_point_data: dict

    id: Any
    image_name: Any
    w2c: Any
    point_ref_counters: Any
    n_points: Any

    c2w_in_gps: Any


def get_norm(v):
    if isinstance(v, torch.Tensor):
        return torch.norm(v, dim=-1)
    return np.sqrt(np.sum(np.power(v, 2), axis=-1))


def read_colmap_sparse_model(path: str, poses_from_exif: dict) -> ColmapImages:
    # read images.bin
    colmap_image_data = colmap.read_images_binary(os.path.join(path, "images.bin"))
    # read points3D.bin
    # colmap_point_data = colmap.read_points3D_binary(os.path.join(path, "points3D.bin"))
    # map from image name to image idx of GPS view
    image_name_to_gps_set_idx = {name: idx for idx, name in enumerate(poses_from_exif["image_name_list"])}

    colmap_images = ColmapImages(
        path,
        colmap_image_data,
        # colmap_point_data,
        id=[],
        image_name=[],
        w2c=[],
        point_ref_counters={},
        n_points=[],
        c2w_in_gps=[],
    )
    image_c2w_in_gps = []
    for idx, image in colmap_image_data.items():
        if image.name not in image_name_to_gps_set_idx:
            print("skip {}".format(image.name))
            continue
        colmap_images.id.append(idx)
        colmap_images.image_name.append(image.name)
        w2c = np.eye(4)
        w2c[:3, :3] = colmap.qvec2rotmat(image.qvec)
        w2c[:3, 3] = image.tvec
        colmap_images.w2c.append(w2c)
        colmap_images.n_points.append((image.point3D_ids > 0).sum())

        image_c2w_in_gps.append(poses_from_exif["c2w"][image_name_to_gps_set_idx[image.name]])

    colmap_images.id = np.asarray(colmap_images.id)
    colmap_images.image_name = np.asarray(colmap_images.image_name)
    colmap_images.w2c = np.asarray(colmap_images.w2c)
    colmap_images.c2w = np.linalg.inv(colmap_images.w2c)
    colmap_images.n_points = np.asarray(colmap_images.n_points)
    image_c2w_in_gps = np.stack(image_c2w_in_gps)

    colmap_images.c2w_in_gps = image_c2w_in_gps

    return colmap_images


def calculate_scale(
        colmap_images: ColmapImages,
        n_reference_images: int = 8,
        min_ref_distance: float = 0.4,
        std_factor: float = 0.5,
) -> float:
    reference_image_ids = np.argpartition(colmap_images.n_points, -n_reference_images)[-n_reference_images:]
    # colmap_images.image_name[reference_image_ids], reference_image_ids
    print("reference_image_ids={}".format(reference_image_ids))
    print("selected_images={}".format(colmap_images.image_name[reference_image_ids]))

    colmap_reference_image_camera_centers = colmap_images.c2w[reference_image_ids, :3, 3]
    gps_reference_image_camera_centers = colmap_images.c2w_in_gps[reference_image_ids, :3, 3]
    # colmap_reference_image_camera_centers, gps_reference_image_camera_centers

    camera_distance_indices = np.triu_indices(n_reference_images, k=1)
    # camera_distance_indices

    # calculate the distance in colmap view
    colmap_reference_image_camera_distances = get_norm(colmap_reference_image_camera_centers[:, None, :] - colmap_reference_image_camera_centers[None, :, :])[camera_distance_indices]

    # avoid selecting images at the same location
    valid_image_distance_mask = colmap_reference_image_camera_distances > min_ref_distance

    # calculate the distances in GPS view
    gps_reference_image_camera_distances = get_norm(gps_reference_image_camera_centers[:, None, :] - gps_reference_image_camera_centers[None, :, :])[camera_distance_indices]

    # calculate the sclaes
    gps_to_colmap_scales = gps_reference_image_camera_distances[valid_image_distance_mask] / colmap_reference_image_camera_distances[valid_image_distance_mask]
    print("gps_to_colmap_scales={}".format(gps_to_colmap_scales))

    # remove outliers
    scale_upper = gps_to_colmap_scales.mean() + std_factor * gps_to_colmap_scales.std()
    scale_bottom = gps_to_colmap_scales.mean() - std_factor * gps_to_colmap_scales.std()
    valid_scale_mask = np.logical_and(gps_to_colmap_scales > scale_bottom, gps_to_colmap_scales < scale_upper)
    scale_factor = gps_to_colmap_scales[valid_scale_mask].mean()

    return scale_factor


def transform_to_gps_view(
        colmap_images: ColmapImages,
        scale_factor: float,
        device,
):
    colmap_images.w2c_rescaled = colmap_images.w2c.copy()
    colmap_images.w2c_rescaled[:, :3, 3] *= scale_factor
    colmap_images.w2c_rescaled[:, :3, 3] / colmap_images.w2c[:, :3, 3]

    # calculate transform matrix
    with torch.no_grad():
        colmap_to_gps_view = torch.from_numpy(colmap_images.c2w_in_gps).to(device=device) @ torch.from_numpy(colmap_images.w2c_rescaled).to(device=device)
        # transform
        colmap_camera_center_in_gps_views = (
            (torch.from_numpy(colmap_images.c2w[:, :3, 3]).to(colmap_to_gps_view.device) * scale_factor) @
            torch.transpose(colmap_to_gps_view[:, :3, :3], 1, 2)
        ) + colmap_to_gps_view[:, :3, 3][:, None, :]

    return colmap_camera_center_in_gps_views


def filter_by_errors(
        colmap_images: ColmapImages,
        colmap_camera_center_in_gps_views: torch.Tensor,
        min_acceptable_error_limit: float,
        max_acceptable_error_limit: float,
):
    assert min_acceptable_error_limit <= max_acceptable_error_limit

    # calculate the errors
    camera_center_errors = get_norm(colmap_camera_center_in_gps_views - torch.from_numpy(colmap_images.c2w_in_gps[:, :3, 3][None, ...]).to(device=colmap_camera_center_in_gps_views.device))
    camera_center_error_means = torch.mean(camera_center_errors, axis=-1).cpu().numpy()
    camera_center_error_stds = torch.std(camera_center_errors, axis=-1).cpu().numpy()
    camera_center_error_means_with_stds = camera_center_error_means + camera_center_error_stds

    camera_center_errors = camera_center_errors.cpu().numpy()

    reference_camera_id = np.argpartition(camera_center_error_means_with_stds, 1)[0]
    print("reference_camera_id={}, image_name={}".format(
        reference_camera_id,
        colmap_images.image_name[reference_camera_id],
    ))

    # set `max_acceptable_error` based one mean and std
    max_acceptable_error = camera_center_error_means[reference_camera_id] + 1. * camera_center_errors[reference_camera_id].std()
    print("max_acceptable_error={}, mean={}, std={}".format(
        max_acceptable_error,
        camera_center_error_means[reference_camera_id],
        camera_center_error_stds[reference_camera_id],
    ))
    assert max_acceptable_error < max_acceptable_error_limit, "max_acceptable_error={}".format(max_acceptable_error)
    if max_acceptable_error < min_acceptable_error_limit:
        max_acceptable_error = min_acceptable_error_limit
        print("max_acceptable_error min clampped to {}".format(min_acceptable_error_limit))

    # filter out images with large errors
    large_error_image_mask = camera_center_errors[reference_camera_id] > max_acceptable_error
    print("large_error_images={}, n={}".format(
        sorted(colmap_images.image_name[np.nonzero(large_error_image_mask)[0]]),
        large_error_image_mask.sum(),
    ))

    return large_error_image_mask


def filter_by_number_of_3d_points(
        colmap_images: ColmapImages,
        min_points_3d: int = 128,
):
    less_point_image_mask = colmap_images.n_points < min_points_3d
    print("less_point_images={}, n={}".format(
        sorted(colmap_images.image_name[np.nonzero(less_point_image_mask)[0]]),
        less_point_image_mask.sum(),
    ))

    return less_point_image_mask


def save(
        colmap_images: ColmapImages,
        filter_out_masks: list,
        output_path: str,
):
    # build final mask
    mask = np.zeros(colmap_images.id.shape[0], dtype=bool)
    for i in filter_out_masks:
        mask = np.logical_or(mask, i)

    print("filter_out_images={}, n={}".format(
        sorted(colmap_images.image_name[mask]),
        mask.sum(-1),
    ))

    # get the colmap id of the images to be removed
    filter_out_image_comap_idx = colmap_images.id[mask]
    filter_out_image_comap_idx_set = {i: True for i in filter_out_image_comap_idx}

    # save
    def do_save(update_3d_points: bool = True):
        # build new dicts
        new_colmap_images = {}
        colmap_point_data = None
        if update_3d_points:
            colmap_point_data = colmap.read_points3D_binary(os.path.join(colmap_images.path, "points3D.bin"))
        for i, image in tqdm(colmap_images.colmap_image_data.items(), leave=False):
            if i in filter_out_image_comap_idx_set:
                if not update_3d_points:
                    continue
                # remove image from points' image_ids
                points_ids = image.point3D_ids
                for point_id in np.unique(points_ids):
                    if point_id < 0:
                        continue
                    point_3D = colmap_point_data[point_id]
                    point_image_id_mask = point_3D.image_ids != i  # True represnets preserve
                    # update `image_ids` and `point2D_idxs`
                    point_3D.image_ids = point_3D.image_ids[point_image_id_mask]
                    point_3D.point2D_idxs = point_3D.point2D_idxs[point_image_id_mask]

                    # if image_ids does not contain any images, delete this point
                    if point_3D.image_ids.shape[0] == 0:
                        # print("delete point #{}".format(point_id))
                        del colmap_point_data[point_id]
            else:
                new_colmap_images[i] = image

        os.makedirs(output_path, exist_ok=True)

        for i in ["images.bin", "cameras.bin", "points3D.bin"]:
            if os.path.exists(os.path.join(output_path, i)):
                os.unlink(os.path.join(output_path, i))

        print("Saving `images.bin`...")
        colmap.write_images_binary(new_colmap_images, os.path.join(output_path, "images.bin"))
        if update_3d_points:
            print("Saving `points3D.bin`...")
            colmap.write_points3D_binary(colmap_point_data, os.path.join(output_path, "points3D.bin"))
        else:
            print("Copying `points3D.bin`...")
            shutil.copyfile(os.path.join(colmap_images.path, "points3D.bin"), os.path.join(output_path, "points3D.bin"))
        print("Copying `cameras.bin`...")
        shutil.copyfile(os.path.join(colmap_images.path, "cameras.bin"), os.path.join(output_path, "cameras.bin"))
        
        return output_path
    return do_save


def load(
    parsed_pose_file,
    sparse_dir: str,
):
    if isinstance(parsed_pose_file, dict):
        poses_from_exif = parsed_pose_file
    else:
        poses_from_exif = np.load(parsed_pose_file, allow_pickle=True).item()

    colmap_images = read_colmap_sparse_model(
        sparse_dir,
        poses_from_exif=poses_from_exif,
    )

    return colmap_images


def filter(
        colmap_images: ColmapImages,
        output_name: str = None,
        output_path: str = None,

        n_reference_images: int = 8,
        min_ref_distance: float = 0.4,
        std_factor: float = 0.5,

        min_acceptable_error_limit: float = 8.,
        max_acceptable_error_limit: float = 64.,

        min_points_3d: int = 128,

        device="cuda",
        dtype=torch.float64,
):
    assert (output_name is None and output_path is None) is False, "`output_name` and `output_path` can not both be None"

    scale_factor = calculate_scale(
        colmap_images=colmap_images,
        n_reference_images=n_reference_images,
        min_ref_distance=min_ref_distance,
        std_factor=std_factor,
    )
    print("scale_factor={}".format(scale_factor))

    colmap_camera_centers_in_gps_view = transform_to_gps_view(
        colmap_images,
        scale_factor,
        device=device
    )
    if colmap_camera_centers_in_gps_view.dtype != dtype:
        colmap_camera_centers_in_gps_view = colmap_camera_centers_in_gps_view.to(dtype)

    error_filter_mask = filter_by_errors(
        colmap_images,
        colmap_camera_centers_in_gps_view,
        min_acceptable_error_limit=min_acceptable_error_limit,
        max_acceptable_error_limit=max_acceptable_error_limit,
    )
    point_filter_mask = filter_by_number_of_3d_points(
        colmap_images,
        min_points_3d=min_points_3d,
    )

    if output_path is None:
        output_path = "{}-{}".format(colmap_images.path.rstrip("/"), output_name)
    assert os.path.realpath(output_path) != os.path.realpath(colmap_images.path)

    return save(
        colmap_images, [
            error_filter_mask,
            point_filter_mask,
        ],
        output_path,
    )
