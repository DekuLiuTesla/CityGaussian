import add_pypath
import os
import sys
import argparse
import subprocess
import numpy as np
import torch
import sqlite3
from internal.utils import colmap


def array_to_blob(array):
    return array.tostring()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--vocab_tree_path", "-v", type=str,
                        default=os.path.expanduser("~/.cache/colmap/vocab_tree_flickr100K_words256K.bin"))
    parser.add_argument("--refine", action="store_true",
                        help="Refine intrinsics and extrinsics")
    args = parser.parse_args()

    assert os.path.exists(args.vocab_tree_path), "Vocabulary Tree not found, please download it from 'https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin', and provide it path via option '-v'"

    return args


def main():
    args = parse_args()

    camera_model = "PINHOLE"
    if args.refine:
        camera_model = "OPENCV"

    coordinates = torch.load(os.path.join(args.path, "coordinates.pt"), map_location="cpu")

    colmap_dir = os.path.join(args.path, "colmap")
    os.makedirs(colmap_dir, exist_ok=True)

    image_metadata_pairs = []
    for split in ["train", "val"]:
        for i in os.scandir(os.path.join(args.path, split, "rgbs")):
            name_without_ext = i.name.split(".")[0]
            image_metadata_pairs.append((
                i.path,
                os.path.join(args.path, split, "metadata", "{}.pt".format(name_without_ext)),
                i.name,
                split,
            ))

    image_dir = os.path.join(colmap_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    for i, _, image_name, split in image_metadata_pairs:
        try:
            os.symlink(os.path.join("..", "..", split, "rgbs", image_name), os.path.join(image_dir, image_name))
        except FileExistsError:
            pass

    colmap_db_path = os.path.join(colmap_dir, "colmap.db")
    assert subprocess.call([
        "colmap",
        "feature_extractor",
        "--database_path", colmap_db_path,
        "--image_path", image_dir,
        "--ImageReader.camera_model", camera_model,
    ]) == 0

    colmap_db = sqlite3.connect(colmap_db_path)

    def select_image(image_name: str):
        cur = colmap_db.cursor()
        try:
            return cur.execute("SELECT image_id, camera_id FROM images WHERE name = ?", [image_name]).fetchone()
        finally:
            cur.close()

    def set_image_camera_id(image_id: int, camera_id: int):
        cur = colmap_db.cursor()
        try:
            cur.execute("UPDATE images SET camera_id = ? WHERE image_id = ?", [camera_id, image_id])
            colmap_db.commit()
        finally:
            cur.close()

    def update_camera_params(camera_id: int, params: np.ndarray):
        cur = colmap_db.cursor()
        try:
            cur.execute("UPDATE cameras SET params = ? WHERE camera_id = ?", [
                array_to_blob(params),
                camera_id,
            ])
            colmap_db.commit()
        finally:
            cur.close()

    def delete_unused_cameras():
        cur = colmap_db.cursor()
        try:
            cur.execute("DELETE FROM cameras WHERE camera_id NOT IN (SELECT camera_id FROM images)")
            colmap_db.commit()
        finally:
            cur.close()

    camera_intrinsics_to_camera_id = {}

    images = {}
    cameras = {}
    points = {}

    c2w_transform = torch.tensor([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float).T
    RDF_TO_DRB_H = torch.tensor([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float)
    for _, metadata_path, image_name, _ in image_metadata_pairs:
        metadata = torch.load(metadata_path, map_location="cpu")
        image_id, camera_id = select_image(image_name)

        # share intrinsics if possible
        intrinsics_as_dict_key = metadata["intrinsics"]
        intrinsics_as_dict_key = torch.concat(
            [intrinsics_as_dict_key, torch.tensor([metadata["W"], metadata["H"]], dtype=torch.float)],
            dim=-1,
        )
        camera_id = camera_intrinsics_to_camera_id.setdefault(intrinsics_as_dict_key.numpy().tobytes(), camera_id)
        set_image_camera_id(image_id, camera_id)

        c2w = torch.eye(4)
        c2w[:3, :] = metadata["c2w"]

        c2w[:3, 3] *= coordinates["pose_scale_factor"]
        c2w[:3, 3] += coordinates["origin_drb"]

        c2w = torch.linalg.inv(RDF_TO_DRB_H) @ c2w @ c2w_transform @ RDF_TO_DRB_H
        w2c = torch.linalg.inv(c2w)

        images[image_id] = colmap.Image(
            image_id,
            qvec=colmap.rotmat2qvec(w2c[:3, :3].numpy()),
            tvec=w2c[:3, 3].numpy(),
            camera_id=camera_id,
            name=image_name,
            xys=np.array([], dtype=np.float64),
            point3D_ids=np.asarray([], dtype=np.int64),
        )

        # a new camera
        if camera_id not in cameras:
            camera_params = metadata["intrinsics"]
            if args.refine:
                camera_params = torch.concat([metadata["intrinsics"], torch.tensor([0., 0., 0., 0.])], dim=-1)
            update_camera_params(camera_id, camera_params.to(torch.float64).numpy())
            cameras[camera_id] = colmap.Camera(
                id=camera_id,
                model=camera_model,
                width=metadata["W"],
                height=metadata["H"],
                params=camera_params.to(torch.float64).numpy(),
            )

    delete_unused_cameras()

    colmap_db.close()

    sparse_manually_model_dir = os.path.join(colmap_dir, "sparse_manually")
    os.makedirs(sparse_manually_model_dir, exist_ok=True)
    colmap.write_images_binary(images, os.path.join(sparse_manually_model_dir, "images.bin"))
    colmap.write_cameras_binary(cameras, os.path.join(sparse_manually_model_dir, "cameras.bin"))
    colmap.write_points3D_binary(points, os.path.join(sparse_manually_model_dir, "points3D.bin"))

    assert subprocess.call([
        "colmap",
        "vocab_tree_matcher",
        "--database_path", colmap_db_path,
        "--VocabTreeMatching.vocab_tree_path", args.vocab_tree_path,
    ]) == 0

    sparse_dir_triangulated = os.path.join(colmap_dir, "sparse_triangulated")
    os.makedirs(sparse_dir_triangulated, exist_ok=True)
    assert subprocess.call([
        "colmap",
        "point_triangulator",
        "--database_path", colmap_db_path,
        "--image_path", image_dir,
        "--input_path", sparse_manually_model_dir,
        "--output_path", sparse_dir_triangulated,
    ]) == 0

    if args.refine:
        # use the intrinsics and extrinsics provided by MegaNeRF will produce a suboptimal result,
        # so run a bundle adjustment to further refine them

        sparse_dir = os.path.join(colmap_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)
        assert subprocess.call([
            "colmap",
            "bundle_adjuster",
            "--input_path", sparse_dir_triangulated,
            "--output_path", sparse_dir,
        ]) == 0

        dense_dir = os.path.join(colmap_dir, "dense")
        os.makedirs(dense_dir, exist_ok=True)
        assert subprocess.call([
            "colmap",
            "image_undistorter",
            "--image_path", image_dir,
            "--input_path", sparse_dir,
            "--output_path", dense_dir,
            "--max_image_size", "1600",
        ]) == 0
        print("Saved to '{}', use this as your dataset path".format(dense_dir))
    else:
        os.rename(sparse_dir_triangulated, os.path.join(colmap_dir, "sparse"))
        print("Saved to '{}', use this as your dataset path".format(colmap_dir))


def test_main():
    sys.argv = [__file__, os.path.expanduser("~/data/Mega-NeRF/rubble-pixsfm")]
    main()


if __name__ == "__main__":
    main()
