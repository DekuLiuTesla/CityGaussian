import os.path
import unittest
import json
import torch
from internal.dataparsers.blender_dataparser import Blender


class BlenderDataparserTestCase(unittest.TestCase):
    def test_blender_dataparser(self):
        dataset_path = os.path.expanduser("~/data/nerf/nerf_synthetic/lego")

        gt_camera_sets = []
        for i in ["train", "val", "test"]:
            with open(os.path.join(dataset_path, "transforms_{}.json".format(i)), "r") as f:
                gt_camera_sets.append(json.load(f))

        dataparser = Blender().instantiate(dataset_path, os.getcwd(), 0)
        dataparser_outputs = dataparser.get_outputs()

        for parsed, gt_set in zip([dataparser_outputs.train_set, dataparser_outputs.val_set, dataparser_outputs.test_set], gt_camera_sets):
            gt_c2w_list = []
            for frame in gt_set["frames"]:
                gt_c2w_list.append(frame["transform_matrix"])
            gt_c2w = torch.tensor(gt_c2w_list)

            w2c = parsed.cameras.world_to_camera.transpose(1, 2)
            c2w = torch.linalg.inv(w2c)
            c2w[:, :3, 1:3] *= -1

            self.assertTrue(torch.allclose(gt_c2w, c2w, atol=3e-7))

            self.assertTrue(torch.all(torch.isclose(parsed.cameras.fov_x, torch.tensor(gt_set["camera_angle_x"]))))
            self.assertTrue(torch.all(torch.isclose(parsed.cameras.fov_y, torch.tensor(gt_set["camera_angle_x"]))))


if __name__ == '__main__':
    unittest.main()
