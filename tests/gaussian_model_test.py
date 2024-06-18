import os.path
import random
import unittest
import numpy as np
import torch

from internal.configs.optimization import OptimizationParams
from internal.utils.graphics_utils import BasicPointCloud
from internal.models.gaussian_model import GaussianModel
from internal.utils.sh_utils import eval_sh
from internal.models.gaussian_model_simplified import GaussianModelSimplified


class GaussianModelTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # random.seed(42)
        # np.random.seed(42)
        # torch.random.manual_seed(42)

    def _generate_point_cloud(self, num_points: int):
        return BasicPointCloud(
            points=(np.random.rand(num_points, 3).astype(np.float32) - 0.5) * 10,
            colors=np.linspace(0, 1, num_points * 3, dtype=np.float32).reshape(num_points, 3),
            normals=np.zeros((num_points, 3)),
        )

    def test_initialization_and_scheduler(self):
        device = torch.device("cuda")

        # initialization point cloud
        num_points = 256
        pcd = self._generate_point_cloud(num_points)

        # with extra features
        model = GaussianModel(sh_degree=3, extra_feature_dims=64)
        model.create_from_pcd(pcd, device)
        with torch.no_grad():
            # check xyz
            self.assertTrue(torch.allclose(model.get_xyz.cpu(), torch.from_numpy(pcd.points)))

            # check color
            color_from_model = (eval_sh(3, model.get_features.transpose(1, 2), torch.zeros((num_points, 3), device=device)) + 0.5).cpu()
            self.assertTrue(torch.allclose(
                color_from_model,
                torch.from_numpy(pcd.colors),
                atol=1e-7,
            ))
            # check features_rest are all 0
            self.assertTrue(torch.all(model.get_features[:, 1:, :] == 0))

            # check has extra feature
            self.assertEqual(model.get_features_extra.shape[-1], 64)

            # check default sh_degree
            self.assertEqual(model.active_sh_degree, 0)

            # check extra feature works during updating, densification and pruning
            new_extra_feature_values = torch.rand_like(model._features_extra)
            model._features_extra.copy_(new_extra_feature_values)
            self.assertTrue(torch.all(model.get_features_extra == new_extra_feature_values))
            self.assertTrue(torch.all(model.get_features[:, 1:, :] == 0))  # check features_rest are still all 0

        # check scheduler
        optimization_params = OptimizationParams(feature_rest_lr_init=random.random(), feature_extra_lr_final_factor=random.random())
        model.training_setup(optimization_params, 10.)
        target_lr_init = {
            "xyz": optimization_params.position_lr_init * 10.,
            "f_dc": optimization_params.feature_lr,
            "f_rest": optimization_params.feature_rest_lr_init,
            "opacity": optimization_params.opacity_lr,
            "scaling": optimization_params.scaling_lr,
            "rotation": optimization_params.rotation_lr,
            "f_extra": optimization_params.feature_extra_lr_init,
        }
        for idx, param_group in enumerate(model.optimizer.param_groups):
            self.assertEqual(param_group["lr"], target_lr_init[param_group["name"]])

        model.update_learning_rate(30000)
        target_lr_final = {
            "xyz": optimization_params.position_lr_final * 10.,
            "f_dc": optimization_params.feature_lr,
            "f_rest": optimization_params.feature_rest_lr_init,  # the lr of f_rest should not be updated because max_steps=-1
            "opacity": optimization_params.opacity_lr,
            "scaling": optimization_params.scaling_lr,
            "rotation": optimization_params.rotation_lr,
            "f_extra": optimization_params.feature_extra_lr_init * optimization_params.feature_extra_lr_final_factor,
        }
        for idx, param_group in enumerate(model.optimizer.param_groups):
            self.assertTrue(np.isclose(param_group["lr"], target_lr_final[param_group["name"]]))

        # check with f_rest scheduler
        optimization_params = OptimizationParams(feature_rest_lr_init=random.random(), feature_rest_lr_final_factor=random.random(), feature_rest_lr_max_steps=30_000)
        model.training_setup(optimization_params, 10.)
        model.update_learning_rate(30000)
        target_lr_final = {
            "xyz": optimization_params.position_lr_final * 10.,
            "f_dc": optimization_params.feature_lr,
            "f_rest": optimization_params.feature_rest_lr_init * optimization_params.feature_rest_lr_final_factor,
            "opacity": optimization_params.opacity_lr,
            "scaling": optimization_params.scaling_lr,
            "rotation": optimization_params.rotation_lr,
            "f_extra": optimization_params.feature_extra_lr_init * optimization_params.feature_extra_lr_final_factor,
        }
        for idx, param_group in enumerate(model.optimizer.param_groups):
            self.assertTrue(np.isclose(param_group["lr"], target_lr_final[param_group["name"]]))

        # assert SHs and extra features shape
        for sh_degree in range(4):
            model = GaussianModel(sh_degree=sh_degree, extra_feature_dims=2 * sh_degree)
            model.create_from_pcd(pcd, device)
            self.assertEqual(model.get_features.shape[-2], ((sh_degree + 1) ** 2))
            self.assertEqual(model.get_features_extra.shape[-1], 2 * sh_degree)

    def test_save_and_load(self):
        device = torch.device("cuda")
        num_points = 256
        pcd = self._generate_point_cloud(num_points)

        model = GaussianModel(sh_degree=3, extra_feature_dims=64)
        model.create_from_pcd(pcd, device)

        # write some random values to parameters
        scales = torch.rand((num_points, 3), dtype=torch.float, device=device)
        rotations = torch.rand((num_points, 4), dtype=torch.float, device=device)
        features_rest = torch.rand((num_points, (model.max_sh_degree + 1) ** 2 - 1, 3), dtype=torch.float, device=device)
        opacities = torch.rand((num_points, 1), dtype=torch.float, device=device)
        features_extra = torch.rand((num_points, 64), dtype=torch.float, device=device)
        with torch.no_grad():
            model._scaling.copy_(scales)
            model._rotation.copy_(rotations)
            model._features_rest.copy_(features_rest)
            model._opacity.copy_(opacities)
            model._features_extra.copy_(features_extra)
        # check values are written
        self.assertTrue(torch.all(model.get_scaling == model.scaling_activation(scales)))
        self.assertTrue(torch.all(model.get_rotation == model.rotation_activation(rotations)))
        self.assertTrue(torch.all(model.get_features[:, 1:, :] == features_rest))
        self.assertTrue(torch.all(model.get_opacity == model.opacity_activation(opacities)))
        self.assertTrue(torch.all(model.get_features_extra == features_extra))

        nn_module = torch.nn.Module()
        nn_module.gaussian_model = model

        # test create simplified model from state dict (GaussianParameterUtils and GaussianModelSimplified)
        state_dict = nn_module.state_dict()
        model_from_state_dict = GaussianModelSimplified.construct_from_state_dict(state_dict, 3, device)
        # check all the parameters are identical
        for i in ["get_scaling", "get_rotation", "get_xyz", "get_features", "get_opacity", "get_features_extra"]:
            self.assertTrue(torch.allclose(getattr(model, i), getattr(model_from_state_dict, i)))

        # test save and load ply
        output_path = os.path.join(os.path.dirname(__file__), "test_model.ply")
        model.save_ply(output_path)
        model_from_ply = GaussianModel(3, 0)
        model_from_ply.load_ply(output_path, device)
        # check all the parameters except `features_extra` are identical
        for i in ["get_scaling", "get_rotation", "get_xyz", "get_features", "get_opacity"]:
            self.assertTrue(torch.allclose(getattr(model, i), getattr(model_from_ply, i)))
        # `features_extra` is excluded in ply file
        self.assertEqual(model_from_ply.get_features_extra.shape[-1], 0)

    def test_densification_and_pruning(self):
        device = torch.device("cuda")
        num_points = 256
        pcd = self._generate_point_cloud(num_points)

        model = GaussianModel(sh_degree=3, extra_feature_dims=64)
        model.create_from_pcd(pcd, device)

        xyz_value = torch.rand_like(model.get_xyz)
        scales_value = torch.rand_like(model.get_scaling)
        rotations_value = torch.rand_like(model.get_rotation)
        features_dc_value = torch.rand_like(model._features_dc)
        features_rest_value = torch.rand_like(model._features_rest)
        opacities_value = torch.rand_like(model.get_opacity)
        features_extra_value = torch.rand_like(model.get_features_extra)

        with torch.no_grad():
            model._xyz.copy_(xyz_value)
            model._scaling.copy_(scales_value)
            model._rotation.copy_(rotations_value)
            model._features_dc.copy_(features_dc_value)
            model._features_rest.copy_(features_rest_value)
            model._opacity.copy_(opacities_value)
            model._features_extra.copy_(features_extra_value)

        model.training_setup(OptimizationParams(), 10.)

        # clone
        gaussian_to_clone = torch.rand((num_points,), device=device) > 0.6  # clone which > 0.6
        model.densify_and_clone(
            torch.ones((num_points, 1), device=device) * gaussian_to_clone.unsqueeze(-1),
            1e-5,
            1e5,  # allow clone large gaussian
        )
        num_cloned_gaussians = gaussian_to_clone.sum().item()
        # check number of the cloned gaussians
        self.assertEqual(model.get_xyz.shape[0], num_points + num_cloned_gaussians)
        # check all the new gaussians have correct parameter values
        self.assertTrue(torch.all(xyz_value[gaussian_to_clone] == model._xyz[-num_cloned_gaussians:]))
        self.assertTrue(torch.all(scales_value[gaussian_to_clone] == model._scaling[-num_cloned_gaussians:]))
        self.assertTrue(torch.all(rotations_value[gaussian_to_clone] == model._rotation[-num_cloned_gaussians:]))
        self.assertTrue(torch.all(features_dc_value[gaussian_to_clone] == model._features_dc[-num_cloned_gaussians:]))
        self.assertTrue(torch.all(features_rest_value[gaussian_to_clone] == model._features_rest[-num_cloned_gaussians:]))
        self.assertTrue(torch.all(opacities_value[gaussian_to_clone] == model._opacity[-num_cloned_gaussians:]))
        self.assertTrue(torch.all(features_extra_value[gaussian_to_clone] == model._features_extra[-num_cloned_gaussians:]))

        # split and prune
        xyz_value = model._xyz.detach().clone()
        scales_value = model._scaling.detach().clone()
        rotations_value = model._rotation.detach().clone()
        features_dc_value = model._features_dc.detach().clone()
        features_rest_value = model._features_rest.detach().clone()
        opacities_value = model._opacity.detach().clone()
        features_extra_value = model._features_extra.detach().clone()

        current_num_points = xyz_value.shape[0]
        gaussian_to_split = torch.rand((current_num_points,), device=device) > 0.6  # clone which > 0.6
        model.densify_and_split(
            torch.ones((current_num_points, 1), device=device) * gaussian_to_split.unsqueeze(-1),
            1e-5,
            0.,  # allow split small gaussian
        )
        num_split_gaussians = gaussian_to_split.sum().item()
        # check number of the split gaussians
        self.assertEqual(model.get_xyz.shape[0], current_num_points + num_split_gaussians)
        # check all the new gaussians have correct parameter values
        # TODO: check xyz and scale
        # self.assertTrue(torch.all(xyz_value[gaussian_to_split] == model._xyz[-num_cloned_gaussians:]))
        # self.assertTrue(torch.all(scales_value[gaussian_to_split] == model._scaling[-num_cloned_gaussians:]))
        # the first part
        self.assertTrue(torch.all(rotations_value[gaussian_to_split] == model._rotation[-num_split_gaussians:]))
        self.assertTrue(torch.all(features_dc_value[gaussian_to_split] == model._features_dc[-num_split_gaussians:]))
        self.assertTrue(torch.all(features_rest_value[gaussian_to_split] == model._features_rest[-num_split_gaussians:]))
        self.assertTrue(torch.all(opacities_value[gaussian_to_split] == model._opacity[-num_split_gaussians:]))
        self.assertTrue(torch.all(features_extra_value[gaussian_to_split] == model._features_extra[-num_split_gaussians:]))
        # the second part
        self.assertTrue(torch.all(rotations_value[gaussian_to_split] == model._rotation[-2 * num_split_gaussians:-num_split_gaussians]))
        self.assertTrue(torch.all(features_dc_value[gaussian_to_split] == model._features_dc[-2 * num_split_gaussians:-num_split_gaussians]))
        self.assertTrue(torch.all(features_rest_value[gaussian_to_split] == model._features_rest[-2 * num_split_gaussians:-num_split_gaussians]))
        self.assertTrue(torch.all(opacities_value[gaussian_to_split] == model._opacity[-2 * num_split_gaussians:-num_split_gaussians]))
        self.assertTrue(torch.all(features_extra_value[gaussian_to_split] == model._features_extra[-2 * num_split_gaussians:-num_split_gaussians]))


if __name__ == '__main__':
    unittest.main()
