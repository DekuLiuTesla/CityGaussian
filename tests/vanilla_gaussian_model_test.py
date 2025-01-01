import unittest
import torch
from internal.models.vanilla_gaussian import VanillaGaussian, VanillaGaussianModel


class VanillaGaussianModelTestCase(unittest.TestCase):
    def test_vanilla_gaussian_model(self):
        model = VanillaGaussian().instantiate()

        N = 10240

        # test setup from number
        model.setup_from_number(N)

        # validate up_sh_degree
        self.assertEqual(model.active_sh_degree, 0)
        model.on_train_batch_end(1000, None)
        self.assertEqual(model.active_sh_degree, 1)
        model.on_train_batch_end(2000, None)
        self.assertEqual(model.active_sh_degree, 2)
        model.on_train_batch_end(3000, None)
        self.assertEqual(model.active_sh_degree, 3)
        model.on_train_batch_end(5000, None)
        self.assertEqual(model.active_sh_degree, 3)

        state_dict = model.state_dict()
        self.assertTrue(state_dict["_active_sh_degree"] == 3)
        for i in model.property_names:
            self.assertTrue("gaussians.{}".format(i) in state_dict)

        # validate SH shape
        for i in range(4):
            model = VanillaGaussian(sh_degree=i).instantiate()
            model.setup_from_number(N)
            self.assertEqual(model.shs_rest.shape, (N, (i + 1) ** 2 - 1, 3))
            self.assertEqual(model.get_shs().shape, (N, (i + 1) ** 2, 3))

        model.setup_from_pcd((torch.rand((N, 3), dtype=torch.float) - 0.5) * 10., torch.rand((N, 3)))
        # validate SHs dims
        self.assertEqual(model.get_shs().dim(), 3)
        self.assertEqual(model.get_shs().shape, (N, 16, 3))
        self.assertTrue(torch.all(torch.eq(model.get_shs(), model.get_features)))

        # validate raw properties setters and getters
        for i in model.property_names:
            self.assertTrue(torch.all(torch.eq(model.gaussians[i], getattr(model, i))))  # validate @property
            self.assertTrue(torch.all(torch.eq(model.gaussians[i], model.get_property(i))))  # validate get_property()

            # validate @.setter
            new_value = torch.randn_like(model.gaussians[i])
            setattr(model, i, new_value)
            self.assertTrue(torch.all(torch.eq(model.gaussians[i], getattr(model, i))))
            self.assertTrue(torch.all(torch.eq(model.gaussians[i], model.get_property(i))))

            # validate `set_property()`
            new_value = torch.randn_like(model.gaussians[i])
            model.set_property(i, new_value)
            self.assertTrue(torch.all(torch.eq(model.gaussians[i], getattr(model, i))))
            self.assertTrue(torch.all(torch.eq(model.gaussians[i], model.get_property(i))))

        # validate activated properties
        activation_dict = {
            "opacities": model.opacity_activation,
            "scales": model.scale_activation,
            "rotations": model.rotation_activation,
        }
        for i in model.property_names:
            activated_getter = getattr(model, f"get_{i}")
            activated_values = activated_getter()

            local_activated_value = model.gaussians[i]
            activation = activation_dict.get(i, None)
            if activation is not None:
                local_activated_value = activation(local_activated_value)

            self.assertTrue(torch.all(torch.eq(activated_values, local_activated_value)), msg=f"auto-activated and manual-activated not match: {i}")

        for l, r in [
            ("get_xyz", "get_means"),
            ("get_scaling", "get_scales"),
            ("get_rotation", "get_rotations"),
            ("get_features", "get_shs"),
            ("get_opacity", "get_opacities"),
        ]:
            # validate the values returned by two getter versions are identical
            self.assertTrue(torch.all(torch.eq(
                getattr(model, l),
                getattr(model, r)(),
            )))

            self.assertEqual(getattr(model, r)().shape[0], N)

        # TODO: validate value
        covariance3d = model.get_covariance()

        # test setup from tensors
        state_dict = model.gaussians.state_dict()
        state_dict["shs_rest"] = torch.empty((N, 0, 3))
        del state_dict["shs_dc"]
        model = VanillaGaussian(sh_degree=0).instantiate()
        unused, unmet = model.setup_from_tensors(state_dict)
        self.assertEqual(unused, [])
        self.assertEqual(unmet, ["shs_dc"])
        self.assertEqual(model.get_shs_rest().shape, (N, 0, 3))

        self.assertEqual(model.n_gaussians, N)


if __name__ == '__main__':
    unittest.main()
