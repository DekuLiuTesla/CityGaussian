import unittest
import torch
from internal.density_controllers.density_controller import Utils


class DensityControllerUtilsTest(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.generator = torch.Generator()
        self.generator.manual_seed(42)

    def get_dummy_properties(self, n: int = 10240):
        rand_kwargs = {
            "dtype": torch.float,
            "generator": self.generator,
        }

        return {
            "means": (torch.rand((n, 3), **rand_kwargs) - 0.5) * 16,
            "scales": torch.rand((n, 3), **rand_kwargs),
            "rotations": torch.rand((n, 4), **rand_kwargs),
            "opacities": torch.rand((n, 1), **rand_kwargs),
            "shs_dc": torch.rand((n, 1, 3), **rand_kwargs),
            "shs_rest": torch.rand((n, 15, 3), **rand_kwargs),
            "no_state": torch.rand((n,), **rand_kwargs),
        }

    def get_dummy_adam_optimizers(self, properties):
        from dataclasses import dataclass
        from typing import List, Dict, Any

        @dataclass
        class DummyAdamOptimizer:
            param_groups: List[Dict[str, Any]]
            state: Dict[torch.Tensor, Dict[str, torch.Tensor]]

            @classmethod
            def create(cls, names: List[str]):
                param_key_by_name = {name: torch.nn.Parameter(properties[name]) for name in names}
                return cls(
                    param_groups=[
                        {"name": name, "params": [param_key_by_name[name]]}
                        for name in names
                    ],
                    state={
                        param_key_by_name[name]: {
                            "step": torch.tensor(10_000, dtype=torch.float),
                            "exp_avg": torch.rand(param_key_by_name[name].shape, generator=self.generator),
                            "exp_avg_sq": torch.rand(param_key_by_name[name].shape, generator=self.generator),
                        } for name in names
                    }
                )

        dummy_optimizers = []
        dummy_optimizers.append(DummyAdamOptimizer.create(["means"]))
        dummy_optimizers.append(DummyAdamOptimizer.create(["scales", "rotations", "opacities", "shs_dc", "shs_rest", "no_state"]))
        del dummy_optimizers[-1].state[dummy_optimizers[-1].param_groups[-1]["params"][0]]  # delete state of `no_state`

        return dummy_optimizers

    def get_dummy_model(self, properties):
        class DummyModel:
            def __init__(self, properties):
                self.properties = properties
                self.property_names = list(properties.keys())

            def get_property(self, name):
                return self.properties[name]

        model = DummyModel(properties)

        return model

    def test_cat_tensors_to_optimizers(self):
        for i in range(2):
            properties = self.get_dummy_properties()
            optimizers = self.get_dummy_adam_optimizers(properties)

            # concat 0 and 1
            new_properties = self.get_dummy_properties()

            if i == 0:
                concated_properties = Utils.cat_tensors_to_optimizers_(new_properties, optimizers)
            elif i == 1:
                # add properties not in optimizers
                properties["notopt_first"] = torch.rand((10240, 3))
                properties["notopt_second"] = torch.rand((10240, 3))
                new_properties["notopt_first"] = torch.rand((10240, 3))
                new_properties["notopt_second"] = torch.rand((10240, 3))

                model = self.get_dummy_model(properties)

                concated_properties = Utils.cat_tensors_to_properties(new_properties, model, optimizers)
            else:
                raise RuntimeError()

            # validate returned tensors
            self.assertEqual(list(properties.keys()), list(concated_properties))
            for key in concated_properties.keys():
                self.assertTrue(torch.all(torch.eq(
                    torch.concat([properties[key], new_properties[key]], dim=0),
                    concated_properties[key],
                )))
                # self.assertTrue(concated_properties[key].requires_grad)
                self.assertTrue(isinstance(concated_properties[key], torch.nn.Parameter))

            # validate tensors in optimizers
            for opt in optimizers:
                for param_group in opt.param_groups:
                    name = param_group["name"]
                    self.assertTrue(torch.all(torch.eq(
                        param_group["params"][0],
                        torch.concat([properties[name], new_properties[name]], dim=0),
                    )))

                    # theses should not appear in optimizers
                    self.assertFalse(name.startswith("notopt_"))

    def test_prune_optimizers(self):
        for i in range(2):
            properties = self.get_dummy_properties()
            optimizers = self.get_dummy_adam_optimizers(properties)

            keep_mask = torch.rand((10240,), generator=self.generator) > 0.5

            if i == 0:
                pruned_properties = Utils.prune_optimizers_(keep_mask, optimizers)
            elif i == 1:
                properties["notopt_first"] = torch.rand((10240, 3))
                properties["notopt_second"] = torch.rand((10240, 3))
                model = self.get_dummy_model(properties)
                pruned_properties = Utils.prune_properties(keep_mask, model, optimizers)
            else:
                raise RuntimeError()

            # validate returned tensors
            self.assertEqual(list(properties.keys()), list(pruned_properties.keys()))
            for key in properties.keys():
                self.assertTrue(torch.all(torch.eq(
                    properties[key][keep_mask],
                    pruned_properties[key],
                )))

            # validate tensors in optimizers
            for opt in optimizers:
                for param_group in opt.param_groups:
                    name = param_group["name"]
                    self.assertTrue(torch.all(torch.eq(
                        param_group["params"][0],
                        properties[name][keep_mask],
                    )))

                    # theses should not appear in optimizers
                    self.assertFalse(name.startswith("notopt_"))

    def test_replace_tensors_to_optimizers(self):
        def validate(replaced_properties, optimizers, new_properties, selector=None):
            # validate returned tensors
            self.assertEqual(list(new_properties.keys()), list(replaced_properties.keys()))
            for key in replaced_properties.keys():
                self.assertTrue(torch.all(torch.eq(
                    new_properties[key],
                    replaced_properties[key],
                )))

            # validate tensors in optimizers
            for opt in optimizers:
                for param_group_idx, param_group in enumerate(opt.param_groups):
                    name = param_group["name"]
                    self.assertTrue(torch.all(torch.eq(
                        param_group["params"][0],
                        new_properties[name],
                    )))

                    # theses should not appear in optimizers
                    self.assertFalse(name.startswith("notopt_"))

                    if name == "no_state":
                        continue

                    state = opt.state[param_group["params"][0]]
                    if selector is None:
                        self.assertTrue(torch.all(torch.eq(
                            state["exp_avg"],
                            0,
                        )))
                        self.assertTrue(torch.all(torch.eq(
                            state["exp_avg_sq"],
                            0,
                        )))
                    else:
                        self.assertTrue(torch.all(torch.eq(
                            state["exp_avg"][selector],
                            0,
                        )))
                        self.assertTrue(torch.all(torch.eq(
                            state["exp_avg_sq"][selector],
                            0,
                        )))
                        self.assertTrue(torch.any(torch.not_equal(
                            state["exp_avg"][~selector],
                            0,
                        )))
                        self.assertTrue(torch.any(torch.not_equal(
                            state["exp_avg_sq"][~selector],
                            0,
                        )))

        def add_extra_properties_and_get_model(properties, new_properties, n_new: int):
            properties["notopt_first"] = torch.rand((10240, 3))
            properties["notopt_second"] = torch.rand((10240, 3))
            new_properties["notopt_first"] = torch.rand((10240, 3))
            new_properties["notopt_second"] = torch.rand((10240, 3))
            model = self.get_dummy_model(properties)

            return model

        for i in range(2):
            # without selector
            properties = self.get_dummy_properties()
            optimizers = self.get_dummy_adam_optimizers(properties)
            new_properties = self.get_dummy_properties(5120)
            if i == 0:
                replaced_properties = Utils.replace_tensors_to_optimizers_(new_properties, optimizers)
            elif i == 1:
                add_extra_properties_and_get_model(properties, new_properties, n_new=5120)
                replaced_properties = Utils.replace_tensors_to_properties(new_properties, optimizers)
            else:
                raise RuntimeError()
            validate(replaced_properties, optimizers, new_properties)

            # with selector
            properties = self.get_dummy_properties()
            optimizers = self.get_dummy_adam_optimizers(properties)
            new_properties = self.get_dummy_properties(10240)
            selector = torch.rand((10240,), generator=self.generator) > 0.5
            if i == 0:
                replaced_properties = Utils.replace_tensors_to_optimizers_(new_properties, optimizers, selector)
            elif i == 1:
                add_extra_properties_and_get_model(properties, new_properties, n_new=10240)
                replaced_properties = Utils.replace_tensors_to_properties(new_properties, optimizers, selector)
            else:
                raise RuntimeError()
            validate(replaced_properties, optimizers, new_properties, selector)


if __name__ == '__main__':
    unittest.main()
