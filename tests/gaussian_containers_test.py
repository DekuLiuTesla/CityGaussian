import unittest
import torch
from internal.utils.gaussian_containers import FreezableParameterDict, TensorDict


class GaussianContainersTestCase(unittest.TestCase):
    def test_freezable_parameter_dict(self):
        tensors = {
            str(i): torch.rand((3, 3))
            for i in range(16)
        }
        for i in tensors.values():
            self.assertFalse(i.requires_grad)

        parameter_dict = torch.nn.ParameterDict(tensors)
        for i in parameter_dict.values():
            self.assertTrue(isinstance(i, torch.nn.Parameter))
            self.assertTrue(i.requires_grad)

        def test_replace_existing_tensor(dict_to_test):
            KEY = "replace_test"
            self.assertEqual(dict_to_test.get(KEY, None), None)

            dict_to_test[KEY] = torch.rand((16, 16))

            # inverse its `requires_grad` state
            previous_state = dict_to_test[KEY].requires_grad
            dict_to_test[KEY].requires_grad_(not previous_state)
            new_state = dict_to_test[KEY].requires_grad
            self.assertNotEqual(previous_state, new_state)

            # the new one should retain the replaced one's state
            new_one = torch.rand((16, 16)).requires_grad_(previous_state)
            dict_to_test[KEY] = new_one
            self.assertTrue(torch.all(torch.eq(new_one, dict_to_test[KEY])))
            self.assertEqual(dict_to_test[KEY].requires_grad, new_state)

            del dict_to_test[KEY]

        # new untouched
        freezable_parameter_dict = FreezableParameterDict(parameter_dict)
        self.assertEqual(freezable_parameter_dict.keys(), parameter_dict.keys())
        for i in freezable_parameter_dict.values():
            self.assertTrue(i.requires_grad)
        freezable_parameter_dict["frozen"] = torch.rand((3, 3))
        self.assertFalse(freezable_parameter_dict["frozen"].requires_grad)
        freezable_parameter_dict["optimizable"] = torch.rand((3, 3)).requires_grad_(True)
        self.assertTrue(freezable_parameter_dict["optimizable"].requires_grad)

        test_replace_existing_tensor(freezable_parameter_dict)

        # new freeze by default
        frozen_parameter_dict = FreezableParameterDict(freezable_parameter_dict, new_requires_grad=False)
        self.assertEqual(frozen_parameter_dict.keys(), freezable_parameter_dict.keys())
        for i in frozen_parameter_dict.values():
            self.assertFalse(i.requires_grad)
        frozen_parameter_dict["freeze_new_tensor"] = torch.rand((3, 3)).requires_grad_(True)
        self.assertFalse(frozen_parameter_dict["freeze_new_tensor"].requires_grad)

        test_replace_existing_tensor(frozen_parameter_dict)

        # new optimizable by default (same as `nn.ParameterDict`)
        optimizable_parameter_dict = FreezableParameterDict(frozen_parameter_dict, new_requires_grad=True)
        self.assertEqual(optimizable_parameter_dict.keys(), frozen_parameter_dict.keys())
        for i in optimizable_parameter_dict.values():
            self.assertTrue(i.requires_grad)
        optimizable_parameter_dict["optimizable_new"] = torch.rand((3, 3))
        self.assertTrue(optimizable_parameter_dict["optimizable_new"].requires_grad)

        test_replace_existing_tensor(optimizable_parameter_dict)

    def test_tensor_dict(self):
        tensor_dict = TensorDict({"a": torch.arange(9).view(3, 3)})
        self.assertFalse(isinstance(tensor_dict["a"], torch.nn.Parameter))


if __name__ == '__main__':
    unittest.main()
