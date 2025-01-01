from typing import Optional, Any
import torch
from torch import nn


class FreezableParameterDict(nn.ParameterDict):
    def __init__(self, parameters: Any = None, new_requires_grad: Optional[bool] = None) -> None:
        self.new_requires_grad = new_requires_grad
        super().__init__(parameters)

    def __setitem__(self, key: str, value: Any) -> None:
        # get existing parameter's `requires_grad` state
        current_value = self.get(key, None)
        if current_value is None:
            # if key not exists, use `self.new_requires_grad`
            requires_grad = self.new_requires_grad
            # if `self.new_requires_grad` is None, get from `value`
            if requires_grad is None:
                requires_grad = value.requires_grad
        else:
            requires_grad = current_value.requires_grad

        super().__setitem__(key, value)

        # update `requires_grad` state in-place
        self[key].requires_grad_(requires_grad)


class HasExtraParameters(nn.ParameterDict):
    def __init__(self, extra_parameters, parameters: Any):
        self.readonly = False
        super().__init__(parameters)
        self.extra_parameter_getter = self.create_parameter_getter(extra_parameters)
        self.readonly = True

    def __getitem__(self, key: str) -> Any:
        return torch.concat([super().__getitem__(key), self.extra_parameter_getter(key)], dim=0)

    def __setitem__(self, key: str, value: Any) -> None:
        if self.readonly:
            # Only update the non-extra part
            n = super().__getitem__(key).shape[0]
            value = value[:n]
        super().__setitem__(key, value)

    @staticmethod
    def create_parameter_getter(container):
        def get(name: str):
            return container[name]

        return get


class TensorDict(nn.ParameterDict):
    def __setitem__(self, key: str, value: Any) -> None:
        self._keys[key] = None
        attr = self._key_to_attr(key)

        assert isinstance(value, torch.Tensor)

        if isinstance(value, nn.Parameter):
            setattr(self, attr, value)
        else:
            self.register_buffer(attr, value)
