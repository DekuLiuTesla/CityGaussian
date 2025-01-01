from typing import Type, Any


class InstantiatableConfig:
    def instantiate(self, *args, **kwargs) -> Any:
        pass
