from argparse import Namespace


def parse_cfg_args(path) -> Namespace:
    with open(path, "r") as f:
        cfg_args = f.read()
    return eval(cfg_args)

def parse_cfg_yaml(data):
    data = Namespace(**data)
    for arg in vars(data):
        if isinstance(getattr(data, arg), dict):
            setattr(data, arg, parse_cfg_yaml(getattr(data, arg)))
    return data