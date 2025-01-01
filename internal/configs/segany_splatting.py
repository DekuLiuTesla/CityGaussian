from dataclasses import dataclass


@dataclass
class Optimization:
    lr: float = 0.0025
    lr_final_factor: float = 1.
