import torch
from torch import nn
import internal.utils.gaussian_utils as gaussian_utils
from internal.utils.general_utils import inverse_sigmoid


class GaussianModelSimplified(nn.Module):
    def __init__(
            self,
            xyz: torch.Tensor,
            features_dc: torch.Tensor,
            features_rest: torch.Tensor,
            scaling: torch.Tensor,
            rotation: torch.Tensor,
            opacity: torch.Tensor,
            features_extra: torch.Tensor,
            sh_degree: int,
            device,
    ) -> None:
        super().__init__()

        self._xyz = xyz.to(device)
        # self._features_dc = features_dc
        # self._features_rest = features_rest
        self._scaling = torch.exp(scaling).to(device)
        self._rotation = torch.nn.functional.normalize(rotation).to(device)
        self._opacity = torch.sigmoid(opacity).to(device)

        # TODO: load only specific dimensions correspond to the sh_degree
        self._features = torch.cat([features_dc, features_rest], dim=1).to(device)

        self._opacity_origin = None

        self._features_extra = features_extra.to(device)

        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree

    def to_device(self, device):
        self._xyz = self._xyz.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._opacity = self._opacity.to(device)
        self._features = self._features.to(device)
        self._features_extra = self._features_extra.to(device)
        return self

    @classmethod
    def construct_from_state_dict(cls, state_dict, active_sh_degree, device):
        # init_args = {
        #     "sh_degree": active_sh_degree,
        #     "device": device,
        # }
        # for i in state_dict:
        #     if i.startswith("gaussian_model._") is False:
        #         continue
        #     init_args[i[len("gaussian_model._"):]] = state_dict[i]
        #
        # if "features_extra" not in init_args:
        #     init_args["features_extra"] = torch.empty((init_args["xyz"].shape[0], 0))

        gaussian = gaussian_utils.Gaussian.load_from_state_dict(active_sh_degree, state_dict)
        return cls(
            xyz=gaussian.xyz,
            features_dc=gaussian.features_dc,
            features_rest=gaussian.features_rest,
            scaling=gaussian.scales,
            rotation=gaussian.rotations,
            opacity=gaussian.opacities,
            features_extra=gaussian.real_features_extra,
            sh_degree=active_sh_degree,
            device=device,
        )

    @classmethod
    def construct_from_ply(cls, ply_path: str, sh_degree, device):
        gaussians = gaussian_utils.Gaussian.load_from_ply(ply_path, sh_degree).to_parameter_structure()
        return cls(
            sh_degree=sh_degree,
            device=device,
            xyz=gaussians.xyz,
            opacity=gaussians.opacities,
            features_dc=gaussians.features_dc,
            features_rest=gaussians.features_rest,
            scaling=gaussians.scales,
            rotation=gaussians.rotations,
            features_extra=gaussians.real_features_extra,
        )

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_features_extra(self):
        return self._features_extra

    def select(self, mask: torch.tensor):
        if self._opacity_origin is None:
            self._opacity_origin = torch.clone(self._opacity)  # make a backup
        else:
            self._opacity = torch.clone(self._opacity_origin)

        self._opacity[mask] = 0.

    def delete_gaussians(self, mask: torch.tensor):
        gaussians_to_be_preserved = torch.bitwise_not(mask).to(self._xyz.device)
        self._xyz = self._xyz[gaussians_to_be_preserved]
        self._scaling = self._scaling[gaussians_to_be_preserved]
        self._rotation = self._rotation[gaussians_to_be_preserved]

        if self._opacity_origin is not None:
            self._opacity = self._opacity_origin
            self._opacity_origin = None
        self._opacity = self._opacity[gaussians_to_be_preserved]

        self._features = self._features[gaussians_to_be_preserved]
        self._features_extra = self._features_extra[gaussians_to_be_preserved]

    def to_parameter_structure(self, device='cpu') -> gaussian_utils.Gaussian:
        xyz = self._xyz.to(device)
        features_dc = self._features[:, :1, :].to(device)
        features_rest = self._features[:, 1:, :].to(device)
        scaling = torch.log(self._scaling).to(device)
        rotation = self._rotation.to(device)
        opacity = inverse_sigmoid(self._opacity).to(device)
        features_extra = self._features_extra.to(device)

        return gaussian_utils.Gaussian(
            sh_degrees=self.max_sh_degree,
            xyz=xyz,
            opacities=opacity,
            features_dc=features_dc,
            features_rest=features_rest,
            scales=scaling,
            rotations=rotation,
            real_features_extra=features_extra,
        )

    def to_ply_structure(self) -> gaussian_utils.Gaussian:
        xyz = self._xyz.cpu().numpy()
        features_dc = self._features[:, :1, :].transpose(1, 2).cpu().numpy()
        features_rest = self._features[:, 1:, :].transpose(1, 2).cpu().numpy()
        scaling = torch.log(self._scaling).cpu().numpy()
        rotation = self._rotation.cpu().numpy()
        opacity = inverse_sigmoid(self._opacity).cpu().numpy()
        features_extra = self._features_extra.cpu().numpy()

        return gaussian_utils.Gaussian(
            sh_degrees=self.max_sh_degree,
            xyz=xyz,
            opacities=opacity,
            features_dc=features_dc,
            features_rest=features_rest,
            scales=scaling,
            rotations=rotation,
            real_features_extra=features_extra,
        )
