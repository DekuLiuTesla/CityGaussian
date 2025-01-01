import torch
import numpy as np
from internal.utils import gaussian_utils


class MultipleGaussianModelEditor:
    def __init__(
            self,
            gaussian_models: list,
            device=None,
    ):
        """
        Args:
            gaussian_models: must be same type, sh_degree and pre-activated state
        """
        self.device = device
        if self.device is None:
            self.device = gaussian_models[0].means.device

        # get total number of gaussians and indices of each model
        total_gaussian_num = 0
        model_gaussian_indices = []
        for i in gaussian_models:
            n = i.get_xyz.shape[0]
            model_gaussian_indices.append((total_gaussian_num, total_gaussian_num + n))
            total_gaussian_num += n
        self.model_gaussian_indices = model_gaussian_indices

        if len(gaussian_models) == 1:
            self.gaussian_model = gaussian_models[0]
        else:
            # create a same model type
            self.gaussian_model = gaussian_models[0].config.instantiate()
            self.gaussian_model.setup_from_number(1)
            self.gaussian_model.to(self.device)
            if gaussian_models[0].is_pre_activated is True:
                self.gaussian_model.pre_activate_all_properties()
            self.gaussian_model.active_sh_degree = gaussian_models[0].active_sh_degree

            # store all properties into this model
            self.gaussian_model.properties = self._concat_all_properties(gaussian_models, self.device)

        self._modified_opacities = None
        self._original_properties = None

    def __getattr__(self, item):
        return getattr(self.gaussian_model, item)

    @staticmethod
    def _concat_all_properties(models, device):
        properties = {}
        for name in models[0].property_names:
            properties[name] = torch.concat([model.get_property(name).to(device) for model in models], dim=0)
        return properties

    def get_opacities(self):
        if self._modified_opacities is None:
            return self.gaussian_model.get_opacities()
        return self._modified_opacities

    @property
    def get_opacity(self):
        return self.get_opacities()

    def select(self, mask: torch.tensor):
        self._modified_opacities = torch.clone(self.gaussian_model.get_opacities())
        self._modified_opacities[mask] = 0.

    def delete_gaussians(self, mask: torch.tensor):
        self._modified_opacities = None

        gaussians_to_be_preserved = torch.bitwise_not(mask).to(device=self.gaussian_model.means.device)

        # delete for each model
        new_properties = {}
        new_original_properties = {}
        for key, value in self.gaussian_model.properties.items():
            new_properties[key] = []
            new_original_properties[key] = []
            for begin, end in self.model_gaussian_indices:
                model_property = value[begin:end]
                model_property_mask = gaussians_to_be_preserved[begin:end]
                new_properties[key].append(model_property[model_property_mask])

                # prune original properties if exists
                if self._original_properties is not None and key in self._original_properties:
                    model_original_property = self._original_properties[key][begin:end]
                    model_original_property_mask = gaussians_to_be_preserved[begin:end].to(model_original_property.device)
                    new_original_properties[key].append(model_original_property[model_original_property_mask])

        # recalculate index range
        total_gaussian_num = 0
        model_gaussian_indices = []
        for i in new_properties["means"]:
            n = i.shape[0]
            model_gaussian_indices.append((total_gaussian_num, total_gaussian_num + n))
            total_gaussian_num += n
        self.model_gaussian_indices = model_gaussian_indices

        # concat slices
        new_properties = {i: torch.concat(new_properties[i], dim=0) for i in new_properties}
        if self._original_properties is not None:
            self._original_properties = {key: torch.concat(new_original_properties[key], dim=0) for key in self._original_properties}

        self.gaussian_model.properties = new_properties

    def backup_properties(self):
        # backup activated properties
        self._original_properties = {
            "means": self.gaussian_model.get_means().clone(),
            "scales": self.gaussian_model.get_scales().clone(),
            "rotations": self.gaussian_model.get_rotations().clone(),
            "shs": self.gaussian_model.get_shs().clone(),
        }

    def get_model_gaussian_indices(self, idx: int):
        return self.model_gaussian_indices[idx]

    @torch.no_grad()
    def transform_with_vectors(
            self,
            idx: int,
            scale: float,
            r_wxyz: np.ndarray,
            t_xyz: np.ndarray,
    ):
        device = self.gaussian_model.means.device

        begin, end = self.get_model_gaussian_indices(idx)

        if self._original_properties is None:
            # only create a backup for the first call
            self.backup_properties()

        model = self.gaussian_model
        # pick properties corresponds to the specified `idx` of model
        xyz = self._original_properties["means"][begin:end].clone().to(device)
        # TODO: avoid memory copy if no rotation or scaling happened compared to previous state
        scaling = self._original_properties["scales"][begin:end].clone().to(device)
        rotation = self._original_properties["rotations"][begin:end].clone().to(device)
        features = self._original_properties["shs"][begin:end].clone().to(device)  # consume a lot of memory

        # rescale
        xyz, scaling = gaussian_utils.GaussianTransformUtils.rescale(
            xyz,
            scaling,
            scale
        )
        # rotate
        xyz, rotation, new_features = gaussian_utils.GaussianTransformUtils.rotate_by_wxyz_quaternions(
            xyz=xyz,
            rotations=rotation,
            features=features,
            quaternions=torch.tensor(r_wxyz).to(xyz),
        )
        # translate
        xyz = gaussian_utils.GaussianTransformUtils.translation(xyz, *t_xyz.tolist())

        # `model.NAME = value` not works if value is a nn.Parameter
        new_properties = {
            "means": xyz,
            "scales": model.scale_inverse_activation(scaling),
            "rotations": model.rotation_inverse_activation(rotation),
            "shs_dc": new_features[:, :1, :],
            "shs_rest": new_features[:, 1:, :],
            # the pre-activated model only has `shs`
            "shs": new_features,
        }
        # since only a slice of each property need to be updated, operating on `model.gaussians` directly is required
        for key, value in new_properties.items():
            if key not in model.gaussians:
                continue
            model.gaussians[key][begin:end] = value
