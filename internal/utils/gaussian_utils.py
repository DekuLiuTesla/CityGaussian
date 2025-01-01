import os
import numpy as np
import torch
from internal.utils.colmap import rotmat2qvec, qvec2rotmat
from typing import Union
from dataclasses import dataclass
from plyfile import PlyData, PlyElement

SHS_REST_DIM_TO_DEGREE = {
    0: 0,
    3: 1,
    8: 2,
    15: 3,
}


@dataclass
class GaussianPlyUtils:
    """
    Load parameters from ply;
    Save to ply;
    """

    sh_degrees: int
    xyz: Union[np.ndarray, torch.Tensor]  # [n, 3]
    opacities: Union[np.ndarray, torch.Tensor]  # [n, 1]
    features_dc: Union[np.ndarray, torch.Tensor]  # ndarray[n, 3, 1], or tensor[n, 1, 3]
    features_rest: Union[np.ndarray, torch.Tensor]  # ndarray[n, 3, 15], or tensor[n, 15, 3]; NOTE: this is features_rest actually!
    scales: Union[np.ndarray, torch.Tensor]  # [n, 3]
    rotations: Union[np.ndarray, torch.Tensor]  # [n, 4]

    @staticmethod
    def detect_sh_degree_from_shs_rest(shs_rest: torch.Tensor):
        assert isinstance(shs_rest, torch.Tensor)
        return SHS_REST_DIM_TO_DEGREE[shs_rest.shape[-2]]

    @staticmethod
    def load_array_from_plyelement(plyelement, name_prefix: str, required: bool = True):
        names = [p.name for p in plyelement.properties if p.name.startswith(name_prefix)]
        if len(names) == 0:
            if required is True:
                raise RuntimeError(f"'{name_prefix}' not found in ply")
            return np.empty((plyelement["x"].shape[0], 0))
        names = sorted(names, key=lambda x: int(x.split('_')[-1]))
        v_list = []
        for idx, attr_name in enumerate(names):
            v_list.append(np.asarray(plyelement[attr_name]))

        return np.stack(v_list, axis=1)

    @classmethod
    def load_from_ply(cls, path: str, sh_degrees: int = -1):
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        features_rest = cls.load_array_from_plyelement(plydata.elements[0], "f_rest_", required=False).reshape((xyz.shape[0], 3, -1))
        if sh_degrees >= 0:
            assert features_rest.shape[-1] == (sh_degrees + 1) ** 2 - 1  # TODO: remove such a assertion
        else:
            # auto determine sh_degrees
            features_rest_dims = features_rest.shape[-1]
            for i in range(4):
                if features_rest_dims == (i + 1) ** 2 - 1:
                    sh_degrees = i
                    break
            assert sh_degrees >= 0, f"invalid sh_degrees={sh_degrees}"

        scales = cls.load_array_from_plyelement(plydata.elements[0], "scale_")
        rots = cls.load_array_from_plyelement(plydata.elements[0], "rot_")

        return cls(
            sh_degrees=sh_degrees,
            xyz=xyz,
            opacities=opacities,
            features_dc=features_dc,
            features_rest=features_rest,
            scales=scales,
            rotations=rots,
        )

    @classmethod
    def load_from_model_properties(cls, properties, sh_degree: int = -1):
        if sh_degree < 0:
            sh_degree = cls.detect_sh_degree_from_shs_rest(properties["shs_rest"])

        init_args = {
            "sh_degrees": sh_degree,
        }

        for name_in_model, name_in_dataclass in [
            ("means", "xyz"),
            ("shs_dc", "features_dc"),
            ("shs_rest", "features_rest"),
            ("scales", "scales"),
            ("rotations", "rotations"),
            ("opacities", "opacities"),
        ]:
            init_args[name_in_dataclass] = properties[name_in_model].detach()

        return cls(**init_args)

    @classmethod
    def load_from_model(cls, model):
        return cls.load_from_model_properties(model.properties, sh_degree=model.max_sh_degree)

    @classmethod
    def load_from_state_dict(cls, state_dict):
        if "gaussian_model.gaussians.means" in state_dict:
            return cls.load_from_new_state_dict(state_dict)
        return cls.load_from_old_state_dict(state_dict)

    @classmethod
    def load_from_new_state_dict(cls, state_dict):
        prefix = "gaussian_model.gaussians."

        init_args = {
            "sh_degrees": cls.detect_sh_degree_from_shs_rest(state_dict["{}shs_rest".format(prefix)]),
        }

        for name_in_dict, name_in_dataclass in [
            ("means", "xyz"),
            ("shs_dc", "features_dc"),
            ("shs_rest", "features_rest"),
            ("scales", "scales"),
            ("rotations", "rotations"),
            ("opacities", "opacities"),
        ]:
            init_args[name_in_dataclass] = state_dict["{}{}".format(prefix, name_in_dict)]

        return cls(**init_args)

    @classmethod
    def load_from_old_state_dict(cls, state_dict):
        key_prefix = "gaussian_model._"

        init_args = {
            "sh_degrees": cls.detect_sh_degree_from_shs_rest(state_dict["{}features_rest".format(key_prefix)]),
        }
        for name_in_dict, name_in_dataclass in [
            ("xyz", "xyz"),
            ("features_dc", "features_dc"),
            ("features_rest", "features_rest"),
            ("scaling", "scales"),
            ("rotation", "rotations"),
            ("opacity", "opacities"),
        ]:
            init_args[name_in_dataclass] = state_dict["{}{}".format(key_prefix, name_in_dict)]

        return cls(**init_args)

    def to_parameter_structure(self):
        assert isinstance(self.xyz, np.ndarray) is True
        return GaussianPlyUtils(
            sh_degrees=self.sh_degrees,
            xyz=torch.tensor(self.xyz, dtype=torch.float),
            opacities=torch.tensor(self.opacities, dtype=torch.float),
            features_dc=torch.tensor(self.features_dc, dtype=torch.float).transpose(1, 2),
            features_rest=torch.tensor(self.features_rest, dtype=torch.float).transpose(1, 2),
            scales=torch.tensor(self.scales, dtype=torch.float),
            rotations=torch.tensor(self.rotations, dtype=torch.float),
        )

    @torch.no_grad()
    def to_ply_format(self):
        assert isinstance(self.xyz, torch.Tensor) is True
        return GaussianPlyUtils(
            sh_degrees=self.sh_degrees,
            xyz=self.xyz.cpu().numpy(),
            opacities=self.opacities.cpu().numpy(),
            features_dc=self.features_dc.transpose(1, 2).cpu().numpy(),
            features_rest=self.features_rest.transpose(1, 2).cpu().numpy(),
            scales=self.scales.cpu().numpy(),
            rotations=self.rotations.cpu().numpy(),
        )

    def save_to_ply(self, path: str, with_colors: bool = False):
        assert isinstance(self.xyz, np.ndarray) is True

        gaussian = self

        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = gaussian.xyz
        normals = np.zeros_like(xyz)
        f_dc = gaussian.features_dc.reshape((gaussian.features_dc.shape[0], -1))
        # TODO: change sh degree
        if gaussian.sh_degrees > 0:
            f_rest = gaussian.features_rest.reshape((gaussian.features_rest.shape[0], -1))
        else:
            f_rest = np.zeros((f_dc.shape[0], 0))
        opacities = gaussian.opacities
        scale = gaussian.scales
        rotation = gaussian.rotations

        def construct_list_of_attributes():
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            # All channels except the 3 DC
            for i in range(gaussian.features_dc.shape[1] * gaussian.features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            if gaussian.sh_degrees > 0:
                for i in range(gaussian.features_rest.shape[1] * gaussian.features_rest.shape[2]):
                    l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(gaussian.scales.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(gaussian.rotations.shape[1]):
                l.append('rot_{}'.format(i))
            return l

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
        attribute_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        if with_colors is True:
            from internal.utils.sh_utils import eval_sh
            rgbs = np.clip((eval_sh(0, self.features_dc, None) + 0.5), 0., 1.)
            rgbs = (rgbs * 255).astype(np.uint8)

            dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            attribute_list.append(rgbs)

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attribute_list, axis=1)
        # do not save 'features_extra' for ply
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, f_extra), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


class GaussianTransformUtils:
    @staticmethod
    def translation(xyz, x: float, y: float, z: float):
        if x == 0. and y == 0. and z == 0.:
            return xyz

        return xyz + torch.tensor([[x, y, z]], device=xyz.device)

    @staticmethod
    def rescale(xyz, scaling, factor: float):
        if factor == 1.:
            return xyz, scaling
        return xyz * factor, scaling * factor

    @staticmethod
    def rx(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[1, 0, 0],
                             [0, torch.cos(theta), -torch.sin(theta)],
                             [0, torch.sin(theta), torch.cos(theta)]], dtype=torch.float)

    @staticmethod
    def ry(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                             [0, 1, 0],
                             [-torch.sin(theta), 0, torch.cos(theta)]], dtype=torch.float)

    @staticmethod
    def rz(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0],
                             [0, 0, 1]], dtype=torch.float)

    @classmethod
    def rotate_by_euler_angles(cls, xyz, rotation, x: float, y: float, z: float):
        """
        rotate in z-y-x order, radians as unit
        """

        if x == 0. and y == 0. and z == 0.:
            return

        # rotate
        rotation_matrix = cls.rx(x) @ cls.ry(y) @ cls.rz(z)
        xyz, rotation = cls.rotate_by_matrix(
            xyz,
            rotation,
            rotation_matrix.to(xyz),
        )

        return xyz, rotation

    @staticmethod
    def transform_shs(features, rotation_matrix):
        """
        https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570
        """

        try:
            from e3nn import o3
            import einops
            from einops import einsum
        except:
            print("Please run `pip install e3nn einops` to enable SHs rotation")
            return features

        if features.shape[1] == 1:
            return features

        features = features.clone()

        shs_feat = features[:, 1:, :]

        ## rotate shs
        P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=shs_feat.dtype, device=shs_feat.device)  # switch axes: yzx -> xyz
        inversed_P = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=shs_feat.dtype, device=shs_feat.device)
        permuted_rotation_matrix = inversed_P @ rotation_matrix @ P
        rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix.cpu())

        # Construction coefficient
        D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
        D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
        D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)

        # rotation of the shs features
        one_degree_shs = shs_feat[:, 0:3]
        one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        one_degree_shs = einsum(
            D_1,
            one_degree_shs,
            "... i j, ... j -> ... i",
        )
        one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 0:3] = one_degree_shs

        if shs_feat.shape[1] >= 4:
            two_degree_shs = shs_feat[:, 3:8]
            two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            two_degree_shs = einsum(
                D_2,
                two_degree_shs,
                "... i j, ... j -> ... i",
            )
            two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            shs_feat[:, 3:8] = two_degree_shs

            if shs_feat.shape[1] >= 9:
                three_degree_shs = shs_feat[:, 8:15]
                three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
                three_degree_shs = einsum(
                    D_3,
                    three_degree_shs,
                    "... i j, ... j -> ... i",
                )
                three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
                shs_feat[:, 8:15] = three_degree_shs

        return features

    @classmethod
    def rotate_by_wxyz_quaternions(cls, xyz, rotations, features, quaternions: torch.tensor):
        if torch.all(quaternions == 0.) or torch.all(quaternions == torch.tensor(
                [1., 0., 0., 0.],
                dtype=quaternions.dtype,
                device=quaternions.device,
        )):
            return xyz, rotations, features

        # convert quaternions to rotation matrix
        rotation_matrix = torch.tensor(qvec2rotmat(quaternions.cpu().numpy()), dtype=torch.float, device=xyz.device)
        # rotate xyz
        xyz = torch.matmul(xyz, rotation_matrix.T)
        # rotate gaussian quaternions
        rotations = torch.nn.functional.normalize(cls.quat_multiply(
            rotations,
            quaternions,
        ))

        features = cls.transform_shs(features, rotation_matrix)

        return xyz, rotations, features

    @staticmethod
    def quat_multiply(quaternion0, quaternion1):
        w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
        w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
        return torch.concatenate((
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ), dim=-1)

    @classmethod
    def rotate_by_matrix(cls, xyz, rotations, rotation_matrix):
        # rotate xyz
        xyz = torch.matmul(xyz, rotation_matrix.T)

        # rotate via quaternion
        rotations = torch.nn.functional.normalize(cls.quat_multiply(
            rotations,
            torch.tensor([rotmat2qvec(rotation_matrix.cpu().numpy())]).to(xyz),
        ))

        return xyz, rotations
