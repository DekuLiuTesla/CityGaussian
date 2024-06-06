from dataclasses import dataclass
import numpy as np
import math
import viser
import viser.transforms as vst
import torch

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

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

    @classmethod
    def rotate_by_wxyz_quaternions(cls, xyz, rotations, quaternions: torch.tensor):
        if torch.all(quaternions == 0.) or torch.all(quaternions == torch.tensor(
                [1., 0., 0., 0.],
                dtype=quaternions.dtype,
                device=quaternions.device,
        )):
            return xyz, rotations

        # convert quaternions to rotation matrix
        rotation_matrix = torch.tensor(qvec2rotmat(quaternions.cpu().numpy()), dtype=torch.float, device=xyz.device)
        # rotate xyz
        xyz = torch.matmul(xyz, rotation_matrix.T)
        # rotate gaussian quaternions
        rotations = torch.nn.functional.normalize(cls.quat_multiply(
            rotations,
            quaternions,
        ))

        return xyz, rotations

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


@dataclass
class ModelPose:
    wxyz: np.ndarray
    position: np.ndarray

    def copy(self):
        return ModelPose(
            wxyz=self.wxyz.copy(),
            position=self.position.copy(),
        )

    def to_dict(self):
        return {
            "wxyz": self.wxyz.tolist(),
            "position": self.position.tolist(),
        }


class TransformPanel:
    def __init__(
            self,
            server: viser.ViserServer,
            viewer,
            n_models: int,
    ):
        self.server = server
        self.viewer = viewer

        self.transform_control_no_handle_update = False

        self.model_poses = []
        self.model_transform_controls: dict[int, viser.TransformControlsHandle] = {}
        self.model_size_sliders = []
        self.model_show_transform_control_checkboxes = []
        self.model_t_xyz_text_handle = []
        self.model_r_xyz_text_handle = []

        self.pose_control_size = server.add_gui_slider(
            "Pose Control Size",
            min=0.,
            max=10.,
            step=0.01,
            initial_value=0.4,
        )
        self.pose_control_size.on_update(self._update_pose_control_size)

        # create gui folder for each model
        for i in range(n_models):
            with server.add_gui_folder("Model {} Transform".format(i)):
                # model size control
                size_slider = server.add_gui_number(
                    "Size",
                    min=0.,
                    # max=5.,
                    step=0.01,
                    initial_value=1.,
                )
                self._make_size_slider_callback(i, size_slider)
                self.model_size_sliders.append(size_slider)

                # model pose control
                self.model_poses.append(ModelPose(
                    np.asarray([1., 0., 0., 0.]),
                    np.zeros((3,)),
                ))
                model_show_transform_control_checkbox = server.add_gui_checkbox(
                    "Pose Control",
                    initial_value=False,
                )
                self._make_show_transform_control_checkbox_callback(i, model_show_transform_control_checkbox)
                self.model_show_transform_control_checkboxes.append(model_show_transform_control_checkbox)

                # add text input (synchronize with model pose control) that control model pose more precisely
                t_xyz_text_handle = server.add_gui_vector3(
                    "t_xyz",
                    initial_value=(0., 0., 0.),
                    step=0.01,
                )
                self._make_t_xyz_text_callback(i, t_xyz_text_handle)
                self.model_t_xyz_text_handle.append(t_xyz_text_handle)

                r_xyz_text_handle = server.add_gui_vector3(
                    "r_xyz",
                    initial_value=(0., 0., 0.),
                    # min=(-180, -180, -180),
                    # max=(180, 180, 180),
                    step=0.1,
                )
                self._make_r_xyz_text_callback(i, r_xyz_text_handle)
                self.model_r_xyz_text_handle.append(r_xyz_text_handle)

    def _make_size_slider_callback(
            self,
            idx: int,
            slider: viser.GuiInputHandle,
    ):
        @slider.on_update
        def _(event: viser.GuiEvent) -> None:
            with self.server.atomic():
                self._transform_model(idx)
                self.viewer.rerender_for_client(event.client_id)

    def set_model_transform_control_value(self, idx, wxyz: np.ndarray, position: np.ndarray):
        if idx in self.model_transform_controls:
            self.transform_control_no_handle_update = True
            try:
                    self.model_transform_controls[idx].wxyz = wxyz
                    self.model_transform_controls[idx].position = position
            finally:
                self.transform_control_no_handle_update = False

    def _make_transform_controls_callback(
            self,
            idx,
            controls: viser.TransformControlsHandle,
    ) -> None:
        @controls.on_update
        def _(event: viser.GuiEvent) -> None:
            if self.transform_control_no_handle_update is True:
                return
            model_pose = self.model_poses[idx]
            model_pose.wxyz = controls.wxyz
            model_pose.position = controls.position

            self.model_t_xyz_text_handle[idx].value = model_pose.position.tolist()
            self.model_r_xyz_text_handle[idx].value = self.quaternion_to_euler_angle_vectorized2(model_pose.wxyz)

            self._transform_model(idx)
            self.viewer.rerender_for_all_client()

    def _show_model_transform_handle(
            self,
            idx: int,
    ):
        model_pose = self.model_poses[idx]
        controls = self.server.add_transform_controls(
            f"/model_transform/{idx}",
            scale=self.pose_control_size.value,
            wxyz=model_pose.wxyz,
            position=model_pose.position,
        )
        self._make_transform_controls_callback(idx, controls)
        self.model_transform_controls[idx] = controls

    def _make_show_transform_control_checkbox_callback(
            self,
            idx: int,
            checkbox: viser.GuiInputHandle,
    ):
        @checkbox.on_update
        def _(event: viser.GuiEvent) -> None:
            if checkbox.value is True:
                self._show_model_transform_handle(idx)
            else:
                if idx in self.model_transform_controls:
                    self.model_transform_controls[idx].remove()
                    del self.model_transform_controls[idx]

    def _update_pose_control_size(self, _):
        with self.server.atomic():
            for i in self.model_transform_controls:
                self.model_transform_controls[i].remove()
                self._show_model_transform_handle(i)

    def _transform_model(self, idx):
        model_pose = self.model_poses[idx]
        self.viewer.gaussian_model.transform_with_vectors(
            idx,
            scale=self.model_size_sliders[idx].value,
            r_wxyz=model_pose.wxyz,
            t_xyz=model_pose.position,
        )

    def _make_t_xyz_text_callback(
            self,
            idx: int,
            handle: viser.GuiInputHandle,
    ):
        @handle.on_update
        def _(event: viser.GuiEvent) -> None:
            if event.client is None:
                return

            with self.server.atomic():
                t = np.asarray(handle.value)
                if idx in self.model_transform_controls:
                    self.model_transform_controls[idx].position = t
                self.model_poses[idx].position = t

                self._transform_model(idx)
                self.viewer.rerender_for_all_client()

    def _make_r_xyz_text_callback(
            self,
            idx: int,
            handle: viser.GuiInputHandle,
    ):
        @handle.on_update
        def _(event: viser.GuiEvent) -> None:
            if event.client is None:
                return

            with self.server.atomic():
                radians = np.radians(np.asarray(handle.value))
                so3 = vst.SO3.from_rpy_radians(*radians.tolist())
                wxyz = np.asarray(so3.wxyz)
                if idx in self.model_transform_controls:
                    self.model_transform_controls[idx].wxyz = wxyz
                self.model_poses[idx].wxyz = wxyz

            self._transform_model(idx)
            self.viewer.rerender_for_all_client()

    @staticmethod
    def quaternion_to_euler_angle_vectorized2(wxyz):
        xyzw = np.zeros_like(wxyz)
        xyzw[[0, 1, 2, 3]] = wxyz[[1, 2, 3, 0]]
        euler_radians = vst.SO3.from_quaternion_xyzw(xyzw).as_rpy_radians()
        return math.degrees(euler_radians.roll), math.degrees(euler_radians.pitch), math.degrees(euler_radians.yaw)
