import queue
import traceback

import numpy as np
import torch
import viser

import viser.transforms as vtf
from queue import Queue

from internal.viewer.viewer import Viewer
from internal.cameras.cameras import Cameras


class MockGaussianModel:
    max_sh_degree: int = 3
    active_sh_degree: int = 3


class TrainingViewerRenderer:
    def __init__(self, camera_queue: Queue, renderer_output_queue: Queue):
        self.camera_queue = camera_queue
        self.renderer_output_queue = renderer_output_queue
        self.gaussian_model = MockGaussianModel

    def setup_options(self, *args, **kwargs):
        return

    def get_outputs(self, camera, scaling_modifier: float = 1.):
        # TODO: support multiple client
        # if multiple clients connected, they may get images mismatch to their camera poses,
        # but I think fix it is unnecessary
        self.camera_queue.put((camera, scaling_modifier))
        return self.renderer_output_queue.get()


# TODO: refactoring the the viewer
class TrainingViewer(Viewer):
    def __init__(
            self,
            camera_names: list,
            cameras: Cameras,
            up_direction: np.ndarray = None,
            camera_center: np.ndarray = None,
            available_appearance_options=None,
            host: str = "0.0.0.0",
            port: int = 8080,
    ):
        self.host = host
        self.port = port
        self.image_format = "jpeg"
        self.enable_transform = False
        self.show_cameras = True
        self.show_edit_panel = False
        self.show_render_panel = False
        self.default_camera_position = None
        self.default_camera_look_at = None
        self.camera_transform = torch.eye(4)
        self.device = torch.device("cpu")

        self.camera_names = camera_names
        self.cameras = cameras
        self.up_direction = up_direction
        self.camera_center = camera_center

        if available_appearance_options is not None:
            # convert key type to str
            available_appearance_options = {str(i): available_appearance_options[i] for i in available_appearance_options}
        self.available_appearance_options = available_appearance_options

        self.camera_queue = Queue()
        self.renderer_output_queue = Queue()
        self.viewer_renderer = TrainingViewerRenderer(self.camera_queue, self.renderer_output_queue)

        self.clients = {}

        self.is_training_paused = False

    def add_cameras_to_scene(self, viser_server):
        self.camera_handles = []

        for idx in range(len(self.cameras)):
            camera = self.cameras[idx]
            name = self.camera_names[idx]

            rotation = camera.world_to_camera[:3, :3].clone()
            rotation[:3, 1:3] *= -1
            R = vtf.SO3.from_matrix(rotation.cpu().numpy())
            R = R @ vtf.SO3.from_x_radians(np.pi)

            cx = camera.width.item() // 2
            cy = camera.height.item() // 2
            fx = camera.fx.item()

            camera_handle = viser_server.add_camera_frustum(
                name="cameras/{}".format(name),
                fov=float(2 * np.arctan(cx / fx)),
                scale=0.1,
                aspect=float(cx / cy),
                wxyz=R.wxyz,
                position=camera.camera_center.cpu().numpy(),
                color=(205, 25, 0),
            )

            @camera_handle.on_click
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles.append(camera_handle)

        self.camera_visible = True

        def toggle_camera_visibility(_):
            with viser_server.atomic():
                self.camera_visible = not self.camera_visible
                for i in self.camera_handles:
                    i.visible = self.camera_visible

        with viser_server.gui.add_folder("Cameras"):
            self.toggle_camera_button = viser_server.gui.add_button("Toggle Camera Visibility")
        self.toggle_camera_button.on_click(toggle_camera_visibility)

    def setup_training_panel(self, viewer, server: viser.ViserServer):
        self.pause_training_button = server.gui.add_button("Pause Training", icon=viser.Icon.PLAYER_PAUSE_FILLED)

        @self.pause_training_button.on_click
        def _(_):
            self.pause_training_button.visible = False
            self.resume_training_button.visible = True
            self.is_training_paused = True

        self.resume_training_button = server.gui.add_button("Resume Training", icon=viser.Icon.PLAYER_PLAY_FILLED, visible=False)

        @self.resume_training_button.on_click
        def _(event):
            self.pause_training_button.visible = True
            self.resume_training_button.visible = False
            # mark training resumed
            self.is_training_paused = False
            # send None to camera queue to wake blocking thread up
            self.camera_queue.put((None, None))

        self.global_step_label = server.gui.add_markdown(content="Step: 0")

        self.render_frequency_slider = server.gui.add_slider("Render Freq", initial_value=10, min=1, max=100, step=1)

    def start(self):
        super().start(False, server_config_fun=self.setup_training_panel, enable_renderer_options=False)
        # avoid keeping wasting GPU to render a higher resolution one
        self.max_res_when_static.value = self.max_res_when_moving.value
        self.jpeg_quality_when_static.value = 100
        self.jpeg_quality_when_moving.value = 100

    def process_all_render_requests(self, gaussian_model, renderer, background_color):
        device = gaussian_model.get_xyz.device
        while True:
            try:
                if self.is_training_paused is True:
                    client_camera, scaling_modifier = self.camera_queue.get()
                    if client_camera is None:
                        break
                else:
                    client_camera, scaling_modifier = self.camera_queue.get_nowait()
            except queue.Empty:
                break

            try:
                with torch.no_grad():
                    self.renderer_output_queue.put(renderer(
                        client_camera.to_device(device),
                        gaussian_model,
                        bg_color=background_color.to(device),
                        scaling_modifier=scaling_modifier,
                    )["render"])
            except:
                traceback.print_exc()

    def training_step(self, gaussian_model, renderer, background_color, step: int):
        self.global_step_label.content = f"Step: {step}"

        if self.is_training_paused is False:
            if self.camera_queue.empty() is True:
                if step % int(self.render_frequency_slider.value) == 0:
                    self.rerender_for_all_client()
                else:
                    return

        self.process_all_render_requests(gaussian_model, renderer, background_color)

    def validation_step(self, gaussian_model, renderer, background_color, step: int):
        if self.camera_queue.empty() is True:
            return
        self.process_all_render_requests(gaussian_model, renderer, background_color)
