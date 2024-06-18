import os
from pathlib import Path
import math
import glob
import time
import json
import argparse
from typing import Tuple, Literal, List

import numpy as np
import viser
import viser.transforms as vtf
import torch
from internal.renderers import VanillaRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.models.simplified_gaussian_model_manager import SimplifiedGaussianModelManager
from internal.viewer import ClientThread, ViewerRenderer
from internal.viewer.ui import populate_render_tab, TransformPanel, EditPanel
from internal.viewer.ui.up_direction_folder import UpDirectionFolder

DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE = "@Direct"


class Viewer:
    def __init__(
            self,
            model_paths: list[str],
            host: str = "0.0.0.0",
            port: int = 8080,
            background_color: Tuple = (0, 0, 0),
            image_format: Literal["jpeg", "png"] = "jpeg",
            reorient: Literal["auto", "enable", "disable"] = "auto",
            sh_degree: int = 3,
            enable_transform: bool = False,
            show_cameras: bool = False,
            cameras_json: str = None,
            vanilla_deformable: bool = False,
            vanilla_gs4d: bool = False,
            vanilla_gs2d: bool = False,
            up: list[float] = None,
            default_camera_position: List[float] = None,
            default_camera_look_at: List[float] = None,
            no_edit_panel: bool = False,
            no_render_panel: bool = False,
            gsplat: bool = False,
    ):
        self.device = torch.device("cuda")

        self.model_paths = model_paths
        self.host = host
        self.port = port
        self.background_color = background_color
        self.image_format = image_format
        self.sh_degree = sh_degree
        self.enable_transform = enable_transform
        self.show_cameras = show_cameras
        self.extra_video_render_args = []

        self.up_direction = np.asarray([0., 0., 1.])
        self.camera_center = np.asarray([0., 0., 0.])
        self.default_camera_position = default_camera_position
        self.default_camera_look_at = default_camera_look_at

        self.use_gsplat = gsplat

        load_from = self._search_load_file(model_paths[0])

        self.simplified_model = True
        self.show_edit_panel = True
        if no_edit_panel is True:
            self.show_edit_panel = False
        self.show_render_panel = True
        if no_render_panel is True:
            self.show_render_panel = False
        # whether model is trained by other implementations
        if vanilla_gs4d is True:
            self.simplified_model = False

        # TODO: load multiple models more elegantly
        # load and create models
        model, renderer, training_output_base_dir, dataset_type, self.checkpoint = self._load_model_from_file(load_from)
        # whether a 2DGS model
        if load_from.endswith(".ply") and model.get_scaling.shape[-1] == 2:
            print("2DGS ply detected")
            vanilla_gs2d = True

        def get_load_iteration() -> int:
            return int(os.path.basename(os.path.dirname(load_from)).replace("iteration_", ""))

        # whether model is trained by other implementations
        if vanilla_deformable is True:
            from internal.renderers.vanilla_deformable_renderer import VanillaDeformableRenderer
            renderer = VanillaDeformableRenderer(
                os.path.dirname(os.path.dirname(os.path.dirname(load_from))),
                get_load_iteration(),
                device=self.device,
            )
            self.show_edit_panel = False
            self.show_render_panel = False
        elif vanilla_gs4d is True:
            from internal.renderers.vanilla_gs4d_renderer import VanillaGS4DRenderer
            renderer = VanillaGS4DRenderer(
                os.path.dirname(os.path.dirname(os.path.dirname(load_from))),
                get_load_iteration(),
                device=self.device,
            )
            self.show_edit_panel = False
            self.show_render_panel = False
        elif vanilla_gs2d is True:
            from internal.renderers.vanilla_2dgs_renderer import Vanilla2DGSRenderer
            renderer = Vanilla2DGSRenderer()
            self.extra_video_render_args.append("--vanilla_gs2d")

        # reorient the scene
        cameras_json_path = cameras_json
        if cameras_json_path is None:
            cameras_json_path = os.path.join(training_output_base_dir, "cameras.json")
        self.camera_transform = self._reorient(cameras_json_path, mode=reorient, dataset_type=dataset_type)
        if up is not None:
            self.camera_transform = torch.eye(4, dtype=torch.float)
            up = torch.tensor(up)
            up = -up / torch.linalg.norm(up)
            self.up_direction = up.numpy()

        # load camera poses
        self.camera_poses = self.load_camera_poses(cameras_json_path)
        # calculate camera center
        if len(self.camera_poses) > 0:
            self.camera_center = np.mean(np.asarray([i["position"] for i in self.camera_poses]), axis=0)

        self.available_appearance_options = None

        self.loaded_model_count = 1
        addition_models = model_paths[1:]
        if len(addition_models) == 0:
            # load appearance groups
            appearance_group_filename = os.path.join(training_output_base_dir, "appearance_group_ids.json")
            if os.path.exists(appearance_group_filename) is True:
                with open(appearance_group_filename, "r") as f:
                    self.available_appearance_options = json.load(f)
            # self.available_appearance_options["@Disabled"] = None

            if enable_transform is True:
                model = SimplifiedGaussianModelManager(
                    [model.to_device(torch.device("cpu"))],
                    enable_transform=True,
                    device=self.device,
                )
        else:
            # switch to vanilla renderer
            model_list = [model.to_device(torch.device("cpu"))]
            renderer = VanillaRenderer()
            for model_path in addition_models:
                load_from = self._search_load_file(model_path)
                if load_from.endswith(".ckpt"):
                    load_results = self._do_initialize_models_from_checkpoint(load_from, device=torch.device("cpu"))
                else:
                    load_results = self._do_initialize_models_from_point_cloud(load_from, self.sh_degree, device=torch.device("cpu"))
                model_list.append(load_results[0])

            model = SimplifiedGaussianModelManager(model_list, enable_transform=enable_transform, device=self.device)

            self.loaded_model_count += len(addition_models)

        self.gaussian_model = model
        # create renderer
        self.viewer_renderer = ViewerRenderer(
            model,
            renderer,
            torch.tensor(background_color, dtype=torch.float, device=self.device),
        )

        self.clients = {}

    @staticmethod
    def _search_load_file(model_path: str) -> str:
        return GaussianModelLoader.search_load_file(model_path)

    def _reorient(self, cameras_json_path: str, mode: str, dataset_type: str = None):
        transform = torch.eye(4, dtype=torch.float)

        if mode == "disable":
            return transform

        # detect whether cameras.json exists
        is_cameras_json_exists = os.path.exists(cameras_json_path)

        if is_cameras_json_exists is False:
            if mode == "enable":
                raise RuntimeError("{} not exists".format(cameras_json_path))
            else:
                return transform

        # skip reorient if dataset type is blender
        if dataset_type in ["blender", "nsvf"] and mode == "auto":
            print("skip reorient for {} dataset".format(dataset_type))
            return transform

        print("load {}".format(cameras_json_path))
        with open(cameras_json_path, "r") as f:
            cameras = json.load(f)
        up = torch.zeros(3)
        for i in cameras:
            up += torch.tensor(i["rotation"])[:3, 1]
        up = -up / torch.linalg.norm(up)

        print("up vector = {}".format(up))
        self.up_direction = up.numpy()

        return transform

        # rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
        # transform[:3, :3] = rotation
        # transform = torch.linalg.inv(transform)
        #
        # return transform

    def load_camera_poses(self, cameras_json_path: str):
        if os.path.exists(cameras_json_path) is False:
            return []
        with open(cameras_json_path, "r") as f:
            return json.load(f)

    def add_cameras_to_scene(self, viser_server):
        if len(self.camera_poses) == 0:
            return

        self.camera_handles = []

        camera_pose_transform = np.linalg.inv(self.camera_transform.cpu().numpy())
        for camera in self.camera_poses:
            name = camera["img_name"]
            c2w = np.eye(4)
            c2w[:3, :3] = np.asarray(camera["rotation"])
            c2w[:3, 3] = np.asarray(camera["position"])
            c2w[:3, 1:3] *= -1
            c2w = np.matmul(camera_pose_transform, c2w)

            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)

            cx = camera["width"] // 2
            cy = camera["height"] // 2
            fx = camera["fx"]

            camera_handle = viser_server.add_camera_frustum(
                name="cameras/{}".format(name),
                fov=float(2 * np.arctan(cx / fx)),
                scale=0.1,
                aspect=float(cx / cy),
                wxyz=R.wxyz,
                position=c2w[:3, 3],
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

        # def update_camera_scale(_):
        #     with viser_server.atomic():
        #         for i in self.camera_handles:
        #             i.scale = self.camera_scale_slider.value

        with viser_server.add_gui_folder("Cameras"):
            self.toggle_camera_button = viser_server.add_gui_button("Toggle Camera Visibility")
            # self.camera_scale_slider = viser_server.add_gui_slider(
            #     "Camera Scale",
            #     min=0.,
            #     max=1.,
            #     step=0.01,
            #     initial_value=0.1,
            # )
        self.toggle_camera_button.on_click(toggle_camera_visibility)
        # self.camera_scale_slider.on_update(update_camera_scale)

    @staticmethod
    def _do_initialize_models_from_checkpoint(checkpoint_path: str, device):
        return GaussianModelLoader.initialize_simplified_model_from_checkpoint(checkpoint_path, device)

    def _initialize_models_from_checkpoint(self, checkpoint_path: str):
        return self._do_initialize_models_from_checkpoint(checkpoint_path, self.device)

    @staticmethod
    def _do_initialize_models_from_point_cloud(point_cloud_path: str, sh_degree, device, simplified: bool = True):
        if simplified is True:
            return GaussianModelLoader.initialize_simplified_model_from_point_cloud(point_cloud_path, sh_degree, device)
        from internal.models.gaussian_model import GaussianModel
        model = GaussianModel(sh_degree=sh_degree)
        model.load_ply(point_cloud_path, device=device)
        return model, VanillaRenderer()

    def _initialize_models_from_point_cloud(self, point_cloud_path: str):
        return self._do_initialize_models_from_point_cloud(point_cloud_path, self.sh_degree, self.device, simplified=self.simplified_model)

    def _load_model_from_file(self, load_from: str):
        print("load model from {}".format(load_from))
        checkpoint = None
        dataset_type = None
        if load_from.endswith(".ckpt") is True:
            model, renderer, checkpoint = self._initialize_models_from_checkpoint(load_from)
            training_output_base_dir = os.path.dirname(os.path.dirname(load_from))
            dataset_type = checkpoint["datamodule_hyper_parameters"]["type"]
            self.sh_degree = model.max_sh_degree
        elif load_from.endswith(".ply") is True:
            model, renderer = self._initialize_models_from_point_cloud(load_from)
            training_output_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(load_from)))
            if self.use_gsplat is True:
                from internal.renderers.gsplat_renderer import GSPlatRenderer
                print("Use GSPlat renderer for ply file")
                renderer = GSPlatRenderer()
        else:
            raise ValueError("unsupported file {}".format(load_from))

        return model, renderer, training_output_base_dir, dataset_type, checkpoint

    def start(self, block: bool = True, server_config_fun=None, tab_config_fun=None):
        # create viser server
        server = viser.ViserServer(host=self.host, port=self.port)
        server.configure_theme(
            control_layout="collapsible",
            show_logo=False,
        )

        if server_config_fun is not None:
            server_config_fun(self, server)

        tabs = server.add_gui_tab_group()

        if tab_config_fun is not None:
            tab_config_fun(self, server, tabs)

        with tabs.add_tab("General"):
            # add render options
            with server.add_gui_folder("Render"):
                self.max_res_when_static = server.add_gui_slider(
                    "Max Res",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1920,
                )
                self.max_res_when_static.on_update(self._handle_option_updated)
                self.jpeg_quality_when_static = server.add_gui_slider(
                    "JPEG Quality",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=100,
                )
                self.jpeg_quality_when_static.on_update(self._handle_option_updated)

                self.max_res_when_moving = server.add_gui_slider(
                    "Max Res when Moving",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1280,
                )
                self.jpeg_quality_when_moving = server.add_gui_slider(
                    "JPEG Quality when Moving",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=60,
                )

            self.viewer_renderer.setup_options(self, server)

            with server.add_gui_folder("Model"):
                self.scaling_modifier = server.add_gui_slider(
                    "Scaling Modifier",
                    min=0.,
                    max=1.,
                    step=0.1,
                    initial_value=1.,
                )
                self.scaling_modifier.on_update(self._handle_option_updated)

                if self.viewer_renderer.gaussian_model.max_sh_degree > 0:
                    self.active_sh_degree_slider = server.add_gui_slider(
                        "Active SH Degree",
                        min=0,
                        max=self.viewer_renderer.gaussian_model.max_sh_degree,
                        step=1,
                        initial_value=self.viewer_renderer.gaussian_model.max_sh_degree,
                    )
                    self.active_sh_degree_slider.on_update(self._handle_activate_sh_degree_slider_updated)

                if self.available_appearance_options is not None:
                    # find max appearance id
                    max_input_id = 0
                    available_option_values = list(self.available_appearance_options.values())
                    if isinstance(available_option_values[0], list) or isinstance(available_option_values[0], tuple):
                        for i in available_option_values:
                            if i[0] > max_input_id:
                                max_input_id = i[0]
                    else:
                        # convert to tuple, compatible with previous version
                        for i in self.available_appearance_options:
                            self.available_appearance_options[i] = (0, self.available_appearance_options[i])
                    self.available_appearance_options[DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE] = None

                    self.appearance_id = server.add_gui_slider(
                        "Appearance Direct",
                        min=0,
                        max=max_input_id,
                        step=1,
                        initial_value=0,
                        visible=max_input_id > 0
                    )

                    self.normalized_appearance_id = server.add_gui_slider(
                        "Normalized Appearance Direct",
                        min=0.,
                        max=1.,
                        step=0.01,
                        initial_value=0.,
                    )

                    appearance_options = list(self.available_appearance_options.keys())

                    self.appearance_group_dropdown = server.add_gui_dropdown(
                        "Appearance Group",
                        options=appearance_options,
                        initial_value=appearance_options[0],
                    )
                    self.appearance_id.on_update(self._handle_appearance_embedding_slider_updated)
                    self.normalized_appearance_id.on_update(self._handle_appearance_embedding_slider_updated)
                    self.appearance_group_dropdown.on_update(self._handel_appearance_group_dropdown_updated)

                self.time_slider = server.add_gui_slider(
                    "Time",
                    min=0.,
                    max=1.,
                    step=0.01,
                    initial_value=0.,
                )
                self.time_slider.on_update(self._handle_option_updated)

            # add cameras
            if self.show_cameras is True:
                self.add_cameras_to_scene(server)

            UpDirectionFolder(self, server)

            go_to_scene_center = server.add_gui_button(
                "Go to scene center",
            )

            @go_to_scene_center.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                event.client.camera.position = self.camera_center + np.asarray([2., 0., 0.])
                event.client.camera.look_at = self.camera_center

        if self.show_edit_panel is True:
            with tabs.add_tab("Edit") as edit_tab:
                self.edit_panel = EditPanel(server, self, edit_tab)

        self.transform_panel: TransformPanel = None
        if self.enable_transform is True:
            with tabs.add_tab("Transform"):
                self.transform_panel = TransformPanel(server, self, self.loaded_model_count)

        if self.show_render_panel is True:
            with tabs.add_tab("Render"):
                populate_render_tab(
                    server,
                    self,
                    self.model_paths,
                    Path("./"),
                    orientation_transform=torch.linalg.inv(self.camera_transform).cpu().numpy(),
                    enable_transform=self.enable_transform,
                    background_color=self.background_color,
                    sh_degree=self.sh_degree,
                    extra_args=self.extra_video_render_args,
                )

        # register hooks
        server.on_client_connect(self._handle_new_client)
        server.on_client_disconnect(self._handle_client_disconnect)

        if block is True:
            while True:
                time.sleep(999)

    def _handle_appearance_embedding_slider_updated(self, event: viser.GuiEvent):
        """
        Change appearance group dropdown to "@Direct" on slider updated
        """

        if event.client is None:  # skip if not updated by client
            return
        self.appearance_group_dropdown.value = DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE
        self._handle_option_updated(event)

    def _handle_activate_sh_degree_slider_updated(self, _):
        self.viewer_renderer.gaussian_model.active_sh_degree = self.active_sh_degree_slider.value
        self._handle_option_updated(_)

    def get_appearance_id_value(self):
        """
        Return appearance id according to the slider and dropdown value
        """

        # no available appearance options, simply return zero
        if self.available_appearance_options is None:
            return (0, 0.)
        name = self.appearance_group_dropdown.value
        # if the value of dropdown is "@Direct", or not in available_appearance_options, return the slider's values
        if name == DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE or name not in self.available_appearance_options:
            return (self.appearance_id.value, self.normalized_appearance_id.value)
        # else return the values according to the dropdown
        return self.available_appearance_options[name]

    def _handel_appearance_group_dropdown_updated(self, event: viser.GuiEvent):
        """
        Update slider's values when dropdown updated
        """

        if event.client is None:  # skip if not updated by client
            return

        # get appearance ids according to the dropdown value
        appearance_id, normalized_appearance_id = self.available_appearance_options[self.appearance_group_dropdown.value]
        # update sliders
        self.appearance_id.value = appearance_id
        self.normalized_appearance_id.value = normalized_appearance_id
        # rerender
        self._handle_option_updated(event)

    def _handle_option_updated(self, _):
        """
        Simply push new render to all client
        """
        return self.rerender_for_all_client()

    def handle_option_updated(self, _):
        return self._handle_option_updated(_)

    def rerender_for_client(self, client_id: int):
        """
        Render for specific client
        """
        try:
            # switch to low resolution mode first, then notify the client to render
            self.clients[client_id].state = "low"
            self.clients[client_id].render_trigger.set()
        except:
            # ignore errors
            pass

    def rerender_for_all_client(self):
        for i in self.clients:
            self.rerender_for_client(i)

    def _handle_new_client(self, client: viser.ClientHandle) -> None:
        """
        Create and start a thread for every new client
        """

        # create client thread
        client_thread = ClientThread(self, self.viewer_renderer, client)
        client_thread.start()
        # store this thread
        self.clients[client.client_id] = client_thread

    def _handle_client_disconnect(self, client: viser.ClientHandle):
        """
        Destroy client thread when client disconnected
        """

        try:
            self.clients[client.client_id].stop()
            del self.clients[client.client_id]
        except Exception as err:
            print(err)


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_paths", type=str, nargs="+")
    parser.add_argument("--host", "-a", type=str, default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--background_color", "--background_color", "--bkg_color", "-b",
                        type=str, nargs="+", default=["black"],
                        help="e.g.: white, black, 0 0 0, 1 1 1")
    parser.add_argument("--image_format", "--image-format", "-f", type=str, default="jpeg")
    parser.add_argument("--reorient", "-r", type=str, default="auto",
                        help="whether reorient the scene, available values: auto, enable, disable")
    parser.add_argument("--sh_degree", "--sh-degree", "--sh",
                        type=int, default=3)
    parser.add_argument("--enable_transform", "--enable-transform",
                        action="store_true", default=False,
                        help="Enable transform options on Web UI. May consume more memory")
    parser.add_argument("--show_cameras", "--show-cameras",
                        action="store_true")
    parser.add_argument("--cameras-json", "--cameras_json", type=str, default=None)
    parser.add_argument("--vanilla_deformable", action="store_true", default=False)
    parser.add_argument("--vanilla_gs4d", action="store_true", default=False)
    parser.add_argument("--vanilla_gs2d", action="store_true", default=False)
    parser.add_argument("--up", nargs=3, required=False, type=float, default=None)
    parser.add_argument("--default_camera_position", "--dcp", nargs=3, required=False, type=float, default=None)
    parser.add_argument("--default_camera_look_at", "--dcla", nargs=3, required=False, type=float, default=None)
    parser.add_argument("--no_edit_panel", action="store_true", default=False)
    parser.add_argument("--no_render_panel", action="store_true", default=False)
    parser.add_argument("--gsplat", action="store_true", default=False,
                        help="Use GSPlat renderer for ply file")
    parser.add_argument("--float32_matmul_precision", "--fp", type=str, default=None)
    args = parser.parse_args()

    # set torch float32_matmul_precision
    if args.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
    del args.float32_matmul_precision

    # arguments post process
    if len(args.background_color) == 1 and isinstance(args.background_color[0], str):
        if args.background_color[0] == "white":
            args.background_color = (1., 1., 1.)
        else:
            args.background_color = (0., 0., 0.)
    else:
        args.background_color = tuple([float(i) for i in args.background_color])

    # create viewer
    viewer_init_args = {key: getattr(args, key) for key in vars(args)}
    viewer = Viewer(**viewer_init_args)

    # start viewer server
    viewer.start()
