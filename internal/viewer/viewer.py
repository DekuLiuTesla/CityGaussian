import os
from pathlib import Path
import time
import json
from typing import Tuple, Literal, List

import numpy as np
import viser
import viser.transforms as vtf
import torch
import yaml

from internal.renderers import VanillaRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gaussian_model_editor import MultipleGaussianModelEditor
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
            gsplat_aa: bool = False,
            gsplat_v1_example: bool = False,
            gsplat_v1_example_aa: bool = False,
            seganygs: str = None,
            vanilla_seganygs: bool = False,
            vanilla_mip: bool = False,
            vanilla_pvg: bool = False,
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
        self.use_gsplat_aa = gsplat_aa

        self.simplified_model = True
        self.show_edit_panel = True
        if no_edit_panel is True:
            self.show_edit_panel = False
        self.show_render_panel = True
        if no_render_panel is True:
            self.show_render_panel = False

        def turn_off_edit_and_video_render_panel():
            self.show_edit_panel = False
            self.show_render_panel = False

        if gsplat_v1_example is True:
            from internal.utils.gaussian_model_loader import GSplatV1ExampleCheckpointLoader
            model, renderer = GSplatV1ExampleCheckpointLoader.load(model_paths[0], self.device, anti_aliased=gsplat_v1_example_aa)
            training_output_base_dir = model_paths[0]
            dataset_type = "Colmap"
        elif vanilla_pvg is True:
            from internal.utils.gaussian_model_loader import VanillaPVGModelLoader
            model, renderer = VanillaPVGModelLoader.search_and_load(model_paths[0], self.device)
            training_output_base_dir = model_paths[0]
            self.checkpoint = None
            dataset_type = "kitti"
            turn_off_edit_and_video_render_panel()
        elif model_paths[0].endswith(".yaml"):
            self.show_edit_panel = False
            self.enable_transform = False
            from internal.models.vanilla_gaussian import VanillaGaussian
            model = VanillaGaussian().instantiate()
            model.setup_from_number(0)
            model.pre_activate_all_properties()
            model.eval()
            from internal.renderers.partition_lod_renderer import PartitionLoDRenderer
            with open(model_paths[0], "r") as f:
                lod_config = yaml.safe_load(f)
            renderer = PartitionLoDRenderer(**lod_config).instantiate()
            renderer.setup("validation")
            training_output_base_dir = os.getcwd()
            dataset_type = "Colmap"
        else:
            load_from = self._search_load_file(model_paths[0])

            # detect whether a SegAnyGS output
            if seganygs is None and load_from.endswith(".ckpt"):
                seganygs_tag_file_path = os.path.join(os.path.dirname(os.path.dirname(load_from)), "seganygs")
                if os.path.exists(seganygs_tag_file_path) is True:
                    print("SegAny Splatting model detected")
                    seganygs = load_from
                    with open(seganygs_tag_file_path, "r") as f:
                        load_from = self._search_load_file(f.read())

            if vanilla_seganygs is True:
                load_from = load_from[:-len(os.path.basename(load_from))]
                load_from = os.path.join(load_from, "scene_point_cloud.ply")

            # whether model is trained by other implementations
            if vanilla_gs4d is True:
                self.simplified_model = False

            # TODO: load multiple models more elegantly
            # load and create models
            model, renderer, training_output_base_dir, dataset_type, self.checkpoint = self._load_model_from_file(load_from)
            if renderer.__class__.__name__ == "GSplatDistributedRendererImpl":
                print(f"[WARNING] You are loading a subset of Gaussians generated by distributed training. If this is not expected, merge them with `utils/merge_distributed_ckpts.py` first.")
                from internal.renderers.gsplat_renderer import GSPlatRenderer
                renderer = GSPlatRenderer()
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
                turn_off_edit_and_video_render_panel()
            elif vanilla_gs4d is True:
                from internal.renderers.vanilla_gs4d_renderer import VanillaGS4DRenderer
                renderer = VanillaGS4DRenderer(
                    os.path.dirname(os.path.dirname(os.path.dirname(load_from))),
                    get_load_iteration(),
                    device=self.device,
                )
                turn_off_edit_and_video_render_panel()
            elif vanilla_gs2d is True:
                from internal.renderers.vanilla_2dgs_renderer import Vanilla2DGSRenderer
                renderer = Vanilla2DGSRenderer()
                self.extra_video_render_args.append("--vanilla_gs2d")
            elif vanilla_seganygs is True:
                renderer = self._load_vanilla_seganygs(load_from)
                turn_off_edit_and_video_render_panel()
            elif vanilla_mip is True:
                renderer = self._load_vanilla_mip(load_from)
                turn_off_edit_and_video_render_panel()

        # reorient the scene
        cameras_json_path = cameras_json
        if cameras_json_path is None:
            cameras_json_path = os.path.join(training_output_base_dir, "cameras.json")
        self.camera_transform = self._reorient(cameras_json_path, mode=reorient, dataset_type=dataset_type)
        if up is not None:
            self.camera_transform = torch.eye(4, dtype=torch.float)
            up = torch.tensor(up)
            up = up / torch.linalg.norm(up)
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

            model.freeze()

            if self.show_edit_panel is True or enable_transform is True:
                model = MultipleGaussianModelEditor([model], device=self.device)
        else:
            # switch to vanilla renderer
            model_list = [model.to(torch.device("cpu"))]
            renderer = VanillaRenderer()
            for model_path in addition_models:
                load_from = self._search_load_file(model_path)
                if load_from.endswith(".ckpt"):
                    load_results = self._do_initialize_models_from_checkpoint(load_from, device=torch.device("cpu"))
                else:
                    load_results = self._do_initialize_models_from_point_cloud(load_from, self.sh_degree, device=torch.device("cpu"))
                model_list.append(load_results[0])

            self.loaded_model_count += len(addition_models)

            for i in model_list:
                i.freeze()

            model = MultipleGaussianModelEditor(model_list, device=self.device)

        self.gaussian_model = model

        if seganygs is not None:
            print("loading SegAnyGaussian...")
            renderer = self._load_seganygs(seganygs)
            turn_off_edit_and_video_render_panel()

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

    def _load_seganygs(self, path):
        load_from = self._search_load_file(path)
        assert load_from.endswith(".ckpt")
        ckpt = torch.load(load_from, map_location="cpu")

        from internal.segany_splatting import SegAnySplatting
        model = SegAnySplatting(**ckpt["hyper_parameters"])
        model.setup_parameters(ckpt["state_dict"]["gaussian_semantic_features"].shape[0])
        model.load_state_dict(ckpt["state_dict"])
        model.on_load_checkpoint(ckpt)
        model.eval()

        semantic_features = model.get_processed_semantic_features()
        scale_gate = model.scale_gate
        del model
        del ckpt
        torch.cuda.empty_cache()

        from internal.renderers.seganygs_renderer import SegAnyGSRenderer
        return SegAnyGSRenderer(semantic_features=semantic_features, scale_gate=scale_gate)

    def _load_vanilla_seganygs(self, path):
        from plyfile import PlyData
        from internal.utils.gaussian_utils import GaussianPlyUtils

        max_iteration = -1
        load_from = None
        model_output = os.path.dirname(os.path.dirname(path))
        for i in os.listdir(model_output):
            if i.startswith("iteration_") is False:
                continue
            ply_file_path = os.path.join(model_output, i, "contrastive_feature_point_cloud.ply")
            if os.path.exists(ply_file_path) is False:
                continue
            try:
                iteration = int(i.split("_")[1])
            except:
                break
            if iteration > max_iteration:
                load_from = ply_file_path
        assert load_from is not None, "'contrastive_feature_point_cloud.ply' not found"
        print(f"load SegAnyGS from '{load_from}'...")

        plydata = PlyData.read(load_from)
        semantic_features = torch.tensor(
            GaussianPlyUtils.load_array_from_plyelement(plydata.elements[0], "f_"),
            dtype=torch.float,
            device=self.device,
        )

        scale_gate_state_dict = torch.load(os.path.join(os.path.dirname(load_from), "scale_gate.pt"), map_location="cpu")
        scale_gate = torch.nn.Sequential(
            torch.nn.Linear(1, 32, bias=True),
            torch.nn.Sigmoid()
        )
        scale_gate.load_state_dict(scale_gate_state_dict)
        scale_gate = scale_gate.to(self.device)

        from internal.renderers.seganygs_renderer import SegAnyGSRenderer
        return SegAnyGSRenderer(semantic_features=semantic_features, scale_gate=scale_gate, anti_aliased=False)

    def _load_vanilla_mip(self, load_from):
        from plyfile import PlyData
        from internal.renderers.mip_splatting_gsplat_renderer import MipSplattingGSplatRenderer

        plydata = PlyData.read(load_from)
        filter_3d = torch.nn.Parameter(torch.tensor(
            plydata.elements[0]["filter_3D"][..., np.newaxis],
            dtype=torch.float,
            device=self.device,
        ), requires_grad=False)

        # TODO: read `kernel_size` from cfg_args
        renderer = MipSplattingGSplatRenderer()
        renderer.filter_3d = filter_3d

        return renderer

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
        if dataset_type in ["blender", "nsvf", "matrixcity"] and mode == "auto":
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

            camera_handle = viser_server.scene.add_camera_frustum(
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

        with viser_server.gui.add_folder("Cameras"):
            self.toggle_camera_button = viser_server.gui.add_button("Toggle Camera Visibility")
            # self.camera_scale_slider = viser_server.gui.add_slider(
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
        return GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(checkpoint_path, device)

    def _initialize_models_from_checkpoint(self, checkpoint_path: str):
        return self._do_initialize_models_from_checkpoint(checkpoint_path, self.device)

    @staticmethod
    def _do_initialize_models_from_point_cloud(point_cloud_path: str, sh_degree, device, simplified: bool = True):
        return GaussianModelLoader.initialize_model_and_renderer_from_ply_file(point_cloud_path, device, pre_activate=simplified)

    def _initialize_models_from_point_cloud(self, point_cloud_path: str):
        return self._do_initialize_models_from_point_cloud(point_cloud_path, self.sh_degree, self.device, simplified=self.simplified_model)

    def _load_model_from_file(self, load_from: str):
        print("load model from {}".format(load_from))
        checkpoint = None
        dataset_type = None
        if load_from.endswith(".ckpt") is True:
            model, renderer, checkpoint = self._initialize_models_from_checkpoint(load_from)
            training_output_base_dir = os.path.dirname(os.path.dirname(load_from))

            # get dataset type
            dataset_type = checkpoint["datamodule_hyper_parameters"].get("type", None)  # previous version
            # new version
            if dataset_type is None:
                try:
                    dataset_type = checkpoint["datamodule_hyper_parameters"].get("parser", None).__class__.__name__.lower()
                except:
                    dataset_type = ""

            self.sh_degree = model.max_sh_degree
        elif load_from.endswith(".ply") is True:
            model, renderer = self._initialize_models_from_point_cloud(load_from)
            training_output_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(load_from)))
            if self.use_gsplat is True:
                from internal.renderers.gsplat_v1_renderer import GSplatV1Renderer
                print("Use gsplat v1 renderer(AA={}) for ply file".format(self.use_gsplat_aa))
                renderer = GSplatV1Renderer(anti_aliased=self.use_gsplat_aa).instantiate()
        else:
            raise ValueError("unsupported file {}".format(load_from))

        return model, renderer, training_output_base_dir, dataset_type, checkpoint

    def show_message(self, message: str, client=None):
        target = client
        if client is None:
            target = self._server
        with target.gui.add_modal("Message") as modal:
            target.gui.add_markdown(message)
            close_button = target.gui.add_button("Close")

            @close_button.on_click
            def _(_) -> None:
                try:
                    modal.close()
                except:
                    pass

    def start(self, block: bool = True, server_config_fun=None, tab_config_fun=None, enable_renderer_options: bool = True):
        # create viser server
        server = viser.ViserServer(host=self.host, port=self.port)
        server.scene.set_up_direction(self.up_direction)
        self._server = server
        server.gui.configure_theme(
            control_layout="collapsible",
            show_logo=False,
        )

        if server_config_fun is not None:
            server_config_fun(self, server)

        tabs = server.gui.add_tab_group()

        if tab_config_fun is not None:
            tab_config_fun(self, server, tabs)

        with tabs.add_tab("General"):
            # add render options
            with server.gui.add_folder("Render"):
                self.max_res_when_static = server.gui.add_slider(
                    "Max Res",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1920,
                )
                self.max_res_when_static.on_update(self._handle_option_updated)
                self.jpeg_quality_when_static = server.gui.add_slider(
                    "JPEG Quality",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=100,
                )
                self.jpeg_quality_when_static.on_update(self._handle_option_updated)

                self.max_res_when_moving = server.gui.add_slider(
                    "Max Res when Moving",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1280,
                )
                self.jpeg_quality_when_moving = server.gui.add_slider(
                    "JPEG Quality when Moving",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=60,
                )

            self.viewer_renderer.setup_options(self, server)

            with server.gui.add_folder("Model"):
                self.scaling_modifier = server.gui.add_slider(
                    "Scaling Modifier",
                    min=0.,
                    max=1.,
                    step=0.01,
                    initial_value=1.,
                )
                self.scaling_modifier.on_update(self._handle_option_updated)

                if self.viewer_renderer.gaussian_model.max_sh_degree > 0:
                    self.active_sh_degree_slider = server.gui.add_slider(
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

                    self.appearance_id = server.gui.add_slider(
                        "Appearance Direct",
                        min=0,
                        max=max_input_id,
                        step=1,
                        initial_value=0,
                        visible=max_input_id > 0
                    )

                    self.normalized_appearance_id = server.gui.add_slider(
                        "Normalized Appearance Direct",
                        min=0.,
                        max=1.,
                        step=0.01,
                        initial_value=0.,
                    )

                    appearance_options = list(self.available_appearance_options.keys())

                    self.appearance_group_dropdown = server.gui.add_dropdown(
                        "Appearance Group",
                        options=appearance_options,
                        initial_value=appearance_options[0],
                    )
                    self.appearance_id.on_update(self._handle_appearance_embedding_slider_updated)
                    self.normalized_appearance_id.on_update(self._handle_appearance_embedding_slider_updated)
                    self.appearance_group_dropdown.on_update(self._handel_appearance_group_dropdown_updated)

                self.time_slider = server.gui.add_slider(
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

            go_to_scene_center = server.gui.add_button(
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

        if enable_renderer_options is True:
            self.viewer_renderer.renderer.setup_web_viewer_tabs(self, server, tabs)

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
        v = self.available_appearance_options.get(self.appearance_group_dropdown.value, None)
        if v is None:
            return
        appearance_id, normalized_appearance_id = v
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
