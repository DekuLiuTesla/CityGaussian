import os
import glob
import torch
from typing import Tuple
from internal.models.gaussian import Gaussian
from internal.renderers import RendererConfig
from internal.renderers.vanilla_renderer import VanillaRenderer


class GaussianModelLoader:
    previous_name_to_new = {
        "_xyz": "means",
        "_features_dc": "shs_dc",
        "_features_rest": "shs_rest",
        "_scaling": "scales",
        "_rotation": "rotations",
        "_opacity": "opacities",
        "_features_extra": "appearance_features",
    }

    previous_state_dict_key_to_new = {
        "_xyz": "gaussians.means",
        "_features_dc": "gaussians.shs_dc",
        "_features_rest": "gaussians.shs_rest",
        "_scaling": "gaussians.scales",
        "_rotation": "gaussians.rotations",
        "_opacity": "gaussians.opacities",
        "_features_extra": "gaussians.appearance_features",
    }

    @staticmethod
    def search_load_file(model_path: str) -> str:
        # if a directory path is provided, auto search checkpoint or ply
        if os.path.isdir(model_path) is False:
            return model_path
        # search checkpoint
        checkpoint_dir = os.path.join(model_path, "checkpoints")
        # find checkpoint with max iterations
        load_from = None
        previous_checkpoint_iteration = -1
        for i in glob.glob(os.path.join(checkpoint_dir, "*.ckpt")):
            try:
                checkpoint_iteration = int(i[i.rfind("=") + 1:i.rfind(".")])
            except Exception as err:
                print("error occurred when parsing iteration from {}: {}".format(i, err))
                continue
            if checkpoint_iteration > previous_checkpoint_iteration:
                previous_checkpoint_iteration = checkpoint_iteration
                load_from = i

        # not a checkpoint can be found, search point cloud
        if load_from is None:
            previous_point_cloud_iteration = -1
            for i in glob.glob(os.path.join(model_path, "point_cloud", "iteration_*")):
                try:
                    point_cloud_iteration = int(os.path.basename(i).replace("iteration_", ""))
                except Exception as err:
                    print("error occurred when parsing iteration from {}: {}".format(i, err))
                    continue

                if point_cloud_iteration > previous_point_cloud_iteration:
                    previous_point_cloud_iteration = point_cloud_iteration
                    load_from = os.path.join(i, "point_cloud.ply")

        assert load_from is not None, "not a checkpoint or point cloud can be found"

        return load_from

    @staticmethod
    def filter_state_dict_by_prefix(state_dict, prefix: str, device=None):
        prefix_len = len(prefix)
        new_state_dict = state_dict.__class__()
        for name in state_dict:
            if name.startswith(prefix):
                state = state_dict[name]
                if device is not None:
                    state = state.to(device)
                new_state_dict[name[prefix_len:]] = state
        return new_state_dict

    @classmethod
    def initialize_model_from_checkpoint(cls, checkpoint: dict, device):
        hparams = checkpoint["hyper_parameters"]

        if isinstance(hparams["gaussian"], Gaussian):
            model = hparams["gaussian"].instantiate()
            model_state_dict = cls.filter_state_dict_by_prefix(checkpoint["state_dict"], "gaussian_model.", device=device)
        else:
            # convert previous checkpoint to new format
            sh_degree = hparams["gaussian"].sh_degree
            model_state_dict = cls.filter_state_dict_by_prefix(checkpoint["state_dict"], "gaussian_model.", device=device)
            model_state_dict = {cls.previous_state_dict_key_to_new.get(name): model_state_dict[name] for name in model_state_dict}
            model_state_dict["_active_sh_degree"] = torch.tensor(sh_degree, dtype=torch.int, device=device)

            from internal.models.vanilla_gaussian import VanillaGaussian

            if "gaussians.appearance_features" in model_state_dict:
                if model_state_dict["gaussians.appearance_features"].shape[-1] > 0:
                    from internal.models.appearance_feature_gaussian import AppearanceFeatureGaussian
                    model = AppearanceFeatureGaussian(sh_degree=sh_degree, appearance_feature_dims=model_state_dict["gaussians.appearance_features"].shape[-1]).instantiate()
                else:
                    del model_state_dict["gaussians.appearance_features"]
                    model = VanillaGaussian(sh_degree=sh_degree).instantiate()
            else:
                model = VanillaGaussian(sh_degree=sh_degree).instantiate()

        model.setup_from_number(model_state_dict["gaussians.means"].shape[0])
        model.to(device)
        model.load_state_dict(model_state_dict)

        return model

    @classmethod
    def initialize_renderer_from_checkpoint(cls, checkpoint: dict, stage: str, device):
        hparams = checkpoint["hyper_parameters"]
        # extract state dict of renderer
        renderer = hparams["renderer"]
        renderer_state_dict = cls.filter_state_dict_by_prefix(checkpoint["state_dict"], "renderer.", device=device)
        # load state dict of renderer
        if isinstance(renderer, RendererConfig):
            renderer = renderer.instantiate()
            renderer.setup(stage=stage)
        renderer = renderer.to(device)
        renderer.load_state_dict(renderer_state_dict)
        return renderer

    @classmethod
    def initialize_model_and_renderer_from_checkpoint_file(
            cls,
            checkpoint_path: str,
            device,
            eval_mode: bool = True,
            pre_activate: bool = True,
    ) -> Tuple[torch.nn.Module, torch.nn.Module, dict]:
        stage = "fit"
        if eval_mode is True:
            stage = "validation"

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model = cls.initialize_model_from_checkpoint(checkpoint, device)
        renderer = cls.initialize_renderer_from_checkpoint(checkpoint, stage, device)

        if eval_mode is True:
            model.eval()
            renderer.eval()

        if pre_activate is True:
            model.pre_activate_all_properties()

        return model, renderer, checkpoint

    @staticmethod
    def initialize_model_and_renderer_from_ply_file(ply_file_path: str, device, eval_mode: bool = True, pre_activate: bool = True):
        from internal.utils.gaussian_utils import GaussianPlyUtils
        gaussian_ply_utils = GaussianPlyUtils.load_from_ply(ply_file_path).to_parameter_structure()
        model_state_dict = {
            "_active_sh_degree": torch.tensor(gaussian_ply_utils.sh_degrees, dtype=torch.int, device=device),
            "gaussians.means": gaussian_ply_utils.xyz.to(device),
            "gaussians.opacities": gaussian_ply_utils.opacities.to(device),
            "gaussians.shs_dc": gaussian_ply_utils.features_dc.to(device),
            "gaussians.shs_rest": gaussian_ply_utils.features_rest.to(device),
            "gaussians.scales": gaussian_ply_utils.scales.to(device),
            "gaussians.rotations": gaussian_ply_utils.rotations.to(device),
        }

        # use 2DGS if the number of last dim is 2
        if model_state_dict["gaussians.scales"].shape[-1] == 2:
            from internal.models.gaussian_2d import Gaussian2D
            model = Gaussian2D(sh_degree=gaussian_ply_utils.sh_degrees).instantiate()
            from internal.renderers.vanilla_2dgs_renderer import Vanilla2DGSRenderer
            renderer_type = Vanilla2DGSRenderer
        else:
            from internal.models.vanilla_gaussian import VanillaGaussian
            model = VanillaGaussian(sh_degree=gaussian_ply_utils.sh_degrees).instantiate()
            renderer_type = VanillaRenderer
        model.setup_from_number(gaussian_ply_utils.xyz.shape[0])
        model.to(device)
        model.load_state_dict(model_state_dict, strict=False)

        renderer = renderer_type().to(device)

        if eval_mode is True:
            renderer.setup("validation")
            model.eval()
        else:
            renderer.setup("fit")

        if pre_activate is True:
            model.pre_activate_all_properties()

        return model, renderer

    @classmethod
    def search_and_load(
            cls,
            model_path: str,
            device,
            eval_mode: bool = True,
            pre_activate: bool = True,
    ):
        load_from = cls.search_load_file(model_path)
        if load_from.endswith(".ckpt"):
            model, renderer, _ = cls.initialize_model_and_renderer_from_checkpoint_file(
                load_from,
                device=device,
                eval_mode=eval_mode,
                pre_activate=pre_activate,
            )
        elif load_from.endswith(".ply"):
            model, renderer = cls.initialize_model_and_renderer_from_ply_file(
                load_from,
                device=device,
                eval_mode=eval_mode,
                pre_activate=pre_activate,
            )
        else:
            raise ValueError("unsupported file {}".format(load_from))

        return model, renderer


class VanillaPVGModelLoader:
    @staticmethod
    def state_tuple_to_state_dict_and_model_config(state_tuple, device):
        (
            active_sh_degree,
            _xyz,
            _features_dc,
            _features_rest,
            _scaling,
            _rotation,
            _opacity,
            _t,
            _scaling_t,
            _velocity,
            max_radii2D,
            xyz_gradient_accum,
            t_gradient_accum,
            denom,
            opt_dict,
            spatial_lr_scale,
            T,
            velocity_decay,
        ) = state_tuple

        state_dict = {
            "_active_sh_degree": torch.tensor(active_sh_degree, dtype=torch.int, device=device),
            "gaussians.means": _xyz.to(device),
            "gaussians.shs_dc": _features_dc.to(device),
            "gaussians.shs_rest": _features_rest.to(device),
            "gaussians.scales": _scaling.to(device),
            "gaussians.rotations": _rotation.to(device),
            "gaussians.opacities": _opacity.to(device),
            "gaussians.t": _t.to(device),
            "gaussians.scale_t": _scaling_t.to(device),
            "gaussians.velocity": _velocity.to(device),
        }

        from internal.models.periodic_vibration_gaussian import PeriodicVibrationGaussian

        return state_dict, PeriodicVibrationGaussian(
            sh_degree=active_sh_degree,
            cycle=T,
            velocity_decay=velocity_decay,
        )

    @classmethod
    def search_and_load(cls, path, device):
        max_iteration = -1
        for i in glob.glob(os.path.join(path, "chkpnt*.pth")):
            basename = os.path.basename(i)
            iteration = int(basename[len("chkpnt"):basename.rfind(".")])
            if iteration > max_iteration:
                max_iteration = iteration

        model_ckpt_file = os.path.join(path, "chkpnt{}.pth".format(max_iteration))
        env_light_ckpt_file = os.path.join(path, "env_light_chkpnt{}.pth".format(max_iteration))

        model_ckpt = torch.load(model_ckpt_file, map_location="cpu")
        model_state_dict, model_config = cls.state_tuple_to_state_dict_and_model_config(model_ckpt[0], device)
        model = model_config.instantiate()
        model.setup_from_number(model_state_dict["gaussians.means"].shape[0])
        model.to(device)
        model.load_state_dict(model_state_dict)
        model.eval()

        from internal.renderers.periodic_vibration_gaussian_renderer import PeriodicVibrationGaussianRenderer

        if os.path.exists(env_light_ckpt_file):
            renderer = PeriodicVibrationGaussianRenderer(anti_aliased=False).instantiate()
            renderer.setup("validation")
            env_ckpt = torch.load(env_light_ckpt_file, map_location="cpu")
            renderer.env_map.base = env_ckpt[0][0]
        else:
            renderer = PeriodicVibrationGaussianRenderer(env_map_res=-1, anti_aliased=False).instantiate()
            renderer.setup("validation")
        renderer.to(device)
        renderer.eval()

        return model, renderer


class GSplatV1ExampleCheckpointLoader:
    @classmethod
    def load_from_ckpt(cls, ckpt, device, anti_aliased: bool = True, eval_mode: bool = True, pre_activate: bool = True):
        gaussian_state_dict = ckpt["splats"]

        means_key = "means"
        if "means3d" in gaussian_state_dict:
            means_key = "means3d"

        from internal.models.vanilla_gaussian import VanillaGaussian
        from internal.utils.gaussian_utils import SHS_REST_DIM_TO_DEGREE
        model = VanillaGaussian(sh_degree=SHS_REST_DIM_TO_DEGREE[gaussian_state_dict["shN"].shape[1]]).instantiate()
        model.setup_from_number(gaussian_state_dict[means_key].shape[0])

        model.load_state_dict({
            "_active_sh_degree": torch.tensor(model.config.sh_degree, dtype=torch.int),
            "gaussians.means": gaussian_state_dict[means_key],
            "gaussians.shs_dc": gaussian_state_dict["sh0"],
            "gaussians.shs_rest": gaussian_state_dict["shN"],
            "gaussians.opacities": gaussian_state_dict["opacities"].unsqueeze(-1),
            "gaussians.scales": gaussian_state_dict["scales"],
            "gaussians.rotations": gaussian_state_dict["quats"],
        })
        model = model.to(device=device)

        from internal.renderers.gsplat_renderer import GSPlatRenderer
        renderer = GSPlatRenderer(anti_aliased=anti_aliased)
        renderer.setup("validation")
        renderer = renderer.to(device=device)

        if eval_mode is True:
            model.eval()
            renderer.eval()
        if pre_activate is True:
            model.pre_activate_all_properties()

        return model, renderer

    @classmethod
    def load(cls, path, device, anti_aliased: bool = True, eval_mode: bool = True, pre_activate: bool = True):
        ckpt = torch.load(path, map_location="cpu")
        return cls.load_from_ckpt(
            ckpt,
            device=device,
            anti_aliased=anti_aliased,
            eval_mode=eval_mode,
            pre_activate=pre_activate,
        )
