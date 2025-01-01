from typing import Tuple
import torch
import internal.renderers as renderers
from internal.utils.visualizers import Visualizers


class ViewerRenderer:
    def __init__(
            self,
            gaussian_model,
            renderer: renderers.Renderer,
            background_color,
    ):
        super().__init__()

        self.gaussian_model = gaussian_model
        self.renderer = renderer
        self.background_color = background_color

        # TODO: initial value should get from renderer
        self.output_info: Tuple[str, renderers.RendererOutputInfo, renderers.RendererOutputVisualizer] = (
            "rgb",
            renderers.RendererOutputInfo("render"),
            self.no_processing,
        )

    def set_output_info(
            self,
            name: str,
            renderer_output_info: renderers.RendererOutputInfo,
            visualizer: renderers.RendererOutputVisualizer,
    ):
        self.output_info = (
            name,
            renderer_output_info,
            visualizer,
        )

    def _setup_depth_map_options(self, viewer, server):
        self.max_depth_gui_number = server.gui.add_number(
            label="Max Clamp",
            initial_value=0.,
            min=0.,
            step=0.01,
            hint="value=0 means that no max clamping, value will be normalized based on the maximum one",
            visible=False,
        )
        self.depth_map_color_map_dropdown = server.gui.add_dropdown(
            label="Color Map",
            options=["turbo", "viridis", "magma", "inferno", "cividis", "gray"],
            initial_value="turbo",
            visible=False,
        )

        @self.max_depth_gui_number.on_update
        @self.depth_map_color_map_dropdown.on_update
        def _(event):
            with server.atomic():
                viewer.rerender_for_all_client()

    def _set_depth_map_option_visibility(self, visible: bool):
        self.max_depth_gui_number.visible = visible
        self.depth_map_color_map_dropdown.visible = visible

    def _set_output_type(self, name: str, renderer_output_info: renderers.RendererOutputInfo):
        """
        Update properties
        """
        # toggle depth map option, only enable when type is `gray` and `visualizer` is None
        self._set_depth_map_option_visibility(renderer_output_info.type == renderers.RendererOutputTypes.GRAY and renderer_output_info.visualizer is None)

        # set visualizer
        visualizer = renderer_output_info.visualizer
        if visualizer is None:
            if renderer_output_info.type == renderers.RendererOutputTypes.RGB:
                visualizer = self.no_processing
            elif renderer_output_info.type == renderers.RendererOutputTypes.GRAY:
                visualizer = self.depth_map_processor
            elif renderer_output_info.type == renderers.RendererOutputTypes.NORMAL_MAP:
                visualizer = self.normal_map_processor
            elif renderer_output_info.type == renderers.RendererOutputTypes.FEATURE_MAP:
                visualizer = self.feature_map_processor
            else:
                raise ValueError(f"Unsupported output type `{renderer_output_info.type}`")

        # update
        self.set_output_info(name, renderer_output_info, visualizer)

    def setup_options(self, viewer, server):
        available_outputs = self.renderer.get_available_outputs()
        first_type_name = list(available_outputs.keys())[0]

        with server.gui.add_folder("Output"):
            # setup output type dropdown
            output_type_dropdown = server.gui.add_dropdown(
                label="Type",
                options=list(available_outputs.keys()),
                initial_value=first_type_name,
            )
            self.output_type_dropdown = output_type_dropdown

            @output_type_dropdown.on_update
            def _(event):
                if event.client is None:
                    return
                with server.atomic():
                    # whether valid type
                    output_type_info = available_outputs.get(output_type_dropdown.value, None)
                    if output_type_info is None:
                        return

                    self._set_output_type(output_type_dropdown.value, output_type_info)

                    viewer.rerender_for_all_client()

            self._setup_depth_map_options(viewer, server)

        # update default output type to the first one, must be placed after gui setup
        self._set_output_type(name=first_type_name, renderer_output_info=available_outputs[first_type_name])

    def get_outputs(self, camera, scaling_modifier: float = 1.):
        render_type, output_info, output_processor = self.output_info

        render_outputs = self.renderer(
            camera,
            self.gaussian_model,
            self.background_color,
            scaling_modifier=scaling_modifier,
            render_types=[render_type],
        )
        image = output_processor(render_outputs[output_info.key], render_outputs, output_info)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image

    def depth_map_processor(self, depth_map, *args, **kwargs):
        # TODO: the pixels not covered by any Gaussian (alpha==0), should be 1. after normalization
        max_depth = self.max_depth_gui_number.value
        if max_depth == 0:
            max_depth = depth_map.max()
        # normalize raw depth_map
        depth_map = depth_map - torch.minimum(depth_map.min(), torch.tensor(0., dtype=torch.float, device=depth_map.device))  # avoid negative values
        depth_map = (depth_map / (max_depth + 1e-8)).clamp(max=1.)
        # apply colormap
        return Visualizers.float_colormap(depth_map, self.depth_map_color_map_dropdown.value)

    def normal_map_processor(self, normal_map, *args, **kwargs):
        return Visualizers.normal_map_colormap(normal_map)

    def feature_map_processor(self, feature_map, *args, **kwargs):
        return Visualizers.pca_colormap(feature_map)

    def no_processing(self, i, *args, **kwargs):
        return i
