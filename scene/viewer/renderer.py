from arguments import GroupParams

class ViewerRenderer:
    def __init__(
            self,
            gaussian_model,
            renderer,
            background_color,
            compute_cov3D_python: bool = False, 
            convert_SHs_python: bool = False
    ):
        super().__init__()

        self.gaussian_model = gaussian_model
        self.renderer = renderer
        self.background_color = background_color
        self.pipeline = GroupParams
        self.pipeline.compute_cov3D_python = compute_cov3D_python
        self.pipeline.convert_SHs_python = convert_SHs_python
        self.pipeline.debug = False

    def get_outputs(self, camera, scaling_modifier: float = 1.):
        return self.renderer(
            camera,
            self.gaussian_model,
            self.pipeline,
            self.background_color,
            scaling_modifier=scaling_modifier,
        )["render"]
