class LoggerMixin:
    def setup(self, stage: str, pl_module: "LightningModule") -> None:
        super().setup(stage, pl_module)

        self.avoid_state_dict = {"pl": pl_module}

    def log_metric(self, name, value):
        self.avoid_state_dict["pl"].logger.log_metrics(
            {
                "density/{}".format(name): value,
            },
            step=self.avoid_state_dict["pl"].trainer.global_step,
        )
