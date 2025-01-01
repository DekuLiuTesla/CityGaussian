import os
import sys
import math
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar, Tqdm


class SaveCheckpoint(Callback):
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.global_rank != 0:
            return
        checkpoint_path = os.path.join(
            pl_module.hparams["output_path"],
            "checkpoints",
            "epoch={}-step={}.ckpt".format(trainer.current_epoch, trainer.global_step),
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        trainer.save_checkpoint(checkpoint_path)


class SaveGaussian(Callback):
    def on_train_end(self, trainer, pl_module) -> None:
        # TODO: should save before densification
        pl_module.save_gaussians()


class KeepRunningIfWebViewerEnabled(Callback):
    def on_train_end(self, trainer, pl_module) -> None:
        if pl_module.web_viewer is None:
            return
        print("Training finished! Web viewer is still running. Press `Ctrl+C` to exist.")
        while True:
            pl_module.web_viewer.is_training_paused = True
            pl_module.web_viewer.process_all_render_requests(pl_module.gaussian_model, pl_module.renderer, pl_module._fixed_background_color())


class StopImageSavingThreads(Callback):
    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        alive_threads = pl_module.image_saving_threads

        while len(alive_threads) > 0:
            # send messages to terminate threads
            while True:
                try:
                    pl_module.image_queue.put(None, block=False)
                except:
                    break

            # check whether any threads are alive
            still_alive_threads = []
            for thread in alive_threads:
                if thread.is_alive() is True:
                    still_alive_threads.append(thread)
            alive_threads = still_alive_threads


class ProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position + 1)
        self.on_epoch_metrics = {}

    def get_metrics(self, trainer, model):
        # only return latest logged metrics
        items = trainer._logger_connector.metrics["pbar"]
        return items

    def on_train_start(self, trainer, pl_module) -> None:
        super().on_train_start(trainer, pl_module)
        self.max_epochs = trainer.max_epochs
        if self.max_epochs < 0:
            self.max_epochs = math.ceil(trainer.max_steps / self.total_train_batches)

        self.epoch_progress_bar = Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position) - 1,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            total=self.max_epochs,
        )
        self.epoch_progress_bar.update(trainer.current_epoch)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self.on_epoch_metrics.update(self.get_metrics(trainer, pl_module))
        self.epoch_progress_bar.set_postfix(self.on_epoch_metrics)
        self.epoch_progress_bar.update()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        self.on_epoch_metrics.update(self.get_metrics(trainer, pl_module))


class ValidateOnTrainEnd(Callback):
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.is_last_batch is False or trainer.current_epoch % trainer.check_val_every_n_epoch != 0:
            trainer.validating = True
            trainer._evaluation_loop.run()
            trainer.validating = False


class StopDataLoaderCacheThread(Callback):
    def _stop_thread(self, data_loader):
        import queue
        if data_loader.cache_thread is not None:
            data_loader.stop_caching = True
            try:
                data_loader.cache_output_queue.get(block=False)
            except queue.Empty:
                pass
            data_loader.cache_thread.join()

    def on_train_end(self, trainer, pl_module) -> None:
        self._stop_thread(trainer.train_dataloader)
