import wandb

def tensorboard_log_image(log_writer, tag: str, image_tensor, step):
    log_writer.experiment.add_image(
        tag,
        image_tensor,
        step,
    )

def wandb_log_image(log_writer, tag: str, image_tensor, step):
    image_dict = {
        tag: wandb.Image(image_tensor),
    }
    log_writer.experiment.log(
        image_dict,
        step=step,
    )