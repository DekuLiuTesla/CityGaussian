import add_pypath
import os
import argparse
import torch
from tqdm import tqdm
from lightning.fabric.utilities.apply_func import move_data_to_device
from internal.dataparsers.spotless_colmap_dataparser import SpotLessColmap
from internal.metrics.spotless_metrics import SpotLessMetrics
from internal.utils.gaussian_model_loader import GaussianModelLoader, GSplatV1ExampleCheckpointLoader
from internal.dataset import Dataset, CacheDataLoader
from common import AsyncImageSaver

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("--vanilla_ckpt", "-v", action="store_true", default=False)
args = parser.parse_args()

device = torch.device("cuda")

if args.vanilla_ckpt is True:
    """
    must run with `--no_normalize`
    
                 torch.save(
                     {
+                        "cfg": dataclasses.asdict(self.cfg),
                         "step": step,
                         "splats": self.splats.state_dict(),
+                        "running_stats": self.running_stats,
+                        "spotless_module": self.spotless_module.state_dict() if self.mlp_spotless else {},
                     },
                     f"{self.ckpt_dir}/ckpt_{step}.pt",
                 )
    """

    ckpt = torch.load(args.model_path, map_location="cpu")
    assert ckpt["cfg"]["normalize"] is False

    model, renderer = GSplatV1ExampleCheckpointLoader.load_from_ckpt(ckpt, device=device)
    renderer.anti_aliased = ckpt["cfg"]["antialiased"]

    model.eval()
    renderer.eval()

    metric = SpotLessMetrics(
        lambda_dssim=ckpt["cfg"]["ssim_lambda"],
        lower_bound=ckpt["cfg"]["lower_bound"],
        upper_bound=ckpt["cfg"]["upper_bound"],
        bin_size=ckpt["cfg"]["bin_size"],
        cluster=ckpt["cfg"]["cluster"],
        schedule=ckpt["cfg"]["schedule"],
        schedule_beta=ckpt["cfg"]["schedule_beta"],
        robust_percentile=ckpt["cfg"]["robust_percentile"],
        max_mlp_mask_size=99999,
    ).instantiate()
    metric.setup("validation", None)
    metric.to(device)

    metric_state_dict = {"spotless_module.{}".format(k): v.to(device) for k, v in ckpt["spotless_module"].items()}
    metric_state_dict["_hist_err"] = ckpt["running_stats"]["hist_err"].to(device)
    metric_state_dict["_avg_err"] = torch.tensor(ckpt["running_stats"]["avg_err"], dtype=torch.float, device=device)
    metric_state_dict["_lower_err"] = torch.tensor(ckpt["running_stats"]["lower_err"], dtype=torch.float, device=device)
    metric_state_dict["_upper_err"] = torch.tensor(ckpt["running_stats"]["upper_err"], dtype=torch.float, device=device)
    metric.load_state_dict(metric_state_dict)
    metric.eval()

    # setup dataset
    dataparser_outputs = SpotLessColmap(
        train_keyword=ckpt["cfg"]["train_keyword"],
        test_keyword=ckpt["cfg"]["test_keyword"],
    ).instantiate(ckpt["cfg"]["data_dir"], output_path=os.getcwd(), global_rank=0).get_outputs()
    dataset = Dataset(dataparser_outputs.train_set, undistort_image=False)
    dataloader = CacheDataLoader(dataset, max_cache_num=-1, shuffle=False)

    image_output_path = os.path.join(os.path.dirname(os.path.dirname(args.model_path)), "{}-sls_predicts".format(os.path.basename(args.model_path)))
else:
    load_file = GaussianModelLoader.search_load_file(args.model_path)
    ckpt = torch.load(load_file, map_location="cpu")

    # setup model and renderer
    model = GaussianModelLoader.initialize_model_from_checkpoint(
        checkpoint=ckpt,
        device=device,
    )
    renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(
        checkpoint=ckpt,
        stage="validation",
        device=device,
    )
    model.eval()
    renderer.eval()

    # setup metric
    metric = ckpt["hyper_parameters"]["metric"].instantiate()
    metric.setup("validation", None)
    metric_state_dict = {}
    for i in ckpt["state_dict"]:
        if i.startswith("metric."):
            metric_state_dict[i[len("metric."):]] = ckpt["state_dict"][i].to(device=device)
    metric.to(device)
    metric.load_state_dict(metric_state_dict)
    metric.eval()

    # setup dataset
    dataparser_outputs = ckpt["datamodule_hyper_parameters"]["parser"].instantiate(path=ckpt["datamodule_hyper_parameters"]["path"], output_path=os.getcwd(), global_rank=0).get_outputs()
    dataset = Dataset(dataparser_outputs.train_set, undistort_image=False)
    dataloader = CacheDataLoader(dataset, max_cache_num=-1, shuffle=False)

    image_output_path = os.path.join(os.path.dirname(os.path.dirname(load_file)), "{}-sls_predicts".format(os.path.basename(load_file)))

os.makedirs(image_output_path, exist_ok=True)

bg_color = torch.zeros((3,), dtype=torch.float, device=device)

image_saver = AsyncImageSaver(is_rgb=True)
try:
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = move_data_to_device(batch, device)

            outputs = renderer(batch[0], model, bg_color)
            metrics, _ = metric.get_train_metrics(None, model, 30_000, batch, outputs)

            mask = outputs["raw_pred_mask"].squeeze(-1).repeat(3, 1, 1)
            bool_mask = mask > 0.5

            fused_rgb_mask = batch[1][1].clone()
            fused_rgb_mask[1:3, ~bool_mask[0]] *= 0.25
            fused_rgb_mask[1:3, ~bool_mask[0]] += 0.75

            output_image = (torch.concat([batch[1][1], mask, fused_rgb_mask, outputs["render"].clamp_max(1.)], dim=2) * 255).permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()

            image_saver.save(output_image, os.path.join(image_output_path, batch[1][0] + ".png"))
finally:
    image_saver.stop()

print(f"Saved to `{image_output_path}`")
