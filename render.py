import glob
import os
import queue
import subprocess
import argparse
import json
import threading
import traceback

import numpy as np
import lightning
import torch
import torchvision
import mediapy
from tqdm import tqdm
from internal.cameras.cameras import Cameras
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gaussian_model_editor import MultipleGaussianModelEditor
from internal.viewer.renderer import ViewerRenderer


def initializer_viewer_renderer(
        model_paths: list[str],
        enable_transform: bool,
        sh_degree: int,
        background_color,
        renderer_override,
        device,
) -> ViewerRenderer:
    if len(args.model_paths) == 1 and args.model_paths[0].endswith(".yaml"):
        import yaml
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

        model_manager = model
    else:
        model_list = []
        renderer = None

        load_device = torch.device("cuda") if len(model_paths) == 1 or enable_transform is False else torch.device("cpu")
        for model_path in model_paths:
            model, renderer = GaussianModelLoader.search_and_load(model_path, load_device)
            model.freeze()
            model_list.append(model)

        if len(model_paths) > 1:
            renderer = VanillaRenderer()
        if renderer_override is not None:
            print(f"Renderer: {renderer_override.__class__}")
            renderer = renderer_override

        model_manager = MultipleGaussianModelEditor(model_list, device)

    return ViewerRenderer(model_manager, renderer, torch.tensor(background_color, dtype=torch.float, device=device))


def parse_camera_poses(camera_path: dict):
    orientation_transform = torch.tensor(camera_path["orientation_transform"], dtype=torch.float)

    c2w_flatten_list = []
    fov_list = []
    aspect_list = []

    for camera in camera_path["camera_path"]:
        c2w_flatten_list.append(camera["camera_to_world"])
        fov_list.append(camera["fov"])
        aspect_list.append(camera["aspect"])

    width = torch.tensor([int(camera_path["render_width"])], dtype=torch.int16).expand(len(c2w_flatten_list))
    height = torch.tensor([int(camera_path["render_height"])], dtype=torch.int16).expand(len(c2w_flatten_list))

    c2w = torch.reshape(torch.tensor(c2w_flatten_list, dtype=torch.float), (-1, 4, 4))
    c2w = torch.matmul(orientation_transform, c2w)
    c2w[..., :3, 1:3] *= -1
    w2c = torch.linalg.inv(c2w)
    fov = torch.deg2rad(torch.tensor(fov_list, dtype=torch.float))
    fx = width / (2 * torch.tan(fov / 2))
    fy = fx

    return Cameras(
        R=w2c[..., :3, :3],
        T=w2c[..., :3, 3],
        fx=fx,
        fy=fy,
        cx=width / 2.,
        cy=height / 2.,
        width=width,
        height=height,
        appearance_id=torch.zeros_like(fx, dtype=torch.long),
        normalized_appearance_id=torch.zeros_like(fx),
        distortion_params=None,
        camera_type=torch.zeros_like(fx),
    )


def parse_model_transformations(camera_path: dict) -> list[list]:
    frame_transformation_list = []

    for frame in camera_path["camera_path"]:
        model_transformation_list = []
        if "model_poses" in frame and frame["model_poses"] is not None:
            for model_idx in range(len(frame["model_poses"])):
                model_transformation_list.append({
                    "size": frame["model_sizes"][model_idx],
                    "wxyz": frame["model_poses"][model_idx]["wxyz"],
                    "position": frame["model_poses"][model_idx]["position"],
                })
        frame_transformation_list.append(model_transformation_list)

    return frame_transformation_list


def save_image(image, output_path):
    torchvision.utils.save_image(image, output_path)


def process_save_image_queue(image_queue: queue.Queue, frame_output_path: str):
    while True:
        image_information = image_queue.get()
        if image_information is None:
            break
        try:
            save_image(image_information[0], os.path.join(frame_output_path, "{:06d}.png".format(image_information[1])))
        except:
            traceback.print_exc()


def process_image_to_video_queue(image_queue: queue.Queue, video_writer: mediapy.VideoWriter):
    while True:
        image_information = image_queue.get()
        if image_information is None:
            break
        video_writer.add_image(image_information[0].permute(1, 2, 0).numpy())


def process_image_queue(image_save_batch: int, image_queue: queue.Queue, video_writer: mediapy.VideoWriter, frame_output_path: str):
    class MockQueue:
        def put(self, *args, **kwargs):
            return

    # create image to file threads
    save_image_threads = []
    frame_saving_queue = MockQueue()
    if frame_output_path is not None:
        frame_saving_queue = queue.Queue(maxsize=max(image_save_batch // 2, 1))
        for _ in range(image_save_batch):
            thread = threading.Thread(target=process_save_image_queue, args=(frame_saving_queue, frame_output_path))
            save_image_threads.append(thread)
            thread.start()

    # create image to video thread
    image_to_video_queue = queue.Queue(maxsize=1)
    image_to_video_thread = threading.Thread(target=process_image_to_video_queue, args=(image_to_video_queue, video_writer))
    image_to_video_thread.start()

    # forward rendered image to threads
    while True:
        image_information = image_queue.get()
        if image_information is None:
            break
        frame_saving_queue.put(image_information)
        image_to_video_queue.put(image_information)

    # wait for all threads finishing
    for _ in range(len(save_image_threads)):
        frame_saving_queue.put(None)
    for i in save_image_threads:
        i.join()
    image_to_video_queue.put(None)
    image_to_video_thread.join()


def render_frames(
        cameras: Cameras,
        model_transformations: list,
        viewer_renderer: ViewerRenderer,
        frame_output_path: str,
        video_writer: mediapy.VideoWriter,
        image_save_batch: int,
        device,
):
    image_queue = queue.Queue(maxsize=1)
    image_process_thread = threading.Thread(target=process_image_queue, args=(
        image_save_batch,
        image_queue,
        video_writer,
        frame_output_path,
    ))
    image_process_thread.start()

    for idx in tqdm(range(len(cameras)), desc="rendering frames"):
        # model transform
        for model_idx, model_transformation in enumerate(model_transformations[idx]):
            viewer_renderer.gaussian_model.transform_with_vectors(
                model_idx,
                scale=model_transformation["size"],
                r_wxyz=np.asarray(model_transformation["wxyz"]),
                t_xyz=np.asarray(model_transformation["position"]),
            )

        # render
        camera = cameras[idx].to_device(device)
        image = viewer_renderer.get_outputs(camera).cpu()
        image_queue.put((image, idx))

    image_queue.put(None)
    image_process_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_paths", type=str, nargs="+")
    parser.add_argument("--camera-path-filename", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--save-images", "--save-image", "--save_image", "--save-frames", action="store_true",
                        help="Whether save each frame to an image file")
    parser.add_argument("--image-save-batch", "-b", type=int, default=8,
                        help="increase this to speedup rendering, but more memory will be consumed")
    parser.add_argument("--disable-transform", action="store_true", default=False)
    parser.add_argument("--vanilla_gs2d", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cuda")

    with open(args.camera_path_filename, "r") as f:
        camera_path = json.load(f)

    # whether a 2DGS model
    renderer_override = None
    if args.vanilla_gs2d is True:
        from internal.renderers.vanilla_2dgs_renderer import Vanilla2DGSRenderer

        renderer_override = Vanilla2DGSRenderer()

    # instantiate renderer
    # TODO: set output type
    renderer = initializer_viewer_renderer(
        args.model_paths,
        enable_transform=camera_path["enable_transform"],
        sh_degree=camera_path["sh_degree"],
        background_color=camera_path["background_color"],
        renderer_override=renderer_override,
        device=device,
    )

    # load cameras
    cameras = parse_camera_poses(camera_path)
    if args.disable_transform is False:
        model_transformations = parse_model_transformations(camera_path)
    else:
        model_transformations = [[] for _ in range(len(cameras))]

    # create output path
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    frame_output_path = None
    if args.save_images is True:
        frame_output_path = args.output_path + "_frames"
        os.makedirs(frame_output_path, exist_ok=True)
        for i in glob.glob(os.path.join(frame_output_path, "*.png")):
            os.unlink(i)

    # start rendering
    with torch.no_grad(), mediapy.VideoWriter(
            path=args.output_path,
            shape=(cameras[0].height.item(), cameras[0].width.item()),
            fps=camera_path["fps"],
    ) as video_writer:
        render_frames(
            cameras,
            model_transformations,
            viewer_renderer=renderer,
            frame_output_path=frame_output_path,
            video_writer=video_writer,
            image_save_batch=args.image_save_batch,
            device=device,
        )

    if frame_output_path is not None:
        print(f"Video frames saved to '{frame_output_path}'")
    print(f"Video saved to '{args.output_path}'")
