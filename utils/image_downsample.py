import concurrent.futures
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from PIL import Image
from tqdm import tqdm


def find_images(path: str, extensions: list) -> list:
    image_list = []
    for extension in extensions:
        image_list += list(glob(os.path.join(path, "**", "*.{}".format(extension)), recursive=True))

    # convert to relative path
    path_length = len(path)
    image_list = [i[path_length:].lstrip("/\\") for i in image_list]

    return image_list


def resize_image(image, factor):
    width, height = image.size
    resized_width, resized_height = round(width / factor), round(height / factor)
    return image.resize((resized_width, resized_height))


def process_task(src: str, dst: str, image_name: str, factor: float):
    image = Image.open(os.path.join(src, image_name))
    image = resize_image(image, factor)

    output_path = os.path.join(dst, image_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    image.save(output_path, quality=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("--dst", default=None)
    parser.add_argument("--factor", type=float, default=2)
    parser.add_argument("--extensions", nargs="+", default=[
        "jpg",
        "JPG",
        "jpeg",
        "JPEG",
        "png",
        "PNG",
    ])
    args = parser.parse_args()

    assert args.src != args.dst

    if args.dst is None:
        args.dst = "{}_{}".format(args.src, args.factor)

    image_list = find_images(args.src, args.extensions)

    with ThreadPoolExecutor() as tpe:
        future_list = []
        for i in image_list:
            future_list.append(tpe.submit(
                process_task,
                args.src,
                args.dst,
                i,
                args.factor,
            ))

        for _ in tqdm(concurrent.futures.as_completed(future_list), total=len(future_list)):
            pass

    print("images saved to {}".format(args.dst))
