import add_pypath
import os
import argparse
from internal.utils.colmap import read_images_binary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("colmap", type=str,
                        help="path to colmap sparse model, e.g.: data/MipNeRF360/garden/sparse/0")
    parser.add_argument("--patterns", "-p", type=str, nargs="*", default=[])
    parser.add_argument("--invert-patterns", "-i", type=str, nargs="*", default=[])
    parser.add_argument("--ranges", type=int, nargs="*", default=[])
    parser.add_argument("--invert-ranges", type=int, nargs="*", default=[])
    parser.add_argument("--step", "-s", type=int, default=1)
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="e.g.: data/MipNeRF360/garden/sparse/0/image_list.txt")
    return parser.parse_args()


def is_match_any_patterns(image_name: str, patterns: list) -> bool:
    for pattern in patterns:
        if image_name.find(pattern) != -1:
            return True
    return False


def is_in_any_ranges(value: int, ranges: list) -> bool:
    for i in range(0, len(ranges), 2):
        start = ranges[i]
        end = ranges[i + 1]

        if value <= end and value >= start:
            return True

    return False


def main():
    args = parse_args()

    print("Loading colmap sparse model...")
    images = read_images_binary(os.path.join(args.colmap, "images.bin"))
    images = dict(sorted(images.items(), key=lambda item: item[0]))

    counter = 0
    with open(args.output, "w") as f:
        for idx, i in enumerate(images):
            image = images[i]
            # step
            if idx % args.step != 0:
                continue

            # patterns
            if len(args.patterns) > 0 and is_match_any_patterns(image.name, args.patterns) is False:
                continue
            if is_match_any_patterns(image.name, args.invert_patterns) is True:
                continue

            # ranges
            if len(args.ranges) > 0 and is_in_any_ranges(image.id, args.ranges) is False:
                continue
            if is_in_any_ranges(image.id, args.invert_ranges) is True:
                continue

            f.write(image.name)
            f.write("\n")
            counter += 1

    print("{} images added to list '{}'".format(counter, args.output))


main()
