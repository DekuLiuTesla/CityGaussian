import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from argparse import ArgumentParser

 
if __name__ == '__main__':
    parser = ArgumentParser(description="clean mesh folder under the output path")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Path to the output folder")
    
    args = parser.parse_args(sys.argv[1:])

    # remove outputs/*/checkpoints/*6999-xyz_rgb.ply and outputs/*/mesh/fuse.ply
    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            if file.endswith("xyz_rgb.ply") or file.endswith("fuse.ply") or file.endswith("=6999.ckpt"):
                os.remove(os.path.join(root, file))
                print(f"Removed {os.path.join(root, file)}")