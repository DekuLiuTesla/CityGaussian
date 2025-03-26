import argparse
import torch
import sys
import yaml
import os

sys.path.append(os.getcwd())

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("path", help="Path to the .ckpt file")
	args = parser.parse_args()
	ckpt = torch.load(args.path, map_location="cpu")

	def tuple_representer(dumper, data):
		return dumper.represent_list(data)
	yaml.add_representer(tuple, tuple_representer)

	print(yaml.dump(ckpt['hyper_parameters'], default_flow_style=False))