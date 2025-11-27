import argparse
import os
import torch
from copy import deepcopy

def merge(theta_0, theta_1):
    w = {
        key: theta_0[key] + theta_1[key]
        for key in theta_0.keys()
    }
    return deepcopy(w)

def load_ckpt(load_path):
    if not os.path.exists(load_path):
        raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

    checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)

    return checkpoint['model_dict'], checkpoint['start_step']


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", nargs='+', type=str, default=[], help="checkpoint paths to merge")
parser.add_argument("--output", type=str, required=True, help="output path for the merged checkpoint")
parser.add_argument("--outdir", type=str, default="./output", help="output directory")
args = parser.parse_args()

cpkt_paths = args.ckpt

if not isinstance(cpkt_paths, list):
    cpkt_paths = [cpkt_paths]

soups, start_step = load_ckpt(cpkt_paths[0])
print(f"Successfully loaded checkpoint from {cpkt_paths[0]}")

for cpkt_path in cpkt_paths[1:]:
    try:
        checkpoint, _ = load_ckpt(cpkt_path)
        soups = merge(soups, checkpoint)
        print(f"Successfully loaded checkpoint from {cpkt_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {cpkt_path}: {e}")

soups = {k: (v.float()/len(cpkt_paths)).half() for k, v in soups.items()}

checkpoint = {
    "model_dict": soups,    
    "start_step": start_step,
}

args.output = os.path.join(args.outdir, args.output)
os.makedirs(args.output, exist_ok=True)
output_path = os.path.join(args.output, "model.pkl")
torch.save(checkpoint, output_path)
print(f"Merged checkpoint saved to {output_path}")
