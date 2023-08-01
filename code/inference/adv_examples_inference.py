"""
Inference the adversarial examples
"""
import torch
import pickle
from models.src.pipeline import load_pipeline
from tools.dataset import get_dataset
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
pl = load_pipeline(device=device)

parser = argparse.ArgumentParser(description='Inference the generated adversarial examples which are a list of tensors stored in binary file using pickle.')
parser.add_argument('filename', metavar='f', action='store', required=True, help='the binary file name of the adversarial examples')
parser.add_argument('output', metavar='o', action='store', required=True, help='the binary file name of the detection results to be written to')
parser.add_argument('dataset', metavar='ds', action='store', required=True, choices=['S2TLD720', 'S2TLD1080'], help='the dataset where the adversarial examples are from')
parser.add_argument('indices', metavar='id', action='store', required=True, help='the txt file of the indices of the adverarial examples in the corresponding dataset')

args = parser.parse_args()

with open(args.filename, 'rb') as f, open(args.indices, 'rb') as f2:
    adversarial_examples = pickle.load(f)
    indices = pickle.load(f2)

ds = get_dataset(args.dataset)
inference_results = []

for i, ds_idx, adv_image in enumerate(zip(indices, adversarial_examples)):
    item = ds[ds_idx]
    # detect the adv_image
    with torch.no_grad():
        valid, rec, assignments, invalid, rpn_attack_data = pl(adv_image.type(torch.long), item['boxes'])
        inference_results.append((valid, rec, assignments, invalid, rpn_attack_data))

with open(args.output, 'wb') as f:
    pickle.dump(inference_results, f)
