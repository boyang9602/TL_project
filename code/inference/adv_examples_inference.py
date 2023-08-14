"""
Inference the adversarial examples
"""
import torch
import pickle
import argparse
from tools.utils import load_topk_idxs
from tools.dataset import get_dataset
from models.pipeline import load_pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pl = load_pipeline(device=device)

parser = argparse.ArgumentParser(description='Inference the generated adversarial examples which are a list of tensors stored in binary file using pickle.')
parser.add_argument('--filename', '-f', action='store', required=True, help='the binary file name of the adversarial examples')
parser.add_argument('--output', '-o', action='store', required=True, help='the binary file name of the detection results to be written to')
parser.add_argument('--dataset', '-ds', action='store', required=True, choices=['S2TLD720', 'S2TLD1080'], help='the dataset where the adversarial examples are from')
parser.add_argument('--topk_file', '-t', action='store', required=False, default=None, help='the selected perfect cases that were used for attack.')
args = parser.parse_args()

topk_filename = args.topk_file
if topk_filename is None:
    topk_filename = f'data/evaluation/{args.dataset}_top200.bin'
indices = load_topk_idxs(topk_filename)
with open(args.filename, 'rb') as f:
    adversarial_examples = pickle.load(f)

ds = get_dataset(args.dataset, True)
inference_results = []

for i, (ds_idx, adv_image) in enumerate(zip(indices, adversarial_examples)):
    item = ds[ds_idx]
    # detect the adv_image
    with torch.no_grad():
        valid, rec, assignments, invalid = pl(adv_image.type(torch.long).to(device), item['boxes'])
        inference_results.append((valid, rec, assignments, invalid))

with open(args.output, 'wb') as f:
    pickle.dump(inference_results, f)
