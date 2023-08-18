"""
Select the top 200 examples from the cases with perfect precison and recall, based on the avg IoU in the picture. 
"""
import argparse
import pickle
import torch
import shutil
from tools.dataset import get_dataset

parser = argparse.ArgumentParser(description='select topk cases.')
parser.add_argument('--filename', '-f', action='store', required=False, default=None, help='the evaluation data file name.')
parser.add_argument('--output', '-o', action='store', required=False, default=None, help='the output file name.')
parser.add_argument('--topk', '-k', action='store', required=False, default=200, type=int, help='top k cases to be selected')
parser.add_argument('--move_files', '-m', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--dataset', '-ds', action='store', required=False, help='needed when --move_files')
parser.add_argument('--skip', '-s', action='append', type=int, required=False, help='the cases ids that you want to skip')
args = parser.parse_args()

filename = args.filename
if filename is None:
    filename = f'data/evaluation/{args.dataset}_pre_recall.bin'
if filename.endswith('txt'):
    results = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            results.append([float(x) for x in line.strip().split(',')])
else:
    with open(filename, 'rb') as f:
        results = pickle.load(f)
results = torch.tensor(results)
# filter perfect cases
perf_prec = results[:, 0] >= 0.999
perf_reca = results[:, 1] >= 0.999
perf_idxs = (perf_prec * perf_reca).nonzero(as_tuple=True)[0]

perf_cases = results[perf_idxs]
topk = args.topk
if args.skip is not None:
    topk += len(args.skip)
topk = torch.topk(perf_cases[:, -1], topk)
topk_idxs = perf_idxs[topk.indices]

id_value_pairs = []
for idx, value in zip(topk_idxs, topk.values):
    if idx in args.skip:
        continue
    id_value_pairs.append((idx.item(), value.item()))
    if len(id_value_pairs) == args.topk:
        break

output = args.output
if output is None:
    output = f'data/evaluation/{args.dataset}_top{args.topk}.txt'
if output.endswith('txt'):
    with open(output, 'w') as f:
        for idx, value in id_value_pairs:
            f.write(f'{idx},{value}\n')
else:
    with open(output, 'wb') as f:
        pickle.dump(id_value_pairs, f)

if args.move_files:
    ds = get_dataset(args.dataset, False)
    for idx, value in id_value_pairs:
        item = ds[idx]
        shutil.copy(item['image_file'], f'{args.dataset}/top200/images/')
        shutil.copy(item['annot_file'], f'{args.dataset}/top200/annotations/')
