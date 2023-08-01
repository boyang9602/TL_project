"""
Select the top 200 examples from the cases with perfect precison and recall, based on the avg IoU in the picture. 
"""
import argparse
import pickle
import torch

parser = argparse.ArgumentParser(description='Find the perfect cases in terms of precision and recall. It will output 3 files, perfect_precision.bin, perfect_recall.bin, perfect_precison_recall.bin.')
parser.add_argument('--filename', '-f', action='store', required=True, help='the evaluation data file name.')
parser.add_argument('--output', '-o', action='store', required=True, help='the output file name.')
parser.add_argument('--topk', '-k', action='store', required=False, default=200, type=int, help='top k cases to be selected')
args = parser.parse_args()

if args.filename.endswith('txt'):
    results = []
    with open(args.filename, 'r') as f:
        for line in f.readlines():
            results.append([float(x) for x in line.strip().split(',')])
else:
    with open(args.filename, 'rb') as f:
        results = pickle.load(f)
results = torch.tensor(results)
# filter perfect cases
perf_prec = results[:, 0] >= 0.999
perf_reca = results[:, 1] >= 0.999
perf_idxs = (perf_prec * perf_reca).nonzero(as_tuple=True)[0]

perf_cases = results[perf_idxs]
topk = torch.topk(perf_cases[:, -1], args.topk)
topk_idxs = perf_idxs[topk.indices]

id_value_pairs = []
for idx, value in zip(topk_idxs, topk.values):
    id_value_pairs.append((idx.item(), value.item()))

if args.output.endswith('txt'):
    with open(args.output, 'w') as f:
        for idx, value in id_value_pairs:
            f.write(f'{idx},{value}\n')
else:
    with open(args.output, 'wb') as f:
        pickle.dump(id_value_pairs, f)
