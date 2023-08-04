#!/usr/bin/env python
import argparse
import pickle
import torch
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', '-ds', action='store', required=True, help='the dataset')
parser.add_argument('--filename', '-f', action='store', required=False, default=None, help='the detection results file.')
parser.add_argument('--output', '-o', action='store', required=False, default=None, help='the output.')
args = parser.parse_args()
TL_TYPES = ['UNK', 'VERT', 'QUAD', 'HORI']

if args.filename is None:
    filename = f'data/inferences/{args.dataset}_detection_results.bin'
else:
    filename = args.filename

with open(filename, 'rb') as f:
    detection_results = pickle.load(f)

types = []
for valid, _, assignments, _, boxes, _ in detection_results:
    curr_types = []
    tl_types = torch.argmax(valid[:, 5:], dim=1)
    for i in range(len(boxes)):
        assignment = assignments[assignments[:, 0] == i]
        assert len(assignment) <= 1
        if len(assignment) == 0:
            curr_types.append('NA')
        else:
            curr_type = int(tl_types[assignment[0][1]].item())
            curr_types.append(TL_TYPES[curr_type])
    types.append(curr_types)

if args.output is None:
    output = f'{args.dataset}/inferred_tl_types.txt'
else:
    output = args.output

with open(output, 'w') as f:
    for curr_types in types:
        f.write(','.join(curr_types) + '\n')
