import pickle
import torch
import argparse
from tools.utils import convert_labels_to_nums, load_topk_idxs, IoU_single
from tools.dataset import get_dataset
from torch import IntTensor, Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.iou import IntersectionOverUnion

parser = argparse.ArgumentParser(description='Evaluate the performance of a adversarial examples.')
parser.add_argument('--filename', '-f', action='store', required=True, help='the binary file name of the detection results.')
parser.add_argument('--dataset', '-ds', action='store', required=True, choices=['S2TLD720', 'S2TLD1080'], help='the dataset name of the detections.')
parser.add_argument('--topk_file', '-k', action='store', required=False, default=None, help='the selected perfect cases.')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(args.filename, 'rb') as f:
    detections = pickle.load(f)
topk_filename = args.topk_file
if topk_filename is None:
    topk_filename = f'data/evaluation/{args.dataset}_top200.bin'

ds = get_dataset(args.dataset, True)
inference_results = []

TL_TYPES = ['UNK', 'VERT', 'QUAD', 'HORI']
COLOR_LABELS = ["off", "red", "yellow", "green"]

targets = []
preds = []
eval_data = []
total_lights = 0
for i, (ds_idx, detection) in enumerate(zip(load_topk_idxs(topk_filename), detections)):
    item = ds[ds_idx]
    boxes = item['boxes']
    colors = item['colors']
    labels = item['inferred_tl_types']
    valid_detections, recognitions, assignments, invalid_detections = detection
    
    target = {
        'boxes': Tensor(boxes).to(device),
        'labels': IntTensor(convert_labels_to_nums(labels, TL_TYPES)).to(device)
    }
    targets.append(target)

    all_detections = torch.vstack([valid_detections, invalid_detections])
    values, indices = torch.max(all_detections[:, 5:], 1)
    pred = {
        'boxes': all_detections[:, 1:5].to(device),
        'scores': values.to(device),
        'labels': indices.type(torch.long).to(device)
    }
    preds.append(pred)

    total_lights += len(boxes)
    unmatched_gt_idxs = {i for i in range(len(boxes))}
    curr = []
    for tup in assignments:
        gt_idx = tup[0].item()
        det_idx = tup[1].item()

        assert gt_idx in unmatched_gt_idxs
        unmatched_gt_idxs.remove(gt_idx)

        curr.append({
            'id': gt_idx,
            'box': {
                'gt': boxes[gt_idx],
                'det': valid_detections[det_idx]
            },
            'color': {
                'gt': colors[gt_idx],
                'det': COLOR_LABELS[torch.argmax(recognitions[det_idx])]
            }
        })
    for gt_idx in unmatched_gt_idxs:
        curr.append({
                'id': gt_idx,
                'box': {
                    'gt': boxes[gt_idx],
                    'det': None
                },
                'color': {
                    'gt': colors[gt_idx],
                    'det': None
                }
        })
    eval_data.append(curr)

unmatched = 0
correct_colors = 0
iou0_0 = 0
iou0_50 = 0
iou50_100 = 0
iou0_0_correct = 0
iou0_50_correct = 0
iou50_100_correct = 0

for pic_data in eval_data:
    # for all lights in all pictures
    # pre, recall
    for light in pic_data:
        if light['color']['det'] == None and light['box']['det'] == None:
            unmatched += 1
            continue
        assert light['color']['det'] != None and light['box']['det'] != None

        if light['color']['gt'] == light['color']['det']:
            correct_colors += 1
        iou = IoU_single(light['box']['gt'], light['box']['det'][1:5])

        if iou == 0:
            iou0_0 += 1
            if light['color']['gt'] == light['color']['det']:
                iou0_0_correct += 1
        elif iou < 0.5:
            iou0_50 += 1
            if light['color']['gt'] == light['color']['det']:
                iou0_50_correct += 1
        else:
            iou50_100 += 1
            if light['color']['gt'] == light['color']['det']:
                iou50_100_correct += 1

metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
metric.update(preds, targets)
result1 = metric.compute()
metric = IntersectionOverUnion()
metric.update(preds, targets)
result2 = metric.compute()
attack_type, filename = args.filename.split('/')[-2:]
print(f'{args.dataset},{attack_type},{filename[:-22]},{unmatched},{correct_colors},{iou0_0},{iou0_0_correct},{iou0_50},{iou0_50_correct},{iou50_100},{iou50_100_correct},{total_lights},{result1["map"].item()},{result2["iou"].item()}')
