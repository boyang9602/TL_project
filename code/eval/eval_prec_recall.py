"""
Evaluate the precision and recall for each picture, iou_threshold == 0.5
TP means iou >= iou_threshold and correct recognition color
FP otherwise
FN means ground truth not matched with any TP detection
Write the results to a txt file, each line is (precision, recall, avg_iou)
"""
import pickle
import torch
import tools.utils as utils
import argparse

parser = argparse.ArgumentParser(description='Find the perfect cases in terms of precision and recall. It will output 3 files, perfect_precision.bin, perfect_recall.bin, perfect_precison_recall.bin.')
parser.add_argument('--detection', '-d', action='store', required=True, help='the binary file name of the detection results.')
parser.add_argument('--output', '-o', action='store', required=True, help='the output file name.')
args = parser.parse_args()

color_labels = ["off", "red", "yellow", "green"]
with open(args.detection, 'rb') as f:
    detections = pickle.load(f)

threshold = 0.5
results = []

for case_id, (valid_detections, recognitions, assignments, invalid_detections, boxes, colors) in enumerate(detections):
    len_invalid = 0 if invalid_detections == None else len(invalid_detections)
    if len(valid_detections) == 0:
        results.append((0, 0, 0))
        continue
    ious = utils.IoU_multi(valid_detections[:, 1:5].cpu(), torch.tensor(boxes))
    maxes = torch.max(ious, 1)
    matched_gts = set()
    TP = 0
    FP = len_invalid
    for dt_idx, (gt_idx, iou) in enumerate(zip(maxes.indices, maxes.values)):
        dt_color = color_labels[torch.argmax(recognitions[dt_idx])]
        gt_color = colors[gt_idx]
        if iou >= threshold and dt_color == gt_color:
            TP += 1
            matched_gts.add(gt_idx)
        else:
            FP += 1
    FN = len(colors) - len(matched_gts)
    assert TP + FP == len(valid_detections) + len_invalid
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    assert TP + FN == len(colors)
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    avg_iou = sum(maxes.values) / len(maxes.values) if len(maxes.values) > 0 else 0
    results.append((precision, recall, avg_iou.item() if type(avg_iou) != int else 0))

if args.output.endswith('txt'):
    with open(args.output, 'w') as f:
        for line in results:
            f.write(','.join([str(x) for x in line]) + '\n')
else:
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)