import pickle
import csv
import torch
import utils
import xml.etree.ElementTree as ET

REC_COLORS = ["off", "red", "yellow", "green"]
with open('detections.bin', 'rb') as f:
    detections = pickle.load(f)

nonexistence = [1697, 1908, 2950]
cursor = 0
perfect_precision = []
perfect_recall = []
perfect = []
threshold = 0.5

for i, (detection, assignments) in enumerate(detections):
    print(i)
    if i + cursor in nonexistence:
        cursor += 1
    case = i + cursor
    annot_file = 'annotations/{:06d}.xml'.format(case)
    tree = ET.parse(annot_file)
    objs = tree.findall("object")
    boxes = utils.readxml(annot_file)
    print('detection:')
    print(detection[:, 1:5])
    print('ground truth')
    print(boxes)
    if len(detection) == 0 or len(boxes) == 0:
        print('no detection, pass')
        continue
    ious = utils.IoU_multi(detection[:, 1:5].cpu(), torch.tensor(boxes))
    print('dt -> gt, ious: ')
    print(ious)
    maxes = torch.max(ious, 1)
    print(maxes)
    values = maxes.values
    indices = maxes.indices
    matched_ious = []
    flag = True
    color_flag = True
    for j, idx in enumerate(indices):
        dt = detection[j]
        gt = boxes[idx]
        gt_color = REC_COLORS.index(objs[idx].find("name").text)
        dt_color = torch.argmax(dt[-4:])
        if gt_color != dt_color:
            color_flag = False
        iou = values[j]
        matched_ious.append(iou)
    if len(matched_ious) > 0 and min(matched_ious) >= threshold and color_flag: # every detection has an IoU >= threshold with at least one ground truth, so 100% precision
        perfect_precision.append((i, i+cursor))
    else:
        flag = False

    ious = utils.IoU_multi(torch.tensor(boxes), detection[:, 1:5].cpu())
    print('gt -> dt, ious: ')
    print(ious)
    maxes = torch.max(ious, 1)
    print(maxes)
    values = maxes.values
    indices = maxes.indices
    matched_ious = []
    color_flag = True
    for j, idx in enumerate(indices):
        gt = boxes[j]
        dt = detection[idx]
        gt_color = REC_COLORS.index(objs[j].find("name").text)
        dt_color = torch.argmax(dt[-4:])
        if gt_color != dt_color:
            color_flag = False
        iou = values[j]
        matched_ious.append(iou)
    if len(matched_ious) > 0 and min(matched_ious) >= threshold and color_flag: # every ground_truth has an IoU >= threshold with at least one detection, so 100% recall
        perfect_recall.append((i, i+cursor))
    else:
        flag = False

    if flag and len(detection) == len(boxes):
        perfect.append((i, i+cursor))

print(len(perfect_precision), len(perfect_recall), len(perfect))
with open('perfect_precision2.bin', 'wb') as f:
    pickle.dump(perfect_precision, f)

with open('perfect_recall2.bin', 'wb') as f:
    pickle.dump(perfect_recall, f)

with open('perfect2.bin', 'wb') as f:
    pickle.dump(perfect, f)