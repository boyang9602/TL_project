import pickle
import csv
import torch
import utils
import shutil
import xml.etree.ElementTree as ET

with open('perfect2.bin', 'rb') as f:
    perfect_cases = pickle.load(f)
with open('detections.bin', 'rb') as f:
    detections = pickle.load(f)
avg_ious = []
for i, case in perfect_cases:
    detection, assignments = detections[i]
    boxes = utils.readxml('annotations/{:06d}.xml'.format(case))
    ious = utils.IoU_multi(detection[:, 1:5].cpu(), torch.tensor(boxes))

    maxes = torch.max(ious, 1)
    values = maxes.values
    indices = maxes.indices
    matched_ious = []
    for j, idx in enumerate(indices):
        dt = detection[j]
        gt = boxes[idx]
        iou = values[j]
        matched_ious.append(iou)
    avg_ious.append(sum(matched_ious)/len(matched_ious))
top200avg = torch.argsort(torch.tensor(avg_ious), descending=True)[:200]
top200_cases = torch.tensor(perfect_cases, dtype=torch.long)[top200avg]
with open('top200avg.bin', 'wb') as f:
    pickle.dump(top200_cases, f)

for i in range(200):
    case = top200_cases[i][1]
    shutil.copy('annotations/{:06d}.xml'.format(case), 'top200/annotations/{:06d}.xml'.format(case))
    shutil.copy('pictures/{:06d}.jpg'.format(case), 'top200/pictures/{:06d}.jpg'.format(case))
