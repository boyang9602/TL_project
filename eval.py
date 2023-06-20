import pickle
import torch
import utils

REC_COLORS = ["off", "red", "yellow", "green"]
with open('detections.bin', 'rb') as f:
    detections = pickle.load(f)

nonexistence = [1697, 1908, 2950]
cursor = 0
perfect_precision = []
perfect_recall = []
perfect = []
threshold = 0.5

for i, (valid_detections, recognitions, assignments, invalid_detections) in enumerate(detections):
    print(i)
    if i + cursor in nonexistence:
        cursor += 1
    case = i + cursor
    folder = 'normal_1' if case <= 778 else 'normal_2'
    annot_file = 'S2TLD/{}/Annotations/{:06d}.xml'.format(folder, case)
    boxes, colors = utils.readxml2(annot_file)
    print('valid detection:')
    print(valid_detections[:, 1:5])
    print('invalid detection:')
    if invalid_detections != None:
        print(invalid_detections[:, 1:5])
    print('ground truth')
    print(boxes)
    print('assignments')
    print(assignments)
    if len(valid_detections) == 0 or len(boxes) == 0:
        print('no valid detection, pass')
        continue
    ious = utils.IoU_multi(valid_detections[:, 1:5].cpu(), torch.tensor(boxes))
    print('dt -> gt, ious: ')
    print(ious)
    maxes = torch.max(ious, 1)
    print(maxes)
    values = maxes.values
    indices = maxes.indices
    matched_ious = []
    flag = True # if all detections are TP
    color_flag = True # if all colors are correct
    for j, idx in enumerate(indices):
        dt = valid_detections[j]
        color_scores = recognitions[j]
        gt = boxes[idx]
        gt_color = REC_COLORS.index(colors[idx])
        dt_color = torch.argmax(color_scores)
        if gt_color != dt_color:
            color_flag = False
        iou = values[j]
        matched_ious.append(iou)
    # precision = TP / (TP + FP)
    # 1. matched_ious > 0 means both detections and ground truths are non empty
    # 2. min matched_iou >= threshold means the detections are all TP in terms of the boxes
    # 3. color_flag is True means the recognized color is correct
    # 4. invalid_detections is None or len(invalid_detections) == 0 means that there is no FP
    # In our words, all detections are TPs, its precision is 100%
    if len(matched_ious) > 0 and \
        min(matched_ious) >= threshold and \
        color_flag and \
        (invalid_detections == None or len(invalid_detections) == 0):
        perfect_precision.append((i, i+cursor))
    else:
        flag = False

    ious = utils.IoU_multi(torch.tensor(boxes), valid_detections[:, 1:5].cpu())
    print('gt -> dt, ious: ')
    print(ious)
    maxes = torch.max(ious, 1)
    print(maxes)
    values = maxes.values
    indices = maxes.indices
    matched_ious = []
    color_flag2 = True
    for j, idx in enumerate(indices):
        gt = boxes[j]
        dt = valid_detections[idx]
        color_scores = recognitions[idx]
        gt_color = REC_COLORS.index(colors[j])
        dt_color = torch.argmax(color_scores)
        if gt_color != dt_color:
            color_flag2 = False
        iou = values[j]
        matched_ious.append(iou)
    # recall = TP / (TP + FN)
    # 1. matched_ious > 0 means both detections and ground truths are non empty
    # 2. min matched_iou >= threshold means all ground_truths have a corresponding TP detection in terms of the boxes, i.e. FN == 0
    # 3. color_flag2 is True means all the recognized colors is correct
    # every ground_truth has an IoU >= threshold with at least one detection, so 100% recall
    if len(matched_ious) > 0 and \
        min(matched_ious) >= threshold and \
        color_flag2: 
        perfect_recall.append((i, i+cursor))
    else:
        flag = False

    # flag means 100% precision and recall, i.e., no FP and FN
    if flag:
        perfect.append((i, i+cursor))

print(len(perfect_precision), len(perfect_recall), len(perfect))
with open('perfect_precision3.bin', 'wb') as f:
    pickle.dump(perfect_precision, f)

with open('perfect_recall3.bin', 'wb') as f:
    pickle.dump(perfect_recall, f)

with open('perfect3.bin', 'wb') as f:
    pickle.dump(perfect, f)