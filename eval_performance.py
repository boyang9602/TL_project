import utils
import pickle
import torch
from torch import IntTensor, Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.iou import IntersectionOverUnion

with open('detections.bin', 'rb') as f:
    detections = pickle.load(f)
REC_COLORS = ["off", "red", "yellow", "green"]
nonexistence = [1697, 1908, 2950]
counter = 0
recognition_eval_data = []
preds = []
preds_valid = []
preds_w_ass = []
targets = []
with torch.no_grad():
    for case in range(4567):
        if case in nonexistence:
            counter += 1
            continue
        folder = 'normal_1' if case <= 778 else 'normal_2'
        image_file = 'S2TLD/{}/JPEGImages/{:06d}.jpg'.format(folder, case)
        annot_file = 'S2TLD/{}/Annotations/{:06d}.xml'.format(folder, case)
        boxes, colors = utils.readxml2(annot_file)
        target = {
            'boxes': Tensor(boxes),
            'labels': IntTensor([1 for i in range(len(boxes))]).type(torch.long)
        }
        targets.append(target)
        valid_detections, recognitions, assignments, invalid_detections = detections[case - counter]
        if invalid_detections != None:
            all_detections = torch.vstack([valid_detections, invalid_detections])
            maxes = torch.max(all_detections[:, 5:], 1)
            pred = {
                'boxes': all_detections[:, 1:5],
                'scores': maxes.values,
                'labels': maxes.indices.type(torch.long)
            }
        else:
            all_detections = valid_detections
            maxes = torch.max(all_detections[:, 5:], 1)
            pred = {
                'boxes': all_detections[:, 1:5],
                'scores': maxes.values,
                'labels': maxes.indices.type(torch.long)
            }
        preds.append(pred)

        maxes = torch.max(valid_detections[:, 5:], 1)
        pred_valid = {
            'boxes': valid_detections[:, 1:5],
            'scores': maxes.values,
            'labels': maxes.indices.type(torch.long)
        }
        # print(pred_valid)
        preds_valid.append(pred_valid)

        correct = 0
        incorrect = 0
        if assignments == None:
            miss = len(boxes)
            correct = 0
            incorrect = 0
            continue
        else:
            miss = len(boxes) - len(assignments)
            for assignment in assignments:
                box = boxes[assignment[0]]
                color = colors[assignment[0]]
                d_box = detections[assignment[1]]
                r_color = REC_COLORS[torch.argmax(recognitions[assignment[1]])]
                if color == r_color:
                    correct += 1
                else:
                    incorrect += 1
        recognition_eval_data.append([correct, incorrect, miss])
recognition_eval_data = torch.Tensor(recognition_eval_data)
print('correct: ', sum(recognition_eval_data[:, 0]), )
print('incorrect: ', sum(recognition_eval_data[:, 1]))
print('miss: ', sum(recognition_eval_data[:, 2]))
print('all: ', sum(sum(recognition_eval_data)))

metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
metric.update(preds, targets)
result = metric.compute()
print(result)

print('--')
metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
metric.update(preds_valid, targets)
result = metric.compute()
print(result)

print('--')
metric = IntersectionOverUnion()
metric.update(preds, targets)
result = metric.compute()
print(result)
print('--')
metric = IntersectionOverUnion()
metric.update(preds_valid, targets)
result = metric.compute()
print(result)
