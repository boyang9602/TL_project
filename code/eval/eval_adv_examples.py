import sys
import tools.utils as utils
import pickle
import torch
from torch import IntTensor, Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.iou import IntersectionOverUnion

with open('results/' + sys.argv[1] + '_detections.bin', 'rb') as f:
    detections = pickle.load(f)
with open('top200avg.bin', 'rb') as f:
    perfect_cases = pickle.load(f)
REC_COLORS = ["off", "red", "yellow", "green"]
nonexistence = [1697, 1908, 2950]
counter = 0
preds = []
preds_valid = []
preds_w_ass = []
targets = []
eval_data = []
with torch.no_grad():
    for (valid_detections, recognitions, assignments, invalid_detections), (idx, case) in zip(detections, perfect_cases):
        if case in nonexistence:
            counter += 1
            continue
        folder = 'normal_1' if case <= 778 else 'normal_2'
        image_file = 'S2TLD/{}/JPEGImages/{:06d}.jpg'.format(folder, case)
        annot_file = 'S2TLD/{}/Annotations/{:06d}.xml'.format(folder, case)
        boxes, colors = utils.readxml2(annot_file)
        target = {
            'boxes': Tensor(boxes).cuda(),
            'labels': IntTensor([1 for i in range(len(boxes))]).type(torch.long).cuda()
        }
        targets.append(target)
        if invalid_detections != None:
            all_detections = torch.vstack([valid_detections, invalid_detections])
            maxes = torch.max(all_detections[:, 5:], 1)
            pred = {
                'boxes': all_detections[:, 1:5].cuda(),
                'scores': maxes.values.cuda(),
                'labels': maxes.indices.type(torch.long).cuda()
            }
        else:
            all_detections = valid_detections
            maxes = torch.max(all_detections[:, 5:], 1)
            pred = {
                'boxes': all_detections[:, 1:5].cuda(),
                'scores': maxes.values.cuda(),
                'labels': maxes.indices.type(torch.long).cuda()
            }
        preds.append(pred)

        maxes = torch.max(valid_detections[:, 5:], 1)
        pred_valid = {
            'boxes': valid_detections[:, 1:5].cuda(),
            'scores': maxes.values.cuda(),
            'labels': maxes.indices.type(torch.long).cuda()
        }
        # print(pred_valid)
        preds_valid.append(pred_valid)

        # for each picture, we record the corresponding matched box or None
        curr = []
        if assignments == None:
            # in this case, all the ground truth TLs are failed to be detected
            for box, color in zip(boxes, colors):
                result = {
                    'box': {
                        'gt': box,
                        'dt': None
                    },
                    'color': {
                        'gt': color,
                        'dt': None
                    }
                }
                curr.append(result)
        else:
            for i, (box, color) in enumerate(zip(boxes, colors)):
                idxs = torch.nonzero(assignments[:, 0] == i).squeeze()
                assert idxs.numel() <= 1
                if idxs.numel() == 0:
                    # no match
                    result = {
                        'box': {
                            'gt': box,
                            'dt': None
                        },
                        'color': {
                            'gt': color,
                            'dt': None
                        }
                    }
                else:
                    # matched
                    assignment = assignments[idxs.item()]
                    result = {
                        'box': {
                            'gt': box,
                            'dt': valid_detections[assignment[1]]
                        },
                        'color': {
                            'gt': color,
                            'dt': REC_COLORS[torch.argmax(recognitions[assignment[1]])]
                        }
                    }
                curr.append(result)
        eval_data.append(curr)

correct_colors = 0
incorrect_colors = 0
detected_lights = 0 # set the thres as 0.0001, very tolerant
detected_n_correct_colors = 0
detected_n_incorrect_colors = 0
undetected_lights = 0

avg_precision = 0
avg_recall = 0
for pic_data in eval_data:
    # for all lights in all pictures
    # pre, recall
    for light in pic_data:
        if light['color']['dt'] == None and light['box']['dt'] == None:
            undetected_lights += 1
            continue
        assert light['color']['dt'] != None and light['box']['dt'] != None

        if light['color']['gt'] == light['color']['dt']:
            correct_colors += 1
        else:
            incorrect_colors += 1
        iou = utils.IoU_single(light['box']['gt'], light['box']['dt'][1:5])
        if iou > 0.0001:
            detected_lights += 1
            if light['color']['gt'] == light['color']['dt']:
                detected_n_correct_colors += 1
            else:
                detected_n_incorrect_colors += 1
        else:
            undetected_lights += 1
    
    # curr_correct_colors = 0
    # curr_incorrect_colors = 0
    # # rec = correct / all_gts
    # # pre = correct / all_detections

print('correct_colors: ', correct_colors, correct_colors/390*100)
print('incorrect_colors: ', incorrect_colors, incorrect_colors/390*100)
print('detected_lights: ', detected_lights, detected_lights/390*100)
print('undetected_lights: ', undetected_lights, undetected_lights/390*100)
print('detected_n_correct_colors', detected_n_correct_colors, detected_n_correct_colors/390*100)
print('detected_n_incorrect_colors', detected_n_incorrect_colors, detected_n_incorrect_colors/390*100)
print('total_lights: ', 390)

metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
metric.update(preds, targets)
result = metric.compute()
print(result)

print('--')
metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
metric.update(preds_valid, targets)
result = metric.compute()
print(result)

# print('--')
# metric = IntersectionOverUnion()
# metric.update(preds, targets)
# result = metric.compute()
# print(result)
# print('--')
# metric = IntersectionOverUnion()
# metric.update(preds_valid, targets)
# result = metric.compute()
# print(result)

print()