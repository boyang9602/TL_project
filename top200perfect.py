import pickle
import csv
import torch
import utils
import shutil
import xml.etree.ElementTree as ET

import cv2
import hungarian_optimizer
from detector import TFModel
from recognizer import Recognizer
from pipeline import Pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device is {device}')
TL_TYPE_TEXTS = ['VERT', 'QUAD', 'HORI']
REC_COLORS = ["off", "red", "yellow", "green"]
# means_det = torch.Tensor([102.9801, 115.9465, 122.7717]).to(device)
# means_rec = torch.Tensor([69.06, 66.58, 66.56]).to(device)

# detector = TFModel()
# detector.load_state_dict(torch.load('models/tl.torch'))
# detector = detector.to(device)
# detector.eval();

# quad_pool_params = {'kernel_size': (4, 4), 'stride': (4, 4)}
# hori_pool_params = {'kernel_size': (2, 6), 'stride': (2, 6)}
# vert_pool_params = {'kernel_size': (6, 2), 'stride': (6, 2)}
# quad_recognizer = Recognizer(quad_pool_params)
# hori_recognizer = Recognizer(hori_pool_params)
# vert_recognizer = Recognizer(vert_pool_params)

# quad_recognizer.load_state_dict(torch.load('models/quad.torch'))
# quad_recognizer = quad_recognizer.to(device)
# quad_recognizer.eval();

# hori_recognizer.load_state_dict(torch.load('models/hori.torch'))
# hori_recognizer = hori_recognizer.to(device)
# hori_recognizer.eval();

# vert_recognizer.load_state_dict(torch.load('models/vert.torch'))
# vert_recognizer = vert_recognizer.to(device)
# vert_recognizer.eval();
# classifiers = [(vert_recognizer, (96, 32, 3)), (quad_recognizer, (64, 64, 3)), (hori_recognizer, (32, 96, 3))]

# ho = hungarian_optimizer.HungarianOptimizer()
# pipeline = Pipeline(detector, classifiers, ho, means_det, means_rec)

with open('perfect3.bin', 'rb') as f:
    perfect_cases = pickle.load(f)
with open('detections.bin', 'rb') as f:
    detections = pickle.load(f)


avg_ious = []
# detections = []
for i, case in perfect_cases:
    folder = 'normal_1' if case <= 778 else 'normal_2'
    image_file = 'S2TLD/{}/JPEGImages/{:06d}.jpg'.format(folder, case)
    annot_file = 'S2TLD/{}/Annotations/{:06d}.xml'.format(folder, case)
    image = torch.from_numpy(cv2.imread(image_file)).to(device)
    boxes, colors = utils.readxml2(annot_file)
    # with torch.no_grad():
    #     detection, assignments = pipeline(image, boxes)
    # detections.append((detection, assignments))

    valid_detections, recognitions, assignments, invalid_detections = detections[i]
    ious = utils.IoU_multi(valid_detections[:, 1:5].cpu(), torch.tensor(boxes))
    assert invalid_detections == None or len(invalid_detections) == 0

    matched_ious = []
    for item in assignments:
        gt_idx = item[0]
        dt_idx = item[1]
        iou = utils.IoU_single(boxes[gt_idx], valid_detections[dt_idx][1:5])
        assert iou >= 0.5
        matched_ious.append(iou)
        gt_color = colors[gt_idx]
        dt_color = REC_COLORS[torch.argmax(recognitions[dt_idx]).item()].lower()
        assert gt_color == dt_color
    avg_ious.append(sum(matched_ious)/len(matched_ious))
with open('avg_ious4perfect_cases.bin', 'wb') as f:
    pickle.dump(avg_ious, f)
top200avg = torch.argsort(torch.tensor(avg_ious), descending=True)[:200]
top200_cases = torch.tensor(perfect_cases, dtype=torch.long)[top200avg]
with open('top200avg.bin', 'wb') as f:
    pickle.dump(top200_cases, f)

# for i in range(200):
#     case = top200_cases[i][1]
#     shutil.copy('S2TLD/annotations/{:06d}.xml'.format(case), 'top200/annotations/{:06d}.xml'.format(case))
#     shutil.copy('S2TLD/pictures/{:06d}.jpg'.format(case), 'top200/pictures/{:06d}.jpg'.format(case))
