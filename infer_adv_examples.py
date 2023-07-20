import os
import json
import cv2
import torch
import sys
from utils import IoU_multi, IoG_multi, readxml2
import pickle
from models.src.detector import TFModel
from models.src.recognizer import Recognizer
from models.src.pipeline import Pipeline
import hungarian_optimizer

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device is {device}')
TL_TYPE_TEXTS = ['VERT', 'QUAD', 'HORI']
REC_COLORS = ["Black", "Red", "Yellow", "Green"]
means_det = torch.Tensor([102.9801, 115.9465, 122.7717]).to(device)
means_rec = torch.Tensor([69.06, 66.58, 66.56]).to(device)

detector = TFModel()
detector.load_state_dict(torch.load('models/tl.torch'))
detector = detector.to(device)
detector.eval();

quad_pool_params = {'kernel_size': (4, 4), 'stride': (4, 4)}
hori_pool_params = {'kernel_size': (2, 6), 'stride': (2, 6)}
vert_pool_params = {'kernel_size': (6, 2), 'stride': (6, 2)}
quad_recognizer = Recognizer(quad_pool_params)
hori_recognizer = Recognizer(hori_pool_params)
vert_recognizer = Recognizer(vert_pool_params)

quad_recognizer.load_state_dict(torch.load('models/quad.torch'))
quad_recognizer = quad_recognizer.to(device)
quad_recognizer.eval();

hori_recognizer.load_state_dict(torch.load('models/hori.torch'))
hori_recognizer = hori_recognizer.to(device)
hori_recognizer.eval();

vert_recognizer.load_state_dict(torch.load('models/vert.torch'))
vert_recognizer = vert_recognizer.to(device)
vert_recognizer.eval();
classifiers = [(vert_recognizer, (96, 32, 3)), (quad_recognizer, (64, 64, 3)), (hori_recognizer, (32, 96, 3))]

ho = hungarian_optimizer.HungarianOptimizer()
pipeline = Pipeline(detector, classifiers, ho, means_det, means_rec)

with open('results/' + sys.argv[1] + '.bin', 'rb') as f:
    results = pickle.load(f)
with open('top200avg.bin', 'rb') as f:
    perfect_cases = pickle.load(f)

all = []

# IoU, IoG, color, location of all detections, the selected boxes
for perfect_case, result in zip(perfect_cases, results):
    folder = 'normal_1' if perfect_case[1] <= 778 else 'normal_2'
    image_file = 'S2TLD/{}/JPEGImages/{:06d}.jpg'.format(folder, perfect_case[1])
    annot_file = 'S2TLD/{}/Annotations/{:06d}.xml'.format(folder, perfect_case[1])
    boxes, colors = readxml2(annot_file)
    adv_image = result.cuda()
    # ana_data = []

    # detect the adv_image
    with torch.no_grad():
        valid, rec, assignments, invalid = pipeline(adv_image.type(torch.long), boxes)
        all.append((valid, rec, assignments, invalid))
with open('results/' + sys.argv[1] + '_detections.bin', 'wb') as f:
    pickle.dump(all, f)
