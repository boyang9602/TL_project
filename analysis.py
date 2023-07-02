import os
import json
from utils import IoU_multi, IoG_multi, readxml2
import torch
import pickle
import torch
import cv2
from detector import TFModel
from recognizer import Recognizer
from pipeline import Pipeline
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


with open('rpn_exp1.bin', 'rb') as f:
    results = pickle.load(f)
with open('top200avg.bin', 'rb') as f:
    perfect_cases = pickle.load(f)

# IoU, IoG, color, location of all detections, the selected boxes
for perfect_case, result in zip(perfect_cases, results):
    folder = 'normal_1' if perfect_case[1] <= 778 else 'normal_2'
    image_file = 'S2TLD/{}/JPEGImages/{:06d}.jpg'.format(folder, perfect_case[1])
    annot_file = 'S2TLD/{}/Annotations/{:06d}.xml'.format(folder, perfect_case[1])
    boxes, colors = readxml2(annot_file)
    adv_image = result
    ana_data = []

    # detect the adv_image
    with torch.no_grad():
        valid, rec, assignments, invalid = pipeline(adv_image.type(torch.long), boxes)

    # get the IoU and IoG of each detection
    ious = IoU_multi(valid[:, 1:5], torch.tensor(boxes))
    iogs = IoG_multi(valid[:, 1:5], torch.tensor(boxes))

    # label each detection if it is selected
    # save the information to a file
    # save the pictures with annotation
    # cursor = 0
    for i, box in enumerate(boxes):
        tmp = []
        matched = -1
        # if assignments != None and cursor < len(assignments) and assignments[cursor, 0] == i:
        #     matched = assignments[cursor, 1]
        #     cursor += 1
        if assignments != None and len(assignments) > 0 and i in assignments[:, 0]:
            matched = (assignments[:, 0] == i).nonzero()[0][0]

        # unk_count = 0
        for j, detection in enumerate(valid):
            tltype = torch.argmax(detection[5:9]).item() - 1
            tlcolor = 'unkown'
            if tltype == -1:
                tltype = 'unknown'
                # unk_count += 1
                tmp.append({
                    'selected': False,
                    'iou': f'{int(round(ious[j, i].item() * 100))}%',
                    'iog': f'{int(round(iogs[j, i].item() * 100))}%',
                    'type': tltype,
                    'color': tlcolor,
                    'coor': detection[1:5].type(torch.long).tolist()
                })
            else:
                tltype = TL_TYPE_TEXTS[tltype].lower()
                tlcolor = REC_COLORS[torch.argmax(rec[j]).item()].lower()
                tmp.append({
                    'selected': bool(matched == j),
                    'iou': f'{int(round(ious[j, i].item() * 100))}%',
                    'iog': f'{int(round(iogs[j, i].item() * 100))}%',
                    'type': tltype,
                    'color': tlcolor,
                    'coor': detection[1:5].type(torch.long).tolist()
                })
        ana_data.append({
            'ground_truth': {
                'color': colors[i],
                'box': box
            },
            'detection': tmp
        })
    # print()
    # fig, ax = viz_pipeline_results(bgr2rgb(adv_image.type(torch.long)),
    #                                valid_detections=valid, recognitions=rec,
    #                                assignments=assignments, invalid_detections=invalid,
    #                                projections=boxes2projections(boxes))
    if not os.path.exists(f'analysis_data11/{perfect_case[1]}/'):
        os.makedirs(f'analysis_data11/{perfect_case[1]}/')
    # fig.savefig(f'analysis_data4/{perfect_case[1]}/adv_img_w_anno.jpg', bbox_inches='tight', dpi=200)
    # plt.close();
    with open(f'analysis_data11/{perfect_case[1]}/analysis_data.json', 'w') as f:
        json.dump(ana_data, f)