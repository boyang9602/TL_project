import cv2
import time
import pickle
import hungarian_optimizer
import utils
import torch
from adversarial import adversarial, create_objective, objective
from detector import TFModel
from recognizer import Recognizer
from pipeline import Pipeline

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

# dataset
# with open('S2TLD/perfect_cases.bin', 'rb') as f:
#     perfect_cases = pickle.load(f)
# print(f'Number of cases: {len(perfect_cases)}')

# run the pipeline
nonexistence = [1697, 1908, 2950]
detections = []
with torch.no_grad():
    t1 = time.perf_counter()
    for case in range(4567):
        if case in nonexistence:
            continue
        folder = 'normal_1' if case <= 778 else 'normal_2'
        image_file = 'S2TLD/{}/JPEGImages/{:06d}.jpg'.format(folder, case)
        annot_file = 'S2TLD/{}/Annotations/{:06d}.xml'.format(folder, case)
        image = torch.from_numpy(cv2.imread(image_file)).to(device)
        boxes = utils.readxml(annot_file)
        valid_detections, recognitions, assignments, invalid_detections = pipeline(image, boxes)
        detections.append((valid_detections, recognitions, assignments, invalid_detections))
        if case % 100 == 0:
            print(f'Progress: {case}')
    t2 = time.perf_counter()
    print(f"Run all the cases in {t2-t1:0.4f} seconds!")
with open('detections.bin', 'wb') as f:
    pickle.dump(detections, f)
# run the adversarial
# obj_fn = create_objective(objective, 5)
# t1 = time.perf_counter()
# for i, case in enumerate(perfect_cases):
#     image_file = 'S2TLD/JPEGImages/{:06d}.jpg'.format(case)
#     annot_file = 'S2TLD/Annotations/{:06d}.xml'.format(case)
#     image = torch.from_numpy(cv2.imread(image_file)).to(device)
#     boxes = utils.readxml(annot_file)
#     adv_img = adversarial(pipeline, image, boxes, obj_fn, 3, 16)
#     if adv_img == None:
#         print(i)
#     if i % 100 == 0:
#         print(f'Progress: {i}')
# t2 = time.perf_counter()
# print(f"Run all the adversarial in {t2-t1:0.4f} seconds!")
