import torch
import cv2
import time
import pickle
from detector import TFModel
from recognizer import Recognizer
from pipeline import Pipeline
import hungarian_optimizer
import utils
import adversarial
import sys

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

with open('top200avg.bin', 'rb') as f:
    perfect_cases = pickle.load(f)

torch.manual_seed(42)
obj = sys.argv[1]
adv_imgs = []
for i, (idx, case) in enumerate(perfect_cases):
    # if i > 5:
    #     break
    folder = 'normal_1' if case <= 778 else 'normal_2'
    image_file = 'S2TLD/{}/JPEGImages/{:06d}.jpg'.format(folder, case)
    annot_file = 'S2TLD/{}/Annotations/{:06d}.xml'.format(folder, case)
    image = torch.from_numpy(cv2.imread(image_file)).to(device)
    boxes, colors = utils.readxml2(annot_file)
    objfn = getattr(adversarial, obj)
    adv_img = adversarial.adversarial(pipeline, image, boxes, objfn, 3, 16, 5)
    adv_imgs.append(adv_img.type(torch.short).cpu())
adv_imgs = torch.stack(adv_imgs)
with open(f'{obj}.bin', 'wb') as f:
    pickle.dump(adv_imgs, f)
