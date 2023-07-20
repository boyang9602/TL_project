import cv2
import utils
import torch
import pickle
from models.src.detector import TFModel
from models.src.recognizer import Recognizer
from models.src.pipeline import Pipeline
from utils import readxml2
from utils import preprocess4rec, preprocess4det
from utils import box2projection, crop
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
means_rec = torch.Tensor([69.06, 66.58, 66.56]).to(device)
REC_COLORS = ["off", "red", "yellow", "green"]
eps = 16


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

with open('top200avg.bin', 'rb') as f:
    perfect_cases = pickle.load(f)

# perfect TL box
count = 0
for x, perfect_case in enumerate(perfect_cases):
    folder = 'normal_1' if perfect_case[1] <= 778 else 'normal_2'
    image_file = 'S2TLD/{}/JPEGImages/{:06d}.jpg'.format(folder, perfect_case[1])
    annot_file = 'S2TLD/{}/Annotations/{:06d}.xml'.format(folder, perfect_case[1])
    image = torch.from_numpy(cv2.imread(image_file)).to(device)
    boxes, colors = readxml2(annot_file)
    recognizer, shape = classifiers[0] # all vertical
    for i, (box, color) in enumerate(zip(boxes, colors)):
        tl_box = preprocess4rec(image, box, shape, means_rec)
        adv_img = tl_box + torch.empty_like(tl_box.type(torch.float)).uniform_(-eps, eps)
        adv_img = adv_img.requires_grad_()
        output = recognizer(adv_img.permute(2, 0, 1).unsqueeze(0))
        tlcolor = REC_COLORS[torch.argmax(output).item()].lower()
        color_loss = -F.nll_loss(output, torch.tensor([REC_COLORS.index(color)], device=device))
        color_loss.backward()
        # this flag is true if all gradients are close to 0, else flase
        flag = torch.allclose(adv_img.grad, torch.zeros_like(adv_img.grad))
        # count the number of TL cases whose gradients are close to 0
        count += 1 if flag else 0 
        # plt.imshow((tl_box + means_rec).type(torch.int)[:,:,[2, 1, 0]])
        # plt.savefig(f'rec_results/{perfect_case[1].item()}_{i}_{tlcolor}.jpg')
        # plt.close()

print("perfect TL, 0 gradient count:", count)
# the result is 384. there are 386 in total. 2 of them are bad cases, I made a mistake to include them to the 200 perfect cases


# imperfect
count = 0
for x, perfect_case in enumerate(perfect_cases):
    folder = 'normal_1' if perfect_case[1] <= 778 else 'normal_2'
    image_file = 'S2TLD/{}/JPEGImages/{:06d}.jpg'.format(folder, perfect_case[1])
    annot_file = 'S2TLD/{}/Annotations/{:06d}.xml'.format(folder, perfect_case[1])
    image = torch.from_numpy(cv2.imread(image_file)).to(device)
    boxes, colors = readxml2(annot_file)
    recognizer, shape = classifiers[0] # all vertical
    for i, (box, color) in enumerate(zip(boxes, colors)):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        width = x2 - x1
        height = y2 - y1
        box = [max(x1 - width, 0), max(y1 - height, 0), min(x2 + width, 1280), min(y2 + height, 720)]
        tl_box = preprocess4rec(image, box, shape, means_rec)
        adv_img = tl_box + torch.empty_like(tl_box.type(torch.float)).uniform_(-eps, eps)
        adv_img = adv_img.requires_grad_()
        output = recognizer(adv_img.permute(2, 0, 1).unsqueeze(0))
        tlcolor = REC_COLORS[torch.argmax(output).item()].lower()
        color_loss = -F.nll_loss(output, torch.tensor([REC_COLORS.index(color)], device=device))
        color_loss.backward()
        # this flag is true if all gradients are close to 0, else flase
        flag = torch.allclose(adv_img.grad, torch.zeros_like(adv_img.grad), atol=1e-04)
        # count the number of TL cases whose gradients are close to 0
        count += 1 if flag else 0 
        # if not flag:
        #     # print(adv_img.grad)
        #     plt.imshow((tl_box + means_rec).type(torch.int)[:,:,[2, 1, 0]])
        #     plt.savefig(f'rec_results/imperfect_{perfect_case[1].item()}_{i}_{tlcolor}.jpg')
        #     plt.close()

print("imperfect TL, 0 gradient count:", count)
# 364

# use the entire crop box
count = 0
for x, perfect_case in enumerate(perfect_cases):
    folder = 'normal_1' if perfect_case[1] <= 778 else 'normal_2'
    image_file = 'S2TLD/{}/JPEGImages/{:06d}.jpg'.format(folder, perfect_case[1])
    annot_file = 'S2TLD/{}/Annotations/{:06d}.xml'.format(folder, perfect_case[1])
    image = torch.from_numpy(cv2.imread(image_file)).to(device)
    boxes, colors = readxml2(annot_file)
    recognizer, shape = classifiers[0] # all vertical
    for i, (box, color) in enumerate(zip(boxes, colors)):
        projection = box2projection(box)
        xl, xr, yt, yb = crop(image, projection)
        tl_box = preprocess4rec(image, [xl, yt, xr, yb], shape, means_rec)
        adv_img = tl_box + torch.empty_like(tl_box.type(torch.float)).uniform_(-eps, eps)
        adv_img = adv_img.requires_grad_()
        output = recognizer(adv_img.permute(2, 0, 1).unsqueeze(0))
        tlcolor = REC_COLORS[torch.argmax(output).item()].lower()
        color_loss = -F.nll_loss(output, torch.tensor([REC_COLORS.index(color)], device=device))
        color_loss.backward()
        # this flag is true if all gradients are close to 0, else flase
        flag = torch.allclose(adv_img.grad, torch.zeros_like(adv_img.grad), atol=1e-04)
        # count the number of TL cases whose gradients are close to 0
        count += 1 if flag else 0 
        # if not flag:
        #     # print(adv_img.grad)
        #     plt.imshow((tl_box + means_rec).type(torch.int)[:,:,[2, 1, 0]])
        #     plt.savefig(f'rec_results/imperfect_{perfect_case[1].item()}_{i}_{tlcolor}.jpg', bbox_inches='tight', dpi=300)
        #     plt.close()

print("ROI TL, 0 gradient count:", count)
