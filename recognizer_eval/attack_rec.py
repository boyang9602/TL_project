import sys
import torch
import pickle
from models.src.pipeline import load_pipeline
from utils import preprocess4rec, box2projection, crop
from dataset import get_dataset
import torch.nn.functional as F

def attack_recognizer(recognizer, tl_box, gt_color, eps=16, step_size=3, max_it=10, device=None):
    with torch.no_grad():
        orig_cls_vector = recognizer(tl_box.permute(2, 0, 1).unsqueeze(0))

    adv_img = tl_box + torch.empty_like(tl_box.type(torch.float)).uniform_(-eps, eps)
    adv_img = adv_img.requires_grad_()

    optimizer = torch.optim.Adam([adv_img], lr=step_size)
    for _ in range(max_it):
        output = recognizer(adv_img.permute(2, 0, 1).unsqueeze(0))
        color_loss = -F.nll_loss(output, torch.tensor([gt_color], device=device))
        color_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        adv_cls_vector = recognizer(adv_img.permute(2, 0, 1).unsqueeze(0))

    return orig_cls_vector, adv_cls_vector

def check_attack_gradients(recognizer, tl_box, gt_color, eps=16, device=None):
    tl_box = preprocess4rec(image, box, shape, means_rec)
    adv_img = tl_box + torch.empty_like(tl_box.type(torch.float)).uniform_(-eps, eps)
    adv_img = adv_img.requires_grad_()
    output = recognizer(adv_img.permute(2, 0, 1).unsqueeze(0))
    color_loss = -F.nll_loss(output, torch.tensor([gt_color], device=device))
    color_loss.backward()
    return adv_img.grad

def att_n_report(recognizer, tl_box, gt_color, device=None):
    orig_cls_vector, adv_cls_vector = attack_recognizer(recognizer, tl_box, gt_color, device=device)
    orig_color = torch.argmax(orig_cls_vector).item()
    adv_color = torch.argmax(adv_cls_vector).item()
    flag_adv = gt_color == adv_color
    flag_orig = gt_color == orig_color

    grad = check_attack_gradients(recognizer, tl_box, gt_color, device=device)
    flag_zg = torch.allclose(grad, torch.zeros_like(grad), atol=1e-4)

    return flag_orig, flag_adv, flag_zg

if __name__ == '__main__':
    # set manual seed for reproducibility
    torch.manual_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    color_labels = ["off", "red", "yellow", "green"]

    ds720 = get_dataset(sys.argv[1], device)

    pl = load_pipeline(device)
    recognizer, shape = pl.classifiers[0]
    means_rec = pl.means_rec

    # tech debt. should be parameterized
    with open('top200avg.bin', 'rb') as f:
        perfect_cases = pickle.load(f)

    # correct recognitions
    # use the ground truth detection box
    count_adv0 = 0 
    count_orig0 = 0
    count_zg0 = 0
    # use the a larger detection box, which is centered by the ground truth box
    count_adv1 = 0 
    count_orig1 = 0
    count_zg1 = 0
    # use the entire ROI
    count_adv2 = 0 
    count_orig2 = 0
    count_zg2 = 0

    total = 0
    for perfect_case in perfect_cases:
        item = ds720[perfect_case[0]]
        image = item['image']
        boxes = item['boxes']
        colors = item['colors']
        for i, (box, color) in enumerate(zip(boxes, colors)):
            total += 1
            gt_color = color_labels.index(color)

            # case 0
            tl_box = preprocess4rec(image, box, shape, means_rec)
            flag_orig0, flag_adv0, flag_zg0 = att_n_report(recognizer, tl_box, gt_color, device=device)
            count_orig0 += flag_orig0
            count_adv0 += flag_adv0
            count_zg0 += flag_zg0

            # case 1
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            width = x2 - x1
            height = y2 - y1
            image_width = ds720.item_shape()[1]
            image_height = ds720.item_shape()[0]
            box = [max(x1 - width, 0), max(y1 - height, 0), min(x2 + width, image_width), min(y2 + height, image_height)]
            tl_box = preprocess4rec(image, box, shape, means_rec)
            flag_orig1, flag_adv1, flag_zg1 = att_n_report(recognizer, tl_box, gt_color, device=device)
            count_orig1 += flag_orig1
            count_adv1 += flag_adv1
            count_zg1 += flag_zg1

            # case 2
            projection = box2projection(box)
            xl, xr, yt, yb = crop(image, projection)
            tl_box = preprocess4rec(image, [xl, yt, xr, yb], shape, means_rec)
            flag_orig2, flag_adv2, flag_zg2 = att_n_report(recognizer, tl_box, gt_color, device=device)
            count_orig2 += flag_orig2
            count_adv2 += flag_adv2
            count_zg2 += flag_zg2

    print('Total cases: ', total)
    print(f'Ground Truth detection box\n\t#Correct on original picture: {count_orig0}\n\t#Correct on adversarial picture: {count_adv0}\n\t#Zero grad: {count_zg0}\n\tSuccess rate: {100 - count_adv0 / total * 100:.2f}%')
    print(f'A bigger detection box\n\t#Correct on original picture: {count_orig1}\n\t#Correct on adversarial picture: {count_adv1}\n\t#Zero grad: {count_zg1}\n\tSuccess rate: {100 - count_adv1 / total * 100:.2f}%')
    print(f'ROI as the detection box\n\t#Correct on original picture: {count_orig2}\n\t#Correct on adversarial picture: {count_adv2}\n\t#Zero grad: {count_zg2}\n\tSuccess rate: {100 - count_adv2 / total * 100:.2f}%')
