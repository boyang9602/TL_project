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

    count0 = 0 # use the ground truth detection box
    counta = 0
    count1 = 0 # use the a larger detection box, which is centered by the ground truth box
    countb = 0
    count2 = 0 # use the entire ROI
    countc = 0
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
            orig_cls_vector, adv_cls_vector = attack_recognizer(recognizer, tl_box, gt_color, device=device)
            orig_color = torch.argmax(orig_cls_vector).item()
            adv_color = torch.argmax(adv_cls_vector).item()
            count0 += gt_color == adv_color
            counta += gt_color == orig_color
            
            # case 1
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            width = x2 - x1
            height = y2 - y1
            image_width = ds720.item_shape()[1]
            image_height = ds720.item_shape()[0]
            box = [max(x1 - width, 0), max(y1 - height, 0), min(x2 + width, image_width), min(y2 + height, image_height)]
            tl_box = preprocess4rec(image, box, shape, means_rec)
            orig_cls_vector, adv_cls_vector = attack_recognizer(recognizer, tl_box, gt_color, device=device)
            orig_color = torch.argmax(orig_cls_vector).item()
            adv_color = torch.argmax(adv_cls_vector).item()
            count1 += gt_color == adv_color
            countb += gt_color == orig_color

            # case 2
            projection = box2projection(box)
            xl, xr, yt, yb = crop(image, projection)
            tl_box = preprocess4rec(image, [xl, yt, xr, yb], shape, means_rec)
            orig_cls_vector, adv_cls_vector = attack_recognizer(recognizer, tl_box, gt_color, device=device)
            orig_color = torch.argmax(orig_cls_vector).item()
            adv_color = torch.argmax(adv_cls_vector).item()
            count2 += gt_color == adv_color
            countc += gt_color == orig_color
    print(f'Ground Truth detection box\n\t#Correct on original picture: {counta}\n\t#Correct on adversarial picture: {count0}')
    print(f'A bigger detection box\n\t#Correct on original picture: {countb}\n\t#Correct on adversarial picture: {count1}')
    print(f'ROI as the detection box\n\t#Correct on original picture: {countc}\n\t#Correct on adversarial picture: {count2}')
    print('Total cases: ', total)