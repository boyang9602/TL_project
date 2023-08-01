import torch
import pickle
from models.src.pipeline import load_pipeline
from tools.dataset import get_dataset
import attack.adversarial as adversarial
import argparse
import os

def load_topk_idxs(filename):
    topk = []
    if filename.endswith('.txt'):
        with open(filename, 'r') as f:
            for line in f.readlines():
                topk.append(int(line.split(',')[0]))
    else:
        with open(filename, 'rb') as f:
            tmp = pickle.load(f)
            for item in tmp:
                topk.append(item[0])
    return topk

def get_loss_fns(box_loss_name, type_loss_name, color_loss_name, rpn_objectiveness_loss_name):
    box_loss = getattr(adversarial, box_loss_name)
    type_loss = getattr(adversarial, type_loss_name)
    color_loss = getattr(adversarial, color_loss_name)
    rpn_objectiveness_loss = getattr(adversarial, rpn_objectiveness_loss_name)
    loss_fns = {
        'box_loss': box_loss,
        'type_loss': type_loss,
        'color_loss': color_loss,
        'rpn_objectiveness_loss': rpn_objectiveness_loss
    }
    return loss_fns

if __name__ == '__main__':
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description='Find the perfect cases in terms of precision and recall. It will output 3 files, perfect_precision.bin, perfect_recall.bin, perfect_precison_recall.bin.')
    parser.add_argument('--dataset', '-ds', action='store', required=True, help='the dataset.')
    parser.add_argument('--topk_file', '-f', action='store', required=False, default=None, help='the selected perfect cases.')
    parser.add_argument('--path', '-p', action='store', required=False, default=None, help='the output path.')

    parser.add_argument('--box_loss', '-b', action='store', required=False, default='dummy_loss', help='box loss function name')
    parser.add_argument('--type_loss', '-t', action='store', required=False, default='dummy_loss', help='box loss function name')
    parser.add_argument('--color_loss', '-c', action='store', required=False, default='dummy_loss', help='box loss function name')
    parser.add_argument('--objectiveness_loss', '-o', action='store', required=False, default='dummy_loss', help='rpn layer objectiveness loss function name')
    args = parser.parse_args()

    pl = load_pipeline(device)
    ds = get_dataset(args.dataset, device=device)

    def objective_fn(data_item, output):
        boxes = torch.tensor(data_item['boxes'], device=device)
        colors = data_item['colors']
        return adversarial.objective(boxes, colors, output, get_loss_fns(args.box_loss, args.type_loss, args.color_loss, args.objectiveness_loss))

    adv_imgs = []
    topk_filename = args.topk_file
    if topk_filename is None:
        topk_filename = f'data/evaluation/{args.dataset}_top200.bin'
    for i, idx in enumerate(load_topk_idxs(topk_filename)):
        if i > 5:
            break
        data_item = ds[idx]
        
        eps = 16
        step_size = 3
        max_iter = 5
        
        adv_img = adversarial.adversarial(pl, data_item, objective_fn, step_size=step_size, eps=eps, budget=max_iter, device=device)
        adv_imgs.append(adv_img.type(torch.uint8).cpu())

    adv_imgs = torch.stack(adv_imgs)

    default_path = f'data/adversarial_results/{args.dataset}/'
    if args.path is None:
        filename = f'{default_path}{args.box_loss}_{args.type_loss}_{args.color_loss}_{args.objectiveness_loss}_{eps}_{step_size}_{max_iter}.bin'
    else:
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        filename = f'{args.path}/{args.box_loss}_{args.type_loss}_{args.color_loss}_{args.objectiveness_loss}_{eps}_{step_size}_{max_iter}.bin'
    print(filename)
    with open(filename, 'wb') as f:
        pickle.dump(adv_imgs, f)
