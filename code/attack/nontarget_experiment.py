import torch
import pickle
from models.src.pipeline import load_pipeline
from tools.dataset import get_dataset
import attack.adversarial as adversarial
import argparse
import os
import time

TL_TYPES = ['UNK', 'VERT', 'QUAD', 'HORI']
COLOR_LABELS = ["off", "red", "yellow", "green"]

def convert_labels_to_nums(labels, label_list):
    ret = []
    for label in labels:
        ret.append(label_list.index(label))
    return ret

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

def get_loss_fns(rcnn_reg_loss_name, rcnn_cls_loss_name, rec_cls_loss_name, rpn_reg_loss_name, rpn_cls_loss_name):
    loss_fns = {
        'rcnn_reg_loss': getattr(adversarial, rcnn_reg_loss_name),
        'rcnn_cls_loss': getattr(adversarial, rcnn_cls_loss_name),
        'rec_cls_loss': getattr(adversarial, rec_cls_loss_name),
        'rpn_reg_loss': getattr(adversarial, rpn_reg_loss_name),
        'rpn_cls_loss': getattr(adversarial, rpn_cls_loss_name)
    }
    return loss_fns

if __name__ == '__main__':
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description='Run the non target attacks')
    parser.add_argument('--dataset', '-ds', action='store', required=True, help='the dataset.')
    parser.add_argument('--topk_file', '-f', action='store', required=False, default=None, help='the selected perfect cases.')
    parser.add_argument('--path', '-p', action='store', required=False, default=None, help='the output path.')

    parser.add_argument('--eps', '-e', action='store', required=False, default=16, type=int)
    parser.add_argument('--step_size', '-s', action='store', required=False, default=3, type=int)
    parser.add_argument('--max_iter', '-m', action='store', required=False, default=5, type=int)

    parser.add_argument('--rcnn_reg_loss', '-b', action='store', required=False, default='dummy_loss', help='RCNN box loss function name')
    parser.add_argument('--rcnn_cls_loss', '-t', action='store', required=False, default='dummy_loss', help='RCNN type loss function name')
    parser.add_argument('--rec_cls_loss', '-c', action='store', required=False, default='dummy_loss', help='Recognizer cls loss function name')
    parser.add_argument('--rpn_reg_loss', '-rb', action='store', required=False, default='dummy_loss', help='RPN box loss function name')
    parser.add_argument('--rpn_cls_loss', '-o', action='store', required=False, default='dummy_loss', help='RPN layer objectiveness loss function name')
    args = parser.parse_args()

    pl = load_pipeline(device)
    ds = get_dataset(args.dataset, device=device)

    def objective_fn(data_item, output):
        boxes = torch.tensor(data_item['boxes'], device=device)
        colors = data_item['colors']
        inferred_tl_types = data_item['inferred_tl_types']
        assert 'NA' not in inferred_tl_types
        loss_fns = get_loss_fns(args.rcnn_reg_loss, args.rcnn_cls_loss, args.rec_cls_loss, args.rpn_reg_loss, args.rpn_cls_loss)
        return adversarial.objective(boxes, convert_labels_to_nums(colors, COLOR_LABELS), convert_labels_to_nums(inferred_tl_types, TL_TYPES), output, loss_fns)

    adv_imgs = []
    topk_filename = args.topk_file
    if topk_filename is None:
        topk_filename = f'data/evaluation/{args.dataset}_top200.bin'
    
    
    t1 = time.perf_counter()
    for i, idx in enumerate(load_topk_idxs(topk_filename)):
        data_item = ds[idx]
        
        adv_img = adversarial.adversarial(pl, data_item, objective_fn, step_size=args.step_size, eps=args.eps, budget=args.max_iter, device=device)
        adv_imgs.append(adv_img.type(torch.uint8).cpu())
    t2 = time.perf_counter()
    print(f"Run all the cases in {t2-t1:0.4f} seconds!")

    adv_imgs = torch.stack(adv_imgs)

    default_path = f'data/adversarial_results/{args.dataset}/'
    if args.path is None:
        filename = f'{default_path}{args.rcnn_reg_loss}_{args.rcnn_cls_loss}_{args.rec_cls_loss}_{args.rpn_reg_loss}_{args.rpn_cls_loss}_{args.eps}_{args.step_size}_{args.max_iter}.bin'
    else:
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        filename = f'{args.path}/{args.rcnn_reg_loss}_{args.rcnn_cls_loss}_{args.rec_cls_loss}_{args.rpn_reg_loss}_{args.rpn_cls_loss}_{args.eps}_{args.step_size}_{args.max_iter}.bin'
    with open(filename, 'wb') as f:
        pickle.dump(adv_imgs, f)
