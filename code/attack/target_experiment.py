import os
import time
import pickle
import argparse
import torch
import torch.nn.functional as F
from tools.utils import IoU_multi
from models.pipeline4attack import load_pipeline
from tools.dataset import get_dataset
import attack.adversarial as adversarial

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

def objective(boxes, colors, inferred_tl_types, output, loss_fns):
    """
    Target objective. It will find all the detected boxes which do not have intersection with the ground truth in RPN and RCNN layer. 
    For these boxes, it will try to increase their confidence scores.
    """
    valid_detections, recognitions, assignments, invalid_detections, rpn_data, rcnn_boxes, rcnn_scores = output

    # rcnn_boxes shape (n, 4, 4), rcnn_scores shape (n, 4)
    assert rcnn_boxes.shape[0] == rcnn_scores.shape[0]
    indices = torch.arange(rcnn_boxes.shape[0])
    rcnn_scores_maxes, rcnn_scores_argmaxes = torch.max(rcnn_scores[:, 1:], 1)
    rcnn_scores_argmaxes += 1
    rcnn_boxes = rcnn_boxes[indices, rcnn_scores_argmaxes] # get the box pointed by the highest class

    score = 0

    ious = IoU_multi(rcnn_boxes, boxes)
    iou_maxes, iou_argmaxes = torch.max(ious, 1)

    good_inds = []
    bad_inds = []

    for det_idx, (gt_idx, iou) in enumerate(zip(iou_argmaxes, iou_maxes)):
        if iou <= 1e-4:
            # no intersection with ground truth, let's help it to get higher confidence scores
            bad_inds.append(det_idx)
        else:
            good_inds.append(det_idx)

    # for the non-overlapped boxes, increase their confidence scores, and try to keep the box position
    for idx in bad_inds:
        score += loss_fns['rcnn_reg_loss'](rcnn_boxes[idx], rcnn_boxes[idx].detach().clone(), iou_maxes[idx], True)
        score += loss_fns['rcnn_cls_loss'](rcnn_scores_argmaxes[idx], rcnn_scores[idx], True)

    return score

def dummy_loss(*args):
    return 0

def box_smooth_l1_loss(gt_box, det_box, iou, gt_preferred):
    # try to make the negative loss of the det_box and gt_box smaller
    # so that the loss will be bigger, which means det_box and gt_box become 
    # much more different
    if gt_preferred: # gt is not preferred, so we make 2 boxes more different
        return F.smooth_l1_loss(gt_box, det_box)
    else:
        return -F.smooth_l1_loss(gt_box, det_box)

def box_iou_loss(gt_box, det_box, iou, gt_preferred):
    # try to make the iou smaller
    if gt_preferred: # gt is preferred, try to increase iou
        return -iou
    else:
        return iou

def cls_nll_loss(gt_idx, scores_vec, gt_preferred):
    if gt_preferred: # minimize because preferred
        return F.nll_loss(scores_vec.unsqueeze(0), torch.tensor([gt_idx], device=scores_vec.device))
    else:
        return -F.nll_loss(scores_vec.unsqueeze(0), torch.tensor([gt_idx], device=scores_vec.device))

def cls_gt_score_loss(gt_idx, scores_vec, gt_preferred):
    if gt_preferred: # preferred, maxize the score
        return -scores_vec[gt_idx]
    else:
        return scores_vec[gt_idx]

def handle_args():
    parser = argparse.ArgumentParser(description='Run the non target attacks')
    parser.add_argument('--dataset', '-ds', action='store', required=True, help='the dataset.')
    parser.add_argument('--topk_file', '-f', action='store', required=False, default=None, help='the selected perfect cases.')
    parser.add_argument('--path', '-p', action='store', required=False, default=None, help='the output path.')

    parser.add_argument('--eps', '-e', action='store', required=False, default=16, type=int)
    parser.add_argument('--step_size', '-s', action='store', required=False, default=3, type=int)
    parser.add_argument('--max_iter', '-m', action='store', required=False, default=5, type=int)

    parser.add_argument('--rcnn_reg_loss', '-b', action='store', required=False, default='dummy_loss', help='RCNN box loss function name')
    parser.add_argument('--rcnn_cls_loss', '-t', action='store', required=False, default='dummy_loss', help='RCNN type loss function name')
    # parser.add_argument('--rec_cls_loss', '-c', action='store', required=False, default='dummy_loss', help='Recognizer cls loss function name')
    # parser.add_argument('--rpn_reg_loss', '-rb', action='store', required=False, default='dummy_loss', help='RPN box loss function name')
    # parser.add_argument('--rpn_cls_loss', '-o', action='store', required=False, default='dummy_loss', help='RPN layer objectiveness loss function name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = handle_args()

    pl = load_pipeline(device)
    ds = get_dataset(args.dataset, device=device)

    loss_fns = {
        'rcnn_reg_loss': locals()[args.rcnn_reg_loss],
        'rcnn_cls_loss': locals()[args.rcnn_cls_loss]
    }

    def objective_fn(data_item, output):
        boxes = torch.tensor(data_item['boxes'], device=device)
        colors = data_item['colors']
        inferred_tl_types = data_item['inferred_tl_types']
        assert 'NA' not in inferred_tl_types
        return objective(boxes, convert_labels_to_nums(colors, COLOR_LABELS), convert_labels_to_nums(inferred_tl_types, TL_TYPES), output, loss_fns)

    topk_filename = args.topk_file
    if topk_filename is None:
        topk_filename = f'data/evaluation/{args.dataset}_top200.bin'

    adv_imgs = []
    t1 = time.perf_counter()
    for i, idx in enumerate(load_topk_idxs(topk_filename)):
        data_item = ds[idx]
        adv_img = adversarial.adversarial(pl, data_item, objective_fn, step_size=args.step_size, eps=args.eps, budget=args.max_iter)
        adv_imgs.append(adv_img.type(torch.uint8).cpu())
    t2 = time.perf_counter()
    print(f"Run all the cases in {t2-t1:0.4f} seconds!")
    adv_imgs = torch.stack(adv_imgs)

    if args.path is not None:
        path = args.path
    else:
        path = f'data/adversarial_results/{args.dataset}/target/'
    filename = f'{path}/{args.rcnn_reg_loss}_{args.rcnn_cls_loss}_{args.eps}_{args.step_size}_{args.max_iter}.bin'

    if not os.path.exists(path):
        os.makedirs(path)
    with open(filename, 'wb') as f:
        pickle.dump(adv_imgs, f)
