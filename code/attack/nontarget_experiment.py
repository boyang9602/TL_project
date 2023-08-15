import os
import time
import pickle
import argparse
import torch
import torch.nn.functional as F
from tools.utils import IoU_multi, IoG_single, load_topk_idxs, convert_labels_to_nums
from models.pipeline4attack import load_pipeline
from tools.dataset import get_dataset
import attack.adversarial as adversarial

TL_TYPES = ['UNK', 'VERT', 'QUAD', 'HORI']
COLOR_LABELS = ["off", "red", "yellow", "green"]

def objective(boxes, colors, inferred_tl_types, output, loss_fns):
    """
    Basic objective. 
    It will apply the loss function on the detections having intersection with the ground truths
    """
    valid_detections, recognitions, assignments, invalid_detections, rpn_data, rcnn_boxes, rcnn_scores = output

    score = 0
    # process valid_detections
    valid_ious = IoU_multi(valid_detections[:, 1:5], boxes)
    maxes = torch.max(valid_ious, 1)
    for det_idx, (gt_idx, iou) in enumerate(zip(maxes.indices, maxes.values)):
        if iou <= 1e-4:
            # no intersection, pass
            continue
        score += loss_fns['rcnn_reg_loss'](boxes[gt_idx], valid_detections[det_idx][1:5], iou)
        score += loss_fns['rcnn_cls_loss'](inferred_tl_types[gt_idx], valid_detections[det_idx][5:])
        score += loss_fns['rec_cls_loss'](colors[gt_idx], recognitions[det_idx])

    # process invalid_detections
    invalid_ious = IoU_multi(invalid_detections[:, 1:5], boxes)
    maxes = torch.max(invalid_ious, 1)
    for det_idx, (gt_idx, iou) in enumerate(zip(maxes.indices, maxes.values)):
        if iou <= 1e-4:
            # no intersection, pass
            continue
        score += loss_fns['rcnn_reg_loss'](boxes[gt_idx], invalid_detections[det_idx][1:5], iou)
        score += loss_fns['rcnn_cls_loss'](inferred_tl_types[gt_idx], invalid_detections[det_idx][5:])

    # process RPN layer intermediate data
    rpn_ious = IoU_multi(rpn_data[:, 1:5], boxes)
    maxes = torch.max(rpn_ious, 1)
    for det_idx, (gt_idx, iou) in enumerate(zip(maxes.indices, maxes.values)):
        if iou <= 1e-4:
            # no intersection, pass
            continue
        score += loss_fns['rpn_reg_loss'](boxes[gt_idx], rpn_data[det_idx][1:5], iou)
        score += loss_fns['rpn_cls_loss'](1, rpn_data[det_idx][5:])

    return score

def naive_tltype_identifier(gt_box):
    if gt_box[3] - gt_box[1] > gt_box[2] - gt_box[0]:
        # height > width, vertical
        return 1
    else:
        # else horizontal
        return 3

def dummy_loss(*args):
    return 0

def box_smooth_l1_loss(gt_box, det_box, iou):
    # try to make the negative loss of the det_box and gt_box smaller
    # so that the loss will be bigger, which means det_box and gt_box become 
    # much more different
    return -F.smooth_l1_loss(gt_box, det_box)

def box_iou_loss(gt_box, det_box, iou):
    # try to make the iou smaller
    return iou

def box_iog_loss(gt_box, det_box, iou):
    # try to make the iog smaller
    return IoG_single(gt_box, det_box)

def cls_nll_loss(gt_idx, scores_vec):
    return -F.nll_loss(scores_vec.unsqueeze(0), torch.tensor([gt_idx], device=scores_vec.device))

def cls_gt_score_loss(gt_idx, scores_vec):
    return scores_vec[gt_idx]

def handle_args():
    parser = argparse.ArgumentParser(description='Run the non target attacks')
    parser.add_argument('--dataset', '-ds', action='store', required=True, help='the dataset.')
    parser.add_argument('--topk_file', '-k', action='store', required=False, default=None, help='the selected perfect cases.')
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
    return args

if __name__ == '__main__':
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = handle_args()

    pl = load_pipeline(device)
    ds = get_dataset(args.dataset, device=device)

    loss_fns = {
        'rcnn_reg_loss': locals()[args.rcnn_reg_loss],
        'rcnn_cls_loss': locals()[args.rcnn_cls_loss],
        'rec_cls_loss': locals()[args.rec_cls_loss],
        'rpn_reg_loss': locals()[args.rpn_reg_loss],
        'rpn_cls_loss': locals()[args.rpn_cls_loss]
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
        path = f'data/adversarial_results/{args.dataset}/nontarget/'
    filename = f'{path}/{args.rcnn_reg_loss}_{args.rcnn_cls_loss}_{args.rec_cls_loss}_{args.rpn_reg_loss}_{args.rpn_cls_loss}_{args.eps}_{args.step_size}_{args.max_iter}.bin'

    if not os.path.exists(path):
        os.makedirs(path)
    with open(filename, 'wb') as f:
        pickle.dump(adv_imgs, f)
