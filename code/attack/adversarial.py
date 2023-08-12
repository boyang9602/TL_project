import torch
import torch.nn.functional as F
from tools.utils import IoU_multi, IoG_single

def process_adv_img(img, adv_img, eps):
    """project the perturbations, not exceeds -eps and eps, new img not exceed 0-255"""
    adv_img = torch.clamp(adv_img.detach().clone(), 0, 255)
    perturbation = adv_img - img
    perturbation = torch.clamp(perturbation, -eps, eps)
    adv_img = img + perturbation
    return adv_img

def naive_tltype_identifier(gt_box):
    if gt_box[3] - gt_box[1] > gt_box[2] - gt_box[0]:
        # height > width, vertical
        return 1
    else:
        # else horizontal
        return 3

def adversarial(model, data_item, objective_fn, step_size=3, eps=16, budget=5, device=None):
    """
    This is the overall framework
    1. init the perturbation
    2. compute the loss
    3. compute the gradients
    4. update the perturbation
    5. stop and return

    model is the model
    data_item is an item (dict) from the dataset defined in this project, it includes the image and the ground truth labels.
    objective_fn is the objective/loss function, which takes 2 parameters, original picture's output and adversarial picture's output
    """
    image = data_item['image']
    boxes = data_item['boxes']
    colors = data_item['colors']

    # init adv_img
    adv_img = image.detach().clone() + torch.empty_like(image.type(torch.float)).uniform_(-eps, eps)
    # clamp if the pixel value is out of range
    adv_img = torch.clamp(adv_img.detach().clone(), 0, 255).requires_grad_()

    iter_num = 0
    while iter_num < budget:
        optimizer = torch.optim.Adam([adv_img], lr=step_size)
        optimizer.zero_grad()
        output  = model(adv_img, boxes)

        score = objective_fn(data_item, output)

        if type(score) == int and score == 0:
            return process_adv_img(adv_img.detach().clone(), adv_img.detach().clone(), eps)

        score.backward()
        optimizer.step()
        iter_num += 1
        adv_img = process_adv_img(adv_img.detach().clone(), adv_img.detach().clone(), eps).requires_grad_()
    return adv_img

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
