from tools.utils import IoU_multi
import torch
import torch.nn.functional as F

def target_objective(boxes, colors, inferred_tl_types, output, loss_fns):
    """
    Target objective. It will find all the detected boxes which do not have intersection with the ground truth in RPN and RCNN layer. 
    For these boxes, it will try to increase their confidence scores.
    """
    valid_detections, recognitions, assignments, invalid_detections, rpn_attack_data = output

    score = 0

    rcnn_target_boxes = []
    rpn_target_boxes = []

    # process valid_detections
    valid_ious = IoU_multi(valid_detections[:, 1:5], boxes)
    maxes = torch.max(valid_ious, 1)

    for det_idx, (gt_idx, iou) in enumerate(zip(maxes.indices, maxes.values)):
        if iou <= 1e-4:
            # FP, try to improve its confidence score
            rcnn_target_boxes.append(valid_detections[det_idx])
        else:
            pass

    # process invalid_detections
    invalid_ious = IoU_multi(invalid_detections[:, 1:5], boxes)
    maxes = torch.max(invalid_ious, 1)
    for det_idx, (gt_idx, iou) in enumerate(zip(maxes.indices, maxes.values)):
        if iou <= 1e-4:
            # FP, try to improve its confidence score
            rcnn_target_boxes.append(invalid_detections[det_idx])
        else:
            pass

    # process RPN layer intermediate data
    rpn_ious = IoU_multi(rpn_attack_data[:, 1:5], boxes)
    maxes = torch.max(rpn_ious, 1)
    for det_idx, (gt_idx, iou) in enumerate(zip(maxes.indices, maxes.values)):
        if iou <= 1e-4:
            # FP, try to improve its confidence score
            rpn_target_boxes.append(rpn_attack_data[det_idx])
            pass
        else:
            pass
    
    # for the non-overlapped boxes, increase their confidence scores, and try to keep the box position
    for item in rcnn_target_boxes:
        box = item[1:5]
        type_scores = item[5:]
        tl_type = torch.argmax(type_scores)
        score += F.nll_loss(type_scores.unsqueeze(0), tl_type.reshape(1))
        score += F.smooth_l1_loss(box, box)
    # try to increase the objectiveness score and keep the box position
    for item in rpn_target_boxes:
        box = item[1:5]
        type_scores = item[5:]
        score += F.nll_loss(type_scores.unsqueeze(0), torch.tensor([1], type_scores.device))
        score += F.smooth_l1_loss(box, box)

    return score

def target_objective2(boxes, colors, inferred_tl_types, output, loss_fns):
    """
    Target objective. 
    For the detected boxes which do not have intersection with the ground truth in RPN and RCNN layer, it will try to increase their confidence scores. 
    For the detected boxes which have intersection with the ground truth, 
        if the type is correct, pass
        else try to increase its confidence score
    """
    valid_detections, recognitions, assignments, invalid_detections, rpn_attack_data = output

    score = 0

    rcnn_target_boxes = []
    rpn_target_boxes = []

    # process valid_detections
    valid_ious = IoU_multi(valid_detections[:, 1:5], boxes)
    maxes = torch.max(valid_ious, 1)

    for det_idx, (gt_idx, iou) in enumerate(zip(maxes.indices, maxes.values)):
        if iou <= 1e-4:
            # FP, try to improve its confidence score
            rcnn_target_boxes.append(valid_detections[det_idx])
        else:
            pass

    # process invalid_detections
    invalid_ious = IoU_multi(invalid_detections[:, 1:5], boxes)
    maxes = torch.max(invalid_ious, 1)
    for det_idx, (gt_idx, iou) in enumerate(zip(maxes.indices, maxes.values)):
        if iou <= 1e-4:
            # FP, try to improve its confidence score
            rcnn_target_boxes.append(invalid_detections[det_idx])
        else:
            pass

    # process RPN layer intermediate data
    rpn_ious = IoU_multi(rpn_attack_data[:, 1:5], boxes)
    maxes = torch.max(rpn_ious, 1)
    for det_idx, (gt_idx, iou) in enumerate(zip(maxes.indices, maxes.values)):
        if iou <= 1e-4:
            # FP, try to improve its confidence score
            rpn_target_boxes.append(rpn_attack_data[det_idx])
            pass
        else:
            pass
    
    # for the non-overlapped boxes, increase their confidence scores, and try to keep the box position
    for item in rcnn_target_boxes:
        box = item[1:5]
        type_scores = item[5:]
        tl_type = torch.argmax(type_scores)
        score += F.nll_loss(type_scores.unsqueeze(0), tl_type.reshape(1))
        score += F.smooth_l1_loss(box, box)
    # try to increase the objectiveness score and keep the box position
    for item in rpn_target_boxes:
        box = item[1:5]
        type_scores = item[5:]
        score += F.nll_loss(type_scores.unsqueeze(0), torch.tensor([1], type_scores.device))
        score += F.smooth_l1_loss(box, box)

    return score
