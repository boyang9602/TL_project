from tools.utils import IoU_multi
import torch
import torch.nn.functional as F

def hybrid_objective(boxes, colors, inferred_tl_types, output, loss_fns):
    """
    A designed objective. 
    It will apply the loss function on the detections having intersection with the ground truths
    Plus, it will also handle the cases not having intersection with the ground truths
    """
    valid_detections, recognitions, assignments, invalid_detections, rpn_attack_data = output

    score = 0

    # process valid_detections
    valid_ious = IoU_multi(valid_detections[:, 1:5], boxes)
    maxes = torch.max(valid_ious, 1)
    for det_idx, (gt_idx, iou) in enumerate(zip(maxes.indices, maxes.values)):
        if iou <= 1e-4:
            # FP, increase
            pass
        else:
            # correct location, discourage it
            score += loss_fns['rcnn_reg_loss'](boxes[gt_idx], valid_detections[det_idx][1:5], iou)
            score += loss_fns['rcnn_cls_loss'](inferred_tl_types[gt_idx], valid_detections[det_idx][5:])
            score += loss_fns['rec_cls_loss'](colors[gt_idx], recognitions[det_idx])

    # process invalid_detections
    invalid_ious = IoU_multi(invalid_detections[:, 1:5], boxes)
    maxes = torch.max(invalid_ious, 1)
    for det_idx, (gt_idx, iou) in enumerate(zip(maxes.indices, maxes.values)):
        if iou <= 1e-4:
            # box location wrong, try to 
            # 1. make its type score increase, to have a higher chance to be kept and selected
            # 2. fix its location
            pass
        else:
            score += loss_fns['rcnn_reg_loss'](boxes[gt_idx], invalid_detections[det_idx][1:5], iou)
            score += loss_fns['rcnn_cls_loss'](inferred_tl_types[gt_idx], invalid_detections[det_idx][5:])

    # process RPN layer intermediate data
    rpn_ious = IoU_multi(invalid_detections[:, 1:5], boxes)
    maxes = torch.max(rpn_ious, 1)
    for det_idx, (gt_idx, iou) in enumerate(zip(maxes.indices, maxes.values)):
        if iou <= 1e-4:
            # box location wrong, try to 
            # 1. make its objectiveness score increase, to have a higher chance to be kept and selected
            # 2. fix its location
            pass
        else:
            score += loss_fns['rpn_reg_loss'](boxes[gt_idx], rpn_attack_data[det_idx][1:5], iou)
            score += loss_fns['rpn_cls_loss'](1, rpn_attack_data[det_idx][5:])

    return score