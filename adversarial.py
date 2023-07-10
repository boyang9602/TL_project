import torch
import torch.nn.functional as F
from utils import IoU_multi
device = "cuda" if torch.cuda.is_available() else "cpu"

# def adversarial(model, input, boxes, objective_fn, step_size, eps, budget, log=False):
#     """
#     :param nn.Module model: the model to consume the input
#     :param torch.Tensor input: the input image, in the format of (h, w, c), channel c should be bgr order. It should **not** subtract the means
#     :param function objective_fn: the objective function, which accepts 3 params, original output of the model given input, the output of the model given input + perturbation, and the current iteration number. It should return None when it is done. 
#     :param int step_size: step_size/learning_rate
#     :param int eps: The max perturbation per pixel
#     :param bool log: if show the logs for the gradients and the image after gradient descent 
#     """
#     # init adv_img
#     adv_img = input.detach().clone() + torch.empty_like(input.type(torch.float)).uniform_(-eps, eps)
#     # clamp if the pixel value is out of range
#     adv_img = torch.clamp(adv_img.detach().clone(), 0, 255).requires_grad_()
    
#     # get the output of the orig input
#     with torch.no_grad():
#         # output, assignments1 = model(input, boxes)
#         output = model(input, boxes)
#         if len(output) == 0:
#             return None
#     # print('\n\nStart Attacking...')
#     it = 0
#     while it < budget:
#         optimizer = torch.optim.Adam([adv_img], lr=step_size)
#         optimizer.zero_grad()
#         output2 = model(adv_img, boxes)
#         score = objective_fn(output, output2)
#         if score == None:
#             return process_adv_img(input.detach().clone(), adv_img.detach().clone(), eps)
#         score.backward()
#         if log:
#             print(adv_img.grad, adv_img)
#         optimizer.step()
#         it += 1
#         adv_img = process_adv_img(input.detach().clone(), adv_img.detach().clone(), eps).requires_grad_()
#     return adv_img

# def objective(ground_truths, detections):
#     # print(f'$$$ Step {it} details: $$$')
#     gt_valid_detections, gt_recognitions, gt_assignments, gt_invalid_detections = ground_truths
#     dt_valid_detections, dt_recognitions, dt_assignments, dt_invalid_detections = detections
#     if len(dt_valid_detections) == 0:
#         # print('No detections!')
#         # print(f'### Step {it} ends! ###\n')
#         return None
#     score = 0

#     # process valid detections
#     iouss = IoU_multi(dt_valid_detections[:, 1:5], gt_valid_detections[:, 1:5])
#     maxes = torch.max(iouss, 1)
#     values = maxes.values
#     indices = maxes.indices
#     for i, idx in enumerate(indices):
#         # this dt has the max iou value with this gt
#         iou = iouss[i, idx]

#         # almost no intersection, so we do not care this detection
#         if iou <= 0.0001:
#             # print(f'detection {i} has no intersection with any ground truth, pass')
#             continue
#         # print(f'detection {i}:')
#         # otherwise, we are going to make the different
#         dt_location = dt_valid_detections[i]
#         gt_location = gt_valid_detections[idx]
#         dt_recognition = dt_recognitions[i]
#         gt_recognition = gt_recognitions[idx]

#         gclass = torch.argmax(gt_valid_detection[5:9])
#         dclass = torch.argmax(dt_valid_detection[5:9])
#         gcolor = torch.argmax(gt_recognition)
#         dcolor = torch.argmax(dt_recognition)
#         type_loss = F.nll_loss(dt_location[5:9].unsqueeze(0), torch.tensor([gclass], device=device))
#         color_loss = F.nll_loss(dt_recognition[9:].unsqueeze(0), torch.tensor([gcolor], device=device))
#         box_loss = F.smooth_l1_loss(dt_location[1:5], gt_location[1:5])
#         # print(f'\tiou: {iou}, box_loss: {box_loss}\n\tgclass: {gclass}, dclass: {dclass}, type_loss: {type_loss}\n\tgcolor: {gcolor}, dcolor: {dcolor}, color_loss: {color_loss}')
        
#         # we penalize the correct color
#         score -= color_loss
#         # we penalize the correct type
#         score -= type_loss
#         # we penalize the correct location
#         score -= box_loss

#     # process invalid detections
#     iouss = IoU_multi(dt_invalid_detections[:, 1:5], gt_valid_detections[:, 1:5])
#     maxes = torch.max(iouss, 1)
#     values = maxes.values
#     indices = maxes.indices
#     for i, idx in enumerate(indices):
#         # this dt has the max iou value with this gt
#         iou = iouss[i, idx]

#         # almost no intersection, so we do not care this detection
#         if iou <= 0.0001:
#             # print(f'detection {i} has no intersection with any ground truth, pass')
#             continue
#         # print(f'detection {i}:')
#         # otherwise, we are going to make the different
#         dt_location = dt_invalid_detections[i]
#         gt_location = gt_invalid_detections[idx]

#         gclass = torch.argmax(gt_valid_detection[5:9])
#         dclass = torch.argmax(dt_valid_detection[5:9])
#         type_loss = F.nll_loss(dt_location[5:9].unsqueeze(0), torch.tensor([gclass], device=device))
#         box_loss = F.smooth_l1_loss(dt_location[1:5], gt_location[1:5])
#         # print(f'\tiou: {iou}, box_loss: {box_loss}\n\tgclass: {gclass}, dclass: {dclass}, type_loss: {type_loss}\n\tgcolor: {gcolor}, dcolor: {dcolor}, color_loss: {color_loss}')

#         # we reward the wrong type
#         score = type_loss
#         # we penalize the correct location
#         score -= box_loss
#     if score == 0: # no intersection, perfect attack in terms of IoU
#         # print('No intersections!')
#         # print(f'### Step {it} ends! ###\n')
#         return None
#     # print(f'### Step {it} ends! ###\n')
#     return score

def process_adv_img(img, adv_img, eps):
    """project the perturbations, not exceeds -eps and eps, new img not exceed 0-255"""
    adv_img = torch.clamp(adv_img.detach().clone(), 0, 255)
    perturbation = adv_img - img
    perturbation = torch.clamp(perturbation, -eps, eps)
    adv_img = img + perturbation
    return adv_img

def objective_rcnn(ground_truths, detections, valid_scores, invalid_scores):
    # print(f'$$$ Step {it} details: $$$')
    gt_valid_detections, gt_recognitions, gt_assignments, gt_invalid_detections = ground_truths
    dt_valid_detections, dt_recognitions, dt_assignments, dt_invalid_detections = detections
    if len(dt_valid_detections) == 0:
        # print('No detections!')
        # print(f'### Step {it} ends! ###\n')
        return None
    score = 0

    # process valid detections
    iouss = IoU_multi(dt_valid_detections[:, 1:5], gt_valid_detections[:, 1:5])
    maxes = torch.max(iouss, 1)
    values = maxes.values
    indices = maxes.indices
    for i, idx in enumerate(indices):
        # this dt has the max iou value with this gt
        iou = iouss[i, idx]

        # almost no intersection, so we do not care this detection
        if iou <= 0.0001:
            # print(f'detection {i} has no intersection with any ground truth, pass')
            continue
        # print(f'detection {i}:')
        # otherwise, we are going to make the different
        dt_location = dt_valid_detections[i]
        gt_location = gt_valid_detections[idx]
        dt_recognition = dt_recognitions[i]
        gt_recognition = gt_recognitions[idx]

        gclass = torch.argmax(gt_location[5:9])
        dclass = torch.argmax(dt_location[5:9])
        gcolor = torch.argmax(gt_recognition)
        dcolor = torch.argmax(dt_recognition)
        type_loss = F.nll_loss(dt_location[5:9].unsqueeze(0), torch.tensor([gclass], device=device))
        color_loss = F.nll_loss(dt_recognition.unsqueeze(0), torch.tensor([gcolor], device=device))
        box_loss = F.smooth_l1_loss(dt_location[1:5], gt_location[1:5])
        # print(f'\tiou: {iou}, box_loss: {box_loss}\n\tgclass: {gclass}, dclass: {dclass}, type_loss: {type_loss}\n\tgcolor: {gcolor}, dcolor: {dcolor}, color_loss: {color_loss}')

        # # we penalize the correct color
        # score -= color_loss
        # # we penalize the correct type
        # score -= type_loss
        # # we penalize the correct location
        # score -= box_loss
        score += valid_scores(type_loss, color_loss, box_loss)

    # process invalid detections
    iouss = IoU_multi(dt_invalid_detections[:, 1:5], gt_valid_detections[:, 1:5])
    maxes = torch.max(iouss, 1)
    values = maxes.values
    indices = maxes.indices
    for i, idx in enumerate(indices):
        # this dt has the max iou value with this gt
        iou = iouss[i, idx]

        # almost no intersection, so we do not care this detection
        if iou <= 0.0001:
            # print(f'detection {i} has no intersection with any ground truth, pass')
            continue
        # print(f'detection {i}:')
        # otherwise, we are going to make the different
        dt_location = dt_invalid_detections[i]
        gt_location = gt_valid_detections[idx]

        gclass = torch.argmax(gt_location[5:9])
        dclass = torch.argmax(dt_location[5:9])
        type_loss = F.nll_loss(dt_location[5:9].unsqueeze(0), torch.tensor([gclass], device=device))
        box_loss = F.smooth_l1_loss(dt_location[1:5], gt_location[1:5])
        # print(f'\tiou: {iou}, box_loss: {box_loss}\n\tgclass: {gclass}, dclass: {dclass}, type_loss: {type_loss}\n\tgcolor: {gcolor}, dcolor: {dcolor}, color_loss: {color_loss}')

        # # we reward the wrong type
        # score = type_loss
        # # we penalize the correct location
        # score -= box_loss
        score += invalid_scores(type_loss, box_loss)
    if score == 0: # no intersection, perfect attack in terms of IoU
        # print('No intersections!')
        # print(f'### Step {it} ends! ###\n')
        return None
    # print(f'### Step {it} ends! ###\n')
    return score

def objective_rpn(ground_truths, rpn_attack_data, valid_scores):
    # print(f'$$$ Step {it} details: $$$')
    gt_valid_detections, gt_recognitions, gt_assignments, gt_invalid_detections = ground_truths
    if len(rpn_attack_data) == 0:
        # print('No detections!')
        # print(f'### Step {it} ends! ###\n')
        return None
    score = 0

    iouss = IoU_multi(rpn_attack_data[:, 1:5], gt_valid_detections[:, 1:5])
    maxes = torch.max(iouss, 1)
    values = maxes.values
    indices = maxes.indices
    for i, idx in enumerate(indices):
        # this dt has the max iou value with this gt
        iou = iouss[i, idx]

        # almost no intersection, so we do not care this detection
        if iou <= 0.0001:
            # print(f'detection {i} has no intersection with any ground truth, pass')
            continue
        # print(f'detection {i}:')
        # otherwise, we are going to make the different
        dt_location = rpn_attack_data[i]
        gt_location = gt_valid_detections[idx]

        objectiveness_loss = F.nll_loss(dt_location[5:].unsqueeze(0), torch.tensor([1], device=device))
        box_loss = F.smooth_l1_loss(dt_location[1:5], gt_location[1:5])
        # print(f'\tiou: {iou}, box_loss: {box_loss}\n\tgclass: {gclass}, dclass: {dclass}, type_loss: {type_loss}\n\tgcolor: {gcolor}, dcolor: {dcolor}, color_loss: {color_loss}')

        score += valid_scores(objectiveness_loss, box_loss)

    if score == 0: # no intersection, perfect attack in terms of IoU
        # print('No intersections!')
        # print(f'### Step {it} ends! ###\n')
        return None
    # print(f'### Step {it} ends! ###\n')
    return score

def objective_target(ground_truths, detections, valid_scores, invalid_scores, target_label_fn, target_box_fn):
    # print(f'$$$ Step {it} details: $$$')
    gt_valid_detections, gt_recognitions, gt_assignments, gt_invalid_detections = ground_truths
    dt_valid_detections, dt_recognitions, dt_assignments, dt_invalid_detections = detections
    if len(dt_valid_detections) == 0:
        # print('No detections!')
        # print(f'### Step {it} ends! ###\n')
        return None
    score = 0

    # process valid detections
    iouss = IoU_multi(dt_valid_detections[:, 1:5], gt_valid_detections[:, 1:5])
    maxes = torch.max(iouss, 1)
    values = maxes.values
    indices = maxes.indices
    for i, idx in enumerate(indices):
        # this dt has the max iou value with this gt
        iou = iouss[i, idx]

        # almost no intersection, so we do not care this detection
        if iou <= 0.0001:
            # print(f'detection {i} has no intersection with any ground truth, pass')
            continue
        # print(f'detection {i}:')
        # otherwise, we are going to make the different
        dt_location = dt_valid_detections[i]
        gt_location = gt_valid_detections[idx]
        dt_recognition = dt_recognitions[i]
        gt_recognition = gt_recognitions[idx]

        gclass = torch.argmax(gt_location[5:9])
        dclass = torch.argmax(dt_location[5:9])
        gcolor = torch.argmax(gt_recognition)
        dcolor = torch.argmax(dt_recognition)
        target_class = target_label_fn(gt_location[5:9])
        target_color = target_label_fn(gt_recognition)
        target_box = target_box_fn(gt_location[1:5])
        type_loss = F.nll_loss(dt_location[5:9].unsqueeze(0), torch.tensor([target_class], device=device))
        color_loss = F.nll_loss(dt_recognition.unsqueeze(0), torch.tensor([target_color], device=device))
        box_loss = F.smooth_l1_loss(dt_location[1:5], target_box)
        # print(f'\tiou: {iou}, box_loss: {box_loss}\n\tgclass: {gclass}, dclass: {dclass}, type_loss: {type_loss}\n\tgcolor: {gcolor}, dcolor: {dcolor}, color_loss: {color_loss}')

        # # we penalize the correct color
        # score -= color_loss
        # # we penalize the correct type
        # score -= type_loss
        # # we penalize the correct location
        # score -= box_loss
        score += valid_scores(type_loss, color_loss, box_loss)

    # process invalid detections
    iouss = IoU_multi(dt_invalid_detections[:, 1:5], gt_valid_detections[:, 1:5])
    maxes = torch.max(iouss, 1)
    values = maxes.values
    indices = maxes.indices
    for i, idx in enumerate(indices):
        # this dt has the max iou value with this gt
        iou = iouss[i, idx]

        # almost no intersection, so we do not care this detection
        if iou <= 0.0001:
            # print(f'detection {i} has no intersection with any ground truth, pass')
            continue
        # print(f'detection {i}:')
        # otherwise, we are going to make the different
        dt_location = dt_invalid_detections[i]
        gt_location = gt_valid_detections[idx]
        gclass = torch.argmax(gt_location[5:9])
        dclass = torch.argmax(dt_location[5:9])
        target_class = target_label_fn(gt_location[5:9])
        target_box = target_box_fn(gt_location[1:5])

        type_loss = F.nll_loss(dt_location[5:9].unsqueeze(0), torch.tensor([target_class], device=device))
        box_loss = F.smooth_l1_loss(dt_location[1:5], target_box)
        # print(f'\tiou: {iou}, box_loss: {box_loss}\n\tgclass: {gclass}, dclass: {dclass}, type_loss: {type_loss}\n\tgcolor: {gcolor}, dcolor: {dcolor}, color_loss: {color_loss}')

        # # we reward the wrong type
        # score = type_loss
        # # we penalize the correct location
        # score -= box_loss
        score += invalid_scores(type_loss, box_loss)
    if score == 0: # no intersection, perfect attack in terms of IoU
        # print('No intersections!')
        # print(f'### Step {it} ends! ###\n')
        return None
    # print(f'### Step {it} ends! ###\n')
    return score

def adversarial(model, input, boxes, objective_fn, step_size, eps, budget, log=False):
    """
    :param nn.Module model: the model to consume the input
    :param torch.Tensor input: the input image, in the format of (h, w, c), channel c should be bgr order. It should **not** subtract the means
    :param function objective_fn: the objective function, which accepts 3 params, original output of the model given input, the output of the model given input + perturbation, and the current iteration number. It should return None when it is done.
    :param int step_size: step_size/learning_rate
    :param int eps: The max perturbation per pixel
    :param bool log: if show the logs for the gradients and the image after gradient descent
    """
    # init adv_img
    adv_img = input.detach().clone() + torch.empty_like(input.type(torch.float)).uniform_(-eps, eps)
    # clamp if the pixel value is out of range
    adv_img = torch.clamp(adv_img.detach().clone(), 0, 255).requires_grad_()

    # get the output of the orig input
    with torch.no_grad():
        # output, assignments1 = model(input, boxes)
        output = model(input, boxes)
        if len(output) == 0:
            return None
    # print('\n\nStart Attacking...')
    it = 0
    while it < budget:
        optimizer = torch.optim.Adam([adv_img], lr=step_size)
        optimizer.zero_grad()
        output2 = model(adv_img, boxes)
        # score = objective_fn(output, model.rpn_attack_data) # tech debt, todo: refactor
        # score = objective_fn(output, output2) # tech debt, todo: refactor
        score = objective_fn(output, model.rpn_attack_data, output2)
        if score == None:
            return process_adv_img(input.detach().clone(), adv_img.detach().clone(), eps)
        score.backward()
        if log:
            print(adv_img.grad, adv_img)
        optimizer.step()
        it += 1
        adv_img = process_adv_img(input.detach().clone(), adv_img.detach().clone(), eps).requires_grad_()
    # print(f'used {it} steps')
    return adv_img

# def create_objective(objective, valid_scores, invalid_scores):
#     def objective_fn(ground_truths, detections):
#         return objective(ground_truths, detections, valid_scores, invalid_scores)
#     return objective_fn

# def create_objective_target(objective, valid_scores, invalid_scores, target_label_fn, target_box_fn):
#     def objective_fn(ground_truths, detections):
#         return objective(ground_truths, detections, valid_scores, invalid_scores, target_label_fn, target_box_fn)
#     return objective_fn

def nontarget_objective_fn_rcnn_cls(rcnn_data_ori, rpn_data_det, rcnn_data_det):
    def valid_scores_rcnn(cls_loss, reg_loss, cnn_loss):
        return -cls_loss
    def invalid_scores(cls_loss, reg_loss):
        return -cls_loss
    return objective_rcnn(rcnn_data_ori, rcnn_data_det, valid_scores_rcnn, invalid_scores)

def nontarget_objective_fn_rcnn_reg(rcnn_data_ori, rpn_data_det, rcnn_data_det):
    def valid_scores_rcnn(cls_loss, reg_loss, cnn_loss):
        return -reg_loss
    def invalid_scores(cls_loss, reg_loss):
        return -reg_loss
    return objective_rcnn(rcnn_data_ori, rcnn_data_det, valid_scores_rcnn, invalid_scores)

def nontarget_objective_fn_rcnn_cls_reg(rcnn_data_ori, rpn_data_det, rcnn_data_det):
    def valid_scores_rcnn(cls_loss, reg_loss, cnn_loss):
        return -reg_loss - cls_loss
    def invalid_scores(cls_loss, reg_loss):
        return -reg_loss - cls_loss
    return objective_rcnn(rcnn_data_ori, rcnn_data_det, valid_scores_rcnn, invalid_scores)

def nontarget_objective_fn_rpn_cls(rcnn_data_ori, rpn_data_det, rcnn_data_det):
    def valid_scores_rpn(cls_loss, reg_loss):
        return -cls_loss
    def invalid_scores(cls_loss, reg_loss):
        return -cls_loss
    return objective_rpn(rcnn_data_ori, rpn_data_det, valid_scores_rpn)

def nontarget_objective_fn_rpn_reg(rcnn_data_ori, rpn_data_det, rcnn_data_det):
    def valid_scores_rpn(cls_loss, reg_loss):
        return -reg_loss
    def invalid_scores(cls_loss, reg_loss):
        return -reg_loss
    return objective_rpn(rcnn_data_ori, rpn_data_det, valid_scores_rpn)

def nontarget_objective_fn_rpn_cls_reg(rcnn_data_ori, rpn_data_det, rcnn_data_det):
    def valid_scores_rpn(cls_loss, reg_loss):
        return -reg_loss - cls_loss
    def invalid_scores(cls_loss, reg_loss):
        return -reg_loss - cls_loss
    return objective_rpn(rcnn_data_ori, rpn_data_det, valid_scores_rpn)

def nontarget_objective_fn_rpn_rcnn_cls(rcnn_data_ori, rpn_data_det, rcnn_data_det):
    def valid_scores_rcnn(cls_loss, reg_loss, cnn_loss):
        return -cls_loss
    def valid_scores_rpn(cls_loss, reg_loss):
        return -cls_loss
    def invalid_scores(cls_loss, reg_loss):
        return -cls_loss
    score_rcnn = objective_rcnn(rcnn_data_ori, rcnn_data_det, valid_scores_rcnn, invalid_scores)
    score_rpn = objective_rpn(rcnn_data_ori, rpn_data_det, valid_scores_rpn)
    if score_rcnn == None:
        print('rcnn none')
        return score_rpn
    elif score_rpn == None:
        print('rpn none')
        return score_rcnn
    else:
        return score_rcnn + score_rpn

def nontarget_objective_fn_rpn_rcnn_reg(rcnn_data_ori, rpn_data_det, rcnn_data_det):
    def valid_scores_rcnn(cls_loss, reg_loss, cnn_loss):
        return -reg_loss
    def valid_scores_rpn(cls_loss, reg_loss):
        return -reg_loss
    def invalid_scores(cls_loss, reg_loss):
        return -reg_loss
    score_rcnn = objective_rcnn(rcnn_data_ori, rcnn_data_det, valid_scores_rcnn, invalid_scores)
    score_rpn = objective_rpn(rcnn_data_ori, rpn_data_det, valid_scores_rpn)
    if score_rcnn == None:
        print('rcnn none')
        return score_rpn
    elif score_rpn == None:
        print('rpn none')
        return score_rcnn
    else:
        return score_rcnn + score_rpn

def nontarget_objective_fn_rpn_rcnn_cls_reg(rcnn_data_ori, rpn_data_det, rcnn_data_det):
    def valid_scores_rcnn(cls_loss, reg_loss, cnn_loss):
        return -reg_loss - cls_loss
    def valid_scores_rpn(cls_loss, reg_loss):
        return -reg_loss - cls_loss
    def invalid_scores(cls_loss, reg_loss):
        return -reg_loss - cls_loss
    score_rcnn = objective_rcnn(rcnn_data_ori, rcnn_data_det, valid_scores_rcnn, invalid_scores)
    score_rpn = objective_rpn(rcnn_data_ori, rpn_data_det, valid_scores_rpn)
    if score_rcnn == None:
        print('rcnn none')
        return score_rpn
    elif score_rpn == None:
        print('rpn none')
        return score_rcnn
    else:
        return score_rcnn + score_rpn

def nontarget_objective_fn_rcnn_cnn_cls(rcnn_data_ori, rpn_data_det, rcnn_data_det):
    def valid_scores_rcnn(cls_loss, reg_loss, cnn_loss):
        return -cls_loss - cnn_loss
    def invalid_scores(cls_loss, reg_loss):
        return -cls_loss
    return objective_rcnn(rcnn_data_ori, rcnn_data_det, valid_scores_rcnn, invalid_scores)

def nontarget_objective_fn_rcnn_cnn_reg(rcnn_data_ori, rpn_data_det, rcnn_data_det):
    def valid_scores_rcnn(cls_loss, reg_loss, cnn_loss):
        return -reg_loss - cnn_loss
    def invalid_scores(cls_loss, reg_loss):
        return -reg_loss
    return objective_rcnn(rcnn_data_ori, rcnn_data_det, valid_scores_rcnn, invalid_scores)

def nontarget_objective_fn_rcnn_cnn_cls_reg(rcnn_data_ori, rpn_data_det, rcnn_data_det):
    def valid_scores_rcnn(cls_loss, reg_loss, cnn_loss):
        return -reg_loss - cls_loss - cnn_loss
    def invalid_scores(cls_loss, reg_loss):
        return -reg_loss - cls_loss
    return objective_rcnn(rcnn_data_ori, rcnn_data_det, valid_scores_rcnn, invalid_scores)
