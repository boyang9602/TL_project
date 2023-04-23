import torch
import torch.nn.functional as F
from utils import IoU_multi
device = "cuda" if torch.cuda.is_available() else "cpu"

def adversarial(model, input, boxes, objective_fn, step_size, eps, log=False):
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
        output, assignments1 = model(input, boxes)
        if len(output) == 0:
            return None
    # print('\n\nStart Attacking...')
    it = 0
    while True:
        optimizer = torch.optim.Adam([adv_img], lr=step_size)
        optimizer.zero_grad()
        output2, assignments2 = model(adv_img, boxes)
        score = objective_fn(output, output2, it)
        if score == None:
            return process_adv_img(input.detach().clone(), adv_img.detach().clone(), eps)
        score.backward()
        if log:
            print(adv_img.grad, adv_img)
        optimizer.step()
        it += 1
        adv_img = process_adv_img(input.detach().clone(), adv_img.detach().clone(), eps).requires_grad_()

def objective(gts, dts, it):
    # print(f'$$$ Step {it} details: $$$')
    if len(dts) == 0:
        # print('No detections!')
        # print(f'### Step {it} ends! ###\n')
        return None
    score = 0
    iouss = IoU_multi(dts[:, 1:5], gts[:, 1:5])
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
        dt = dts[i]
        gt = gts[idx]

        gclass = torch.argmax(gt[5:9])
        dclass = torch.argmax(dt[5:9])
        gcolor = torch.argmax(gt[9:])
        dcolor = torch.argmax(dt[9:])
        type_loss = F.nll_loss(dt[5:9].unsqueeze(0), torch.tensor([gclass], device=device))
        color_loss = F.nll_loss(dt[9:].unsqueeze(0), torch.tensor([gcolor], device=device))
        box_loss = F.smooth_l1_loss(dt[1:5], gt[1:5])
        # print(f'\tiou: {iou}, box_loss: {box_loss}\n\tgclass: {gclass}, dclass: {dclass}, type_loss: {type_loss}\n\tgcolor: {gcolor}, dcolor: {dcolor}, color_loss: {color_loss}')
        score -= color_loss
        score -= type_loss
        score -= box_loss
    if score == 0: # no intersection, perfect attack in terms of IoU
        # print('No intersections!')
        # print(f'### Step {it} ends! ###\n')
        return None
    # print(f'### Step {it} ends! ###\n')
    return score

def process_adv_img(img, adv_img, eps):
    """project the perturbations, not exceeds -eps and eps, new img not exceed 0-255"""
    adv_img = torch.clamp(adv_img.detach().clone(), 0, 255)
    perturbation = adv_img - img
    perturbation = torch.clamp(perturbation, -eps, eps)
    adv_img = img + perturbation
    return adv_img

def create_objective(objective, max_it):
    def objective_w_max_it(orig, adv, it):
        if it >= max_it:
            return None
        return objective(orig, adv, it)
    return objective_w_max_it