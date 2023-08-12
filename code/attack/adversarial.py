import torch

def process_adv_img(img, adv_img, eps):
    """project the perturbations, not exceeds -eps and eps, new img not exceed 0-255"""
    adv_img = torch.clamp(adv_img.detach().clone(), 0, 255)
    perturbation = adv_img - img
    perturbation = torch.clamp(perturbation, -eps, eps)
    adv_img = img + perturbation
    return adv_img

def adversarial(model, data_item, objective_fn, step_size=3, eps=16, budget=5):
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
