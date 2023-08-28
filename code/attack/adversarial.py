import torch

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
    adv_img = image + torch.empty_like(image.type(torch.float)).uniform_(-eps, eps)
    # clamp if the pixel value is out of range
    adv_img = torch.clamp(adv_img, 0, 255).requires_grad_()
    iter_num = 0
    while iter_num < budget:
        output  = model(adv_img, boxes)
        score = objective_fn(data_item, output)

        if type(score) == int and score == 0:
            return adv_img.detach()

        grad = torch.autograd.grad(score, adv_img)[0]
        adv_img = adv_img.detach() - step_size * grad.sign()

        # clamp if eps is out of bounds
        perturbation = torch.clamp(adv_img - image, -eps, eps)
        adv_img = image + perturbation
        adv_img = torch.clamp(adv_img, 0, 255).requires_grad_()
        iter_num += 1
    return adv_img.detach()
