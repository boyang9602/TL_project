import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import csv
import xml.etree.ElementTree as ET
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
fake_image = torch.empty([720, 1280, 3])
TL_TYPE_TEXTS = ['VERT', 'QUAD', 'HORI']
REC_COLORS = ["off", "red", "yellow", "green"]

def readxml2(filename):
    """return the bounding boxes of the given annotation file in xml format"""
    bboxes = []
    colors = []
    tree = ET.parse(filename)
    objs = tree.findall("object")
    for obj in objs:
        xmin = int(float(obj.find("bndbox/xmin").text))
        xmax = int(float(obj.find("bndbox/xmax").text))
        ymin = int(float(obj.find("bndbox/ymin").text))
        ymax = int(float(obj.find("bndbox/ymax").text))
        bboxes.append([xmin, ymin, xmax, ymax])
        color = obj.find('name').text
        colors.append(color)
    return bboxes, colors

def IoU_single(box1, box2):
    """
    use it to inspect two single boxes
    boxes should be in shape [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    w = max(w, 0)
    h = max(h, 0)
    inter = w * h
    
    a1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    a2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union = a1 + a2 - inter
    return inter / union

def IoG_single(box1, box2):
    """
    intersection over ground truth
    """
    x1 = int(max(box1[0], box2[0]))
    y1 = int(max(box1[1], box2[1]))
    x2 = int(min(box1[2], box2[2]))
    y2 = int(min(box1[3], box2[3]))
    
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    w = max(w, 0)
    h = max(h, 0)
    inter = w * h
    
    a1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)

    return inter / a1

def viz_pipeline_results(image, valid_detections=None, recognitions=None, assignments=None, invalid_detections=None, projections=None, fig_width=20.48, fig_height=15.36):
    """
      results is a n*13 tensor, n TL instances, 
      for each instance,
      index 0: pass
      index 1-4: bbox
      index 5-8: TL type softmax
      index 9-end: classification softmax
    """
    fig, ax = plt.subplots(1)
    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_height)
    ax.imshow(image)
    if valid_detections != None:
        for i, result in enumerate(valid_detections):
            xmin = int(result[1].item())
            ymin = int(result[2].item())
            xmax = int(result[3].item())
            ymax = int(result[4].item())
            edge_color = 'r'
            if i in assignments[:, 1]:
                edge_color = 'g'
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=edge_color, facecolor="none")
            ax.add_patch(rect)
            tl_type = torch.argmax(result[5:9]).item() - 1
            assert tl_type >= 0 # tl_type should be not unknown or the classification scores should be all -1s. 
            tl_color = torch.argmax(recognitions[i])
            ax.text(xmin, ymin-2, f'{TL_TYPE_TEXTS[tl_type]}:{REC_COLORS[tl_color]}', c='r', fontsize='large')
    if invalid_detections != None:
        for i, result in enumerate(invalid_detections):
            xmin = int(result[1].item())
            ymin = int(result[2].item())
            xmax = int(result[3].item())
            ymax = int(result[4].item())
            edge_color = 'r'
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=edge_color, facecolor="none")
            ax.add_patch(rect)
    if projections != None:
        for projection in projections:
            # print(projection)
            rect = patches.Rectangle((projection.x, projection.y), projection.w, projection.h, linewidth=2, edgecolor='c', facecolor="none")
            ax.add_patch(rect)
            coor = crop(image, projection)
            rect = patches.Rectangle((coor[0], coor[2]), coor[1] - coor[0], coor[3] - coor[2], linewidth=2, edgecolor='y', facecolor="none")
            ax.add_patch(rect)
    return fig, ax

def nms(boxes, thresh_iou):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image 
            along with the class predscores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """
    # we extract coordinates for every 
    # prediction box present in P
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # calculate area of every block in P
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # initialise an empty list for 
    # filtered prediction boxes
    keep_inds = []
    idxs = torch.arange(boxes.shape[0]).to(device)
    while len(idxs) > 0:
        idx = idxs[0]
        # push S in filtered predictions list
        keep_inds.append(idx)
        idxs = idxs[1:]

        if len(idxs) == 0:
            break
        
        # select coordinates of BBoxes according to 
        # the indices in order
        xx1 = x1[idxs]
        xx2 = x2[idxs]
        yy1 = y1[idxs]
        yy2 = y2[idxs]

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1 + 1
        h = yy2 - yy1 + 1
        
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = areas[idxs]

        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]
        
        # find the IoU of every prediction in P with S
        IoU = inter / union

        # keep the boxes with IoU less than thresh_iou
        mask = IoU <= thresh_iou
        idxs = idxs[mask]
    return torch.tensor(keep_inds, dtype=torch.long)

class ProjectionROI:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.xl = x
        self.yt = y
        self.xr = x + w - 1
        self.yb = y + h - 1
        self.center_x = int((self.xl + self.xr) / 2)
        self.center_y = int((self.yt + self.yb) / 2)

def ResizeGPU(src, dst, means):
    """
    src is in the shape of (H,  W,  C ),
    dst is in the shape of (H', W', C'),
    """
    fx = src.shape[1] / dst.shape[1]
    fy = src.shape[0] / dst.shape[0]

    ###########################Use Meshgrid#############################
    # dst_x = torch.zeros(dst.shape[0], dst.shape[1], device=device)
    # for i in range(0, dst.shape[1]):
    #     dst_x[:,i] = i
    # dst_y = torch.zeros(dst.shape[0], dst.shape[1], device=device)
    # for i in range(0, dst.shape[0]):
    #     dst_y[i,:] = i
    ####################################################################
    dst_x, dst_y = torch.meshgrid([torch.arange(dst.shape[1], device=device), torch.arange(dst.shape[0], device=device)], indexing="xy")

    src_x = (dst_x + 0.5) * fx - 0.5
    src_y = (dst_y + 0.5) * fy - 0.5
    x1 = torch.floor(src_x).type(torch.long)
    y1 = torch.floor(src_y).type(torch.long)
    x1_read = torch.clamp(x1, 0)
    y1_read = torch.clamp(y1, 0)
    x2 = x1 + 1
    y2 = y1 + 1
    x2_read = torch.clamp(x2, max=src.shape[1] - 1)
    y2_read = torch.clamp(y2, max=src.shape[0] - 1)
    
    src_reg = src[y1_read, x1_read, :]
    dst += (x2 - src_x)[:, :, None] * (y2 - src_y)[:, :, None] * src_reg
    src_reg = src[y1_read, x2_read, :]
    dst += (src_x - x1)[:, :, None] * (y2 - src_y)[:, :, None] * src_reg
    src_reg = src[y2_read, x1_read, :]
    dst += (x2 - src_x)[:, :, None] * (src_y - y1)[:, :, None] * src_reg
    src_reg = src[y2_read, x2_read, :]
    dst += (src_x - x1)[:, :, None] * (src_y - y1)[:, :, None] * src_reg
    dst = torch.clamp(dst.clone(), 0, 255)
    if means != None:
        dst -= means
    return dst

def crop(image, projection):
    width = image.shape[1]
    height = image.shape[0]
    crop_scale = 2.5
    min_crop_size = 270
    resize = crop_scale * max(projection.w, projection.h)
    resize = max(resize, min_crop_size)
    resize = min(resize, width)
    resize = min(resize, height)
    xl = projection.center_x - resize/2 + 1
    xl = 0 if xl < 0 else xl
    yt = projection.center_y - resize/2 + 1
    yt = 0 if yt < 0 else yt
    xr = xl + resize - 1
    yb = yt + resize - 1
    if xr >= width - 1:
        xl -= xr - width + 1
        xr = width - 1
    if yb >= height - 1:
        yt -= yb - height + 1
        yb = height - 1
    return [int(xl), int(xr + 1), int(yt), int(yb + 1)]

def preprocess4det(image, projection, means=None):
    xl, xr, yt, yb = crop(image, projection)
    src = image[yt:yb,xl:xr]
    dst = torch.zeros(270, 270, 3, device=device)
    resized = ResizeGPU(src, dst, means)
    return resized

def preprocess4rec(image, bbox, shape, means=None):
    xl = bbox[0]
    xr = bbox[2]
    yt = bbox[1]
    yb = bbox[3]
    src = image[yt:yb,xl:xr]
    # if means != None:
    #     src = src - means
    # dst = F.interpolate((src).permute(2,0,1).unsqueeze(0), shape, mode="bilinear").squeeze().permute(1, 2, 0)
    dst = torch.zeros(shape, device=device)
    resized = ResizeGPU(src, dst, means)
    return resized

def bgr2rgb(image):
    return image[:,:,[2,1,0]]

def readxml(filename):
    """return the bounding boxes of the given annotation file in xml format"""
    bboxes = []
    tree = ET.parse(filename)
    objs = tree.findall("object")
    for obj in objs:
        xmin = int(float(obj.find("bndbox/xmin").text))
        xmax = int(float(obj.find("bndbox/xmax").text))
        ymin = int(float(obj.find("bndbox/ymin").text))
        ymax = int(float(obj.find("bndbox/ymax").text))
        bboxes.append([xmin, ymin, xmax, ymax])
    return bboxes

def restore_boxes_to_full_image(image, detections, projections):
    """
    detections is a list of the output of each projection, each is in shape (n, 9)
    this func will convert the the detected boxes from the ROI's location to the full image's location
    """
    ret = []
    assert len(detections) == len(projections), f'{len(detections)} == {len(projections)}'
    for detection, projection in zip(detections, projections):
        for i in range(detection.shape[0]):
            if detection[i][0] < 0:
                detection = detection[:i]
                break
        xl, xr, yt, yb = crop(image, projection)
        detection[:, 1] += xl
        detection[:, 2] += yt
        detection[:, 3] += xl
        detection[:, 4] += yt
        ret.append(detection)
    return ret

def box2projection(box):
    return ProjectionROI(box[0], box[1], box[2] - box[0], box[3] - box[1])

def boxes2projections(boxes):
    projections = []
    for box in boxes:
        projection = box2projection(box)
        projections.append(projection)
    return projections

def IoU_multi(boxes1, boxes2):
    """
    boxes1 should be (m, 4), boxes2 should be (n, 4)
    The results will be (m, n), where i-th row is the area between the i-th element of boxes1 and all the elements of boxes2. 
    """
    x11 = boxes1[:, None, :][:, :, 0]
    x12 = boxes1[:, None, :][:, :, 2]
    y11 = boxes1[:, None, :][:, :, 1]
    y12 = boxes1[:, None, :][:, :, 3]

    x21 = boxes2[None, :, :][:, :, 0]
    x22 = boxes2[None, :, :][:, :, 2]
    y21 = boxes2[None, :, :][:, :, 1]
    y22 = boxes2[None, :, :][:, :, 3]

    # let's say dt_* is (m, 4), gt_* is (n, 4)
    # inter_* will be (m, n)
    # each line has n numbers, which are the x1 or x2 for the intersection between i_th detection box and all the ground_truth boxes
    inter_x1 = torch.max(x11, x21)
    inter_x2 = torch.min(x12, x22)
    inter_y1 = torch.max(y11, y21)
    inter_y2 = torch.min(y12, y22)

    width = inter_x2 - inter_x1 + 1
    height = inter_y2 - inter_y1 + 1

    width = torch.clamp(width, 0.0)
    height = torch.clamp(height, 0.0)
    inter = width * height

    areas1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    areas2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    union = areas1 + areas2 - inter

    ious = inter / union
    return ious

def test_IoU_multi():
    """
    verify the correctness using IoU_single, which is a much simpler solution but very slow compared to batch operation.
    """
    boxes1 = torch.randn(3, 4)
    boxes2 = torch.randn(6, 4)
    ious = IoU_multi(boxes1, boxes2)
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou = IoU_single(box1, box2)
            assert abs(ious[i, j] - iou) < 1e-8, f'box1: {box1}, box2: {box2}, iou_multi: {ious[i, j]}, iou_single: {iou}'
for i in range(10):
    test_IoU_multi()
