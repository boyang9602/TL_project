import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import ceil, floor
from utils import preprocess4det, preprocess4rec, restore_boxes_to_full_image, nms, boxes2projections
from selector import select_tls
device = "cuda" if torch.cuda.is_available() else "cpu"

class Pipeline(nn.Module):
    """
    This class will be responsible for detecting and recognizing a single ROI.
    """
    def __init__(self, detector, classifiers, ho, means_det, means_rec):
        super().__init__()
        self.detector = detector
        self.classifiers = classifiers
        self.means_det = means_det
        self.means_rec = means_rec
        self.ho = ho
        self.rpn_attack_data = None
    def detect(self, image, boxes):
        """bboxes should be a list of list, each sub-list is like [xmin, ymin, xmax, ymax]"""
        detected_boxes = []
        projections = boxes2projections(boxes)
        rpn_attack_data = []
        for projection in projections:
            img = image.clone().to(device)
            input = preprocess4det(img, projection, self.means_det)
            bboxes, rois, objness_scores = self.detector(input.unsqueeze(0).permute(0, 3, 1, 2))
            detected_boxes.append(bboxes)
            rois_w_scores = torch.hstack([rois, objness_scores])
            rpn_attack_data.append(rois_w_scores)
        detections = restore_boxes_to_full_image(image, detected_boxes, projections)
        detections = torch.vstack(detections)
        rpn_attack_data = restore_boxes_to_full_image(image, rpn_attack_data, projections)
        rpn_attack_data = torch.vstack(rpn_attack_data)
        idxs = nms(detections[:, 1:5], 0.6)
        detections = detections[idxs]
        idxs = nms(rpn_attack_data[:, 1:5], 0.6)
        self.rpn_attack_data = rpn_attack_data[idxs]
        return detections
    def recognize(self, img, detections):
        recognitions = []
        for i, detection in enumerate(detections):
            det_box = detection[1:5].type(torch.long)
            tl_type = torch.argmax(detection[5:]).item() - 1
            # if tl_type == -1: # unknown type will not be recognized
            #     recognition = torch.ones(1, 4, device=device) * -1
            #     recognitions.append(recognition)
            #     continue
            recognizer, shape = self.classifiers[tl_type]
            input = preprocess4rec(img, det_box, shape, self.means_rec)
            output = recognizer(input.permute(2, 0, 1).unsqueeze(0))
            assert output.shape[0] == 1
            recognitions.append(output[0])
        if len(recognitions) == 0:
            return torch.empty([0, 4], device=device)
        return torch.vstack(recognitions)
    def forward(self, img, boxes):
        """img should not substract the means, if there's a perturbation, the perturbation should be added to the img
        return valid_detections, recognitions, assignments, invalid_detections
        """
        detections = self.detect(img, boxes)
        if len(detections) == 0:
            return torch.empty([0, 9], device=device), None, None, None
        tl_types = torch.argmax(detections[:, 5:], dim=1)
        valid_inds = tl_types != 0
        invalid_inds = tl_types == 0
        valid_detections = detections[valid_inds]
        invalid_detections = detections[invalid_inds]
        assignments = select_tls(self.ho, valid_detections, boxes2projections(boxes))
        # in theory, we only recognize the selected TLs. 
        # however, for attacking, it would be better to gain more information
        # so we recognize all. It will be slower but it's fine.
        recognitions = self.recognize(img, valid_detections)
        return valid_detections, recognitions, assignments, invalid_detections