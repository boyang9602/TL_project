import torch
import torch.nn as nn
from .detector_v2 import TFModel
from .recognizer import Recognizer
import models.src.hungarian_optimizer as hungarian_optimizer
from tools.utils import preprocess4det, preprocess4rec, restore_boxes_to_full_image, nms, boxes2projections
from .selector import select_tls

class Pipeline(nn.Module):
    """
    This class will be responsible for detecting and recognizing a single ROI.
    """
    def __init__(self, detector, classifiers, ho, means_det, means_rec, device=None):
        super().__init__()
        self.detector = detector
        self.classifiers = classifiers
        self.means_det = means_det
        self.means_rec = means_rec
        self.ho = ho
        self.device = device
    def detect(self, image, boxes):
        """bboxes should be a list of list, each sub-list is like [xmin, ymin, xmax, ymax]"""
        detected_boxes = []
        projections = boxes2projections(boxes)
        rpn_attack_data = []
        for projection in projections:
            img = image.clone().to(self.device)
            input = preprocess4det(img, projection, self.means_det)
            bboxes, rois, objness_scores = self.detector(input.unsqueeze(0).permute(0, 3, 1, 2))
            detected_boxes.append(bboxes)
            rois_w_scores = torch.hstack([rois, objness_scores])
            rpn_attack_data.append(rois_w_scores)
        detections = restore_boxes_to_full_image(image, detected_boxes, projections)
        rpn_attack_data = restore_boxes_to_full_image(image, rpn_attack_data, projections)
        if len(detections) == 0:
            detections = torch.empty([0, 9], device=self.device)
        else:
            detections = torch.vstack(detections)
        if len(rpn_attack_data) == 0:
            rpn_attack_data = torch.empty([0, 7], device=self.device)
        else:
            rpn_attack_data = torch.vstack(rpn_attack_data)
        idxs = nms(detections[:, 1:5], 0.6)
        detections = detections[idxs]
        idxs = nms(rpn_attack_data[:, 1:5], 0.6)
        rpn_attack_data = rpn_attack_data[idxs]
        return detections, rpn_attack_data
    def recognize(self, img, detections):
        recognitions = []
        for _, detection in enumerate(detections):
            det_box = detection[1:5].type(torch.long)
            tl_type = torch.argmax(detection[5:]).item() - 1
            recognizer, shape = self.classifiers[tl_type]
            input = preprocess4rec(img, det_box, shape, self.means_rec)
            output = recognizer(input.permute(2, 0, 1).unsqueeze(0))
            assert output.shape[0] == 1
            recognitions.append(output[0])
        if len(recognitions) == 0:
            return torch.empty([0, 4], device=self.device)
        return torch.vstack(recognitions)
    def forward(self, img, boxes):
        """img should not substract the means, if there's a perturbation, the perturbation should be added to the img
        return valid_detections, recognitions, assignments, invalid_detections
        """
        detections, rpn_attack_data = self.detect(img, boxes)
        if len(detections) == 0:
            return torch.empty([0, 9], device=self.device), \
                    torch.empty([0, 4], device=self.device), \
                    torch.empty([0, 2], device=self.device), \
                    torch.empty([0, 9], device=self.device), \
                    torch.empty([0, 7], device=self.device)
        tl_types = torch.argmax(detections[:, 5:], dim=1)
        valid_inds = tl_types != 0
        invalid_inds = tl_types == 0
        valid_detections = detections[valid_inds]
        invalid_detections = detections[invalid_inds]
        assignments = select_tls(self.ho, valid_detections, boxes2projections(boxes), img.shape).to(self.device)
        # in theory, we only recognize the selected TLs. 
        # however, for attacking, it would be better to gain more information
        # so we recognize all. It will be slower but it's fine.
        recognitions = self.recognize(img, valid_detections)
        return valid_detections, recognitions, assignments, invalid_detections, rpn_attack_data
    
def load_pipeline(device=None):
    print(f'Loaded the TL pipeline. Device is {device}')
    means_det = torch.Tensor([102.9801, 115.9465, 122.7717]).to(device)
    means_rec = torch.Tensor([69.06, 66.58, 66.56]).to(device)

    detector = TFModel(device=device)
    detector.load_state_dict(torch.load('code/models/weights/tl.torch'))
    detector = detector.to(device)
    detector.eval();

    quad_pool_params = {'kernel_size': (4, 4), 'stride': (4, 4)}
    hori_pool_params = {'kernel_size': (2, 6), 'stride': (2, 6)}
    vert_pool_params = {'kernel_size': (6, 2), 'stride': (6, 2)}
    quad_recognizer = Recognizer(quad_pool_params)
    hori_recognizer = Recognizer(hori_pool_params)
    vert_recognizer = Recognizer(vert_pool_params)

    quad_recognizer.load_state_dict(torch.load('code/models/weights/quad.torch'))
    quad_recognizer = quad_recognizer.to(device)
    quad_recognizer.eval();

    hori_recognizer.load_state_dict(torch.load('code/models/weights/hori.torch'))
    hori_recognizer = hori_recognizer.to(device)
    hori_recognizer.eval();

    vert_recognizer.load_state_dict(torch.load('code/models/weights/vert.torch'))
    vert_recognizer = vert_recognizer.to(device)
    vert_recognizer.eval();
    classifiers = [(vert_recognizer, (96, 32, 3)), (quad_recognizer, (64, 64, 3)), (hori_recognizer, (32, 96, 3))]

    ho = hungarian_optimizer.HungarianOptimizer()
    pipeline = Pipeline(detector, classifiers, ho, means_det, means_rec, device=device)
    return pipeline