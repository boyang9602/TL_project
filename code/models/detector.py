import torch
import torch.nn as nn
import torch.nn.functional as F
from .rpn_proposal import RPNProposalSSD
from .dfmb_roi_align import DFMBPSROIAlign
from .faster_rcnn import RCNNProposal
from .feature_net import FeatureNet

im_info = torch.tensor([270, 270])
bbox_reg_param = {
    'bbox_mean': [0.000437, 0.002586, -0.123953, -0.081469],
    'bbox_std':  [0.126770, 0.095741,  0.317300,  0.281042]
}
detection_output_ssd_param = {
    'heat_map_a': 8,
    'min_size_h': 6.160560,
    'min_size_w': 6.160560,
    'min_size_mode': 'HEIGHT_OR_WIDTH',
    'threshold_objectness': 0.200000,
    'gen_anchor_param': {
        'anchor_widths': [9.232984,   16.0, 27.712813,  18.465969,  32.0,   55.425626,  36.931937,  64.0, 
                          110.851252, 73.863875,  128.0,  221.702503, 147.72775,  256.0,    443.405007],
        'anchor_heights': [27.72668,  16.0, 9.237604,   55.453359,  32.0,   18.475209,  110.906719, 64.0,
                           36.950417, 221.813438, 128.0,  73.900834,  443.626876, 256.0,    147.801669]
    },
    'refine_out_of_map_bbox': True,
    'nms_param': {
        'overlap_ratio': 0.700000,
        'top_n': 300,
        'max_candidate_n': 3000,
        'use_soft_nms': False,
        'voting': False,
        'vote_iou': 0.700000
    }
}

dfmb_psroi_pooling_param = {
    'heat_map_a': 8,
    'output_dim': 10,
    'group_height': 7,
    'group_width': 7,
    'pooled_height': 7,
    'pooled_width': 7,
    'pad_ratio': 0.000000,
    'sample_per_part': 4
}

rcnn_bbox_reg_param = {
    'bbox_mean': [0.000000, 0.000000, 0.000000, 0.000000],
    'bbox_std': [0.100000, 0.100000, 0.200000, 0.200000]
}
rcnn_detection_output_ssd_param = {
    'num_class': 3,
    'rpn_proposal_output_score': True,
    'regress_agnostic': False,
    'min_size_h': 8.800800,
    'min_size_w': 8.800800,
    'min_size_mode': 'HEIGHT_OR_WIDTH',
    'threshold_objectness': 0.100000,
    'thresholds': [0.100000, 0.100000, 0.100000],
    'refine_out_of_map_bbox': True,
    'nms_param': {
        'overlap_ratio': 0.500000,
        'top_n': 5,
        'max_candidate_n': 300,
        'use_soft_nms': False,
        'voting': False,
        'vote_iou': 0.600000
    }
}

class TFModel(nn.Module):
    """
    The entire network for traffic light detection.
    """
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.feature_net = FeatureNet().to(device)
        self.proposal = RPNProposalSSD(bbox_reg_param, detection_output_ssd_param, device=self.device)
        self.psroi_rois = DFMBPSROIAlign(dfmb_psroi_pooling_param, device=self.device)
        self.inner_rois = nn.Linear(in_features=10 * 7 * 7, out_features=2048, bias=True, device=self.device)
        self.cls_score = nn.Linear(in_features=2048, out_features=4, bias=True, device=self.device)
        self.bbox_pred = nn.Linear(in_features=2048, out_features=16, bias=True, device=self.device)
        self.rcnn_proposal = RCNNProposal(rcnn_bbox_reg_param, rcnn_detection_output_ssd_param, device=self.device)

    def forward(self, x):
        rpn_cls_prob_reshape, rpn_bbox_pred, ft_add_left_right = self.feature_net(x)
        # print(rpn_cls_prob_reshape.shape, rpn_bbox_pred.shape, ft_add_left_right.shape)
        rois, scores = self.proposal(rpn_cls_prob_reshape, rpn_bbox_pred, im_info)
        # print(rois.shape)
        psroi_rois = self.psroi_rois(ft_add_left_right, rois)
        # print(psroi_rois.shape)
        inner_rois = F.relu(self.inner_rois(psroi_rois.reshape(-1, 490)))
        # print(inner_rois.shape)
        cls_score = self.cls_score(inner_rois)
        bbox_pred = self.bbox_pred(inner_rois)
        cls_score_softmax = F.softmax(cls_score, dim=1)
        # print(cls_score.shape, cls_score_softmax.shape, bbox_pred.shape)
        bboxes = self.rcnn_proposal(cls_score_softmax, bbox_pred, rois, im_info)
        # print(bboxes.shape)

        return bboxes, rois, scores

# torch.Size([1, 30, 34, 34]) torch.Size([1, 60, 34, 34]) torch.Size([1, 490, 34, 34])
# torch.Size([52, 5])
# torch.Size([52, 10, 49])
# torch.Size([52, 2048])
# torch.Size([52, 4]) torch.Size([52, 4]) torch.Size([52, 16])
# torch.Size([3, 9])