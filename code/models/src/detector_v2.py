import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import nms
import time
PERF_LOG = False

# detection
class RPNProposalSSD(nn.Module):
    def __init__(self, bbox_reg_param, detection_output_ssd_param, device=None):
        super(RPNProposalSSD, self).__init__()
        self.device = device
        # read box params
        self.bbox_mean = torch.tensor(bbox_reg_param['bbox_mean'], device=self.device)
        self.bbox_std = torch.tensor(bbox_reg_param['bbox_std'], device=self.device)
        # read detection param
        self.anchor_stride = detection_output_ssd_param['heat_map_a']
        self.gen_anchor_param = detection_output_ssd_param['gen_anchor_param']
        self.num_anchor_per_point = len(self.gen_anchor_param['anchor_widths'])
        self.min_size_mode = detection_output_ssd_param['min_size_mode']
        self.min_size_h = detection_output_ssd_param['min_size_h']
        self.min_size_w = detection_output_ssd_param['min_size_w']
        self.threshold_objectness = detection_output_ssd_param['threshold_objectness']
        self.nms_param = detection_output_ssd_param['nms_param']
        self.refine_out_of_map_bbox = detection_output_ssd_param['refine_out_of_map_bbox']

    def generate_anchors(self):
        """
        anchor is represented by 4 pts (xmin, ymin, xmax, ymax)
        """
        anchor_widths = torch.tensor(self.gen_anchor_param['anchor_widths'], device=self.device)
        anchor_heights = torch.tensor(self.gen_anchor_param['anchor_heights'], device=self.device)
        xmins = - 0.5 * (anchor_widths - 1)
        xmaxs = + 0.5 * (anchor_widths - 1)
        ymins = - 0.5 * (anchor_heights - 1)
        ymaxs = + 0.5 * (anchor_heights - 1)
        anchors = torch.vstack((xmins, ymins, xmaxs, ymaxs)).transpose(1, 0)
        return anchors

    def bbox_transform_inv(self, boxes, deltas):
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1)

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes = torch.zeros(deltas.shape, dtype=deltas.dtype, device=self.device)
        # x1
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * (pred_w - 1)
        # y1
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * (pred_h - 1)
        # x2
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * (pred_w - 1)
        # y2
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * (pred_h - 1)
        return pred_boxes

    def clip_boxes(self, boxes, height, width):
        """
        Clip boxes to image boundaries.
        """
        clone = boxes.clone()
        boxes[:, 0::4] = torch.clamp(clone[:, 0::4], 0, width - 1)
        boxes[:, 1::4] = torch.clamp(clone[:, 1::4], 0, height - 1)
        boxes[:, 2::4] = torch.clamp(clone[:, 2::4], 0, width - 1)
        boxes[:, 3::4] = torch.clamp(clone[:, 3::4], 0, height - 1)
        return boxes

    def filter_boxes(self, proposals, scores, num_box, num_class, filter_class, min_size_mode, min_size_h, min_size_w, threshold_score):
        # filter cases whose scores are below the threshold
        keep = scores[:, filter_class] > threshold_score
        proposals = proposals[keep]
        scores = scores[keep]

        # filter out cases whose widths and heights are lower than the min_size_w/h
        ws = proposals[:, 2] - proposals[:, 0] + 1
        hs = proposals[:, 3] - proposals[:, 1] + 1
        assert torch.all(ws >= 0) and torch.all(hs >= 0)

        if min_size_mode == 'HEIGHT_AND_WIDTH':
            keep = ws >= min_size_w
            keep *= hs >= min_size_h # get the && of boolean
        elif min_size_mode == 'HEIGHT_OR_WIDTH':
            keep = ws >= min_size_w
            keep += hs >= min_size_h # get the || of boolean
        else:
            raise
        return proposals[keep], scores[keep]

    def forward(self, rpn_cls_prob_reshape, rpn_bbox_pred, im_info):
        """
        rpn_cls_prob_reshape has a shape of [N, 2 * num_anchor_per_point, W, H]
        rpn_bbox_pred        has a shape of [N, 4 * num_anchor_per_point, W, H]
        im_info (origin_width, origin_height, )
        part of the implementation refers to https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/proposal_layer.py#L47
        """
        if PERF_LOG:
            tic = time.perf_counter()
        assert rpn_cls_prob_reshape.shape[0] == 1 # only support batch=1
        origin_height = im_info[0]
        origin_width  = im_info[1]
        height = rpn_cls_prob_reshape.shape[-2]
        width  = rpn_cls_prob_reshape.shape[-1]
        num_anchor = self.num_anchor_per_point * height * width
        anchor_size = num_anchor * 4

        # Enumerate all shifts
        shift_x = torch.arange(width, device=self.device) * self.anchor_stride
        shift_y = torch.arange(height, device=self.device) * self.anchor_stride
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing='ij')
        shifts = torch.vstack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel())).transpose(1, 0)
        
        anchors = self.generate_anchors()
        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        anchors = anchors.unsqueeze(0) + shifts.unsqueeze(1)
        anchors = anchors.reshape((-1, 4))

        # reshape the predicted regressors to (W * H * num_anchor_per_point, 4)
        rpn_bbox_pred = rpn_bbox_pred.reshape(self.num_anchor_per_point, 4, 34, 34).permute(2, 3, 0, 1).reshape(-1, 4)
        rpn_bbox_pred = rpn_bbox_pred * self.bbox_std # multiply the 4 std with each row (i.e., the 4 regressors)
        rpn_bbox_pred = rpn_bbox_pred + self.bbox_mean # add the 4 mean to each row (i.e., the 4 regressors)

        # Convert anchors into proposals via bbox transformations
        # print(anchors.shape, rpn_bbox_pred.shape)
        proposals = self.bbox_transform_inv(anchors, rpn_bbox_pred)
        # clip boxes, i.e. refine proposals which are out of map
        if self.refine_out_of_map_bbox:
            proposals = self.clip_boxes(proposals, origin_height, origin_width)

        # reshape scores
        scores = rpn_cls_prob_reshape.reshape(2, self.num_anchor_per_point, -1).permute(2, 1, 0).reshape(-1, 2)

        proposals, scores = self.filter_boxes(proposals, scores, num_anchor, 2, 1, self.min_size_mode, self.min_size_h, self.min_size_w, self.threshold_objectness)

        # keep max N candidates
        top_indices = torch.topk(scores[:, 1], min(scores.shape[0], self.nms_param['max_candidate_n'])).indices
        proposals = proposals[top_indices]
        scores = scores[top_indices]

        # apply NMS
        # nms_indices = torchvision.ops.nms(proposals, scores[:, 1], self.nms_param['overlap_ratio'])
        # nms_indices = self.nms(proposals, self.nms_param['overlap_ratio'])
        nms_indices = nms(proposals, self.nms_param['overlap_ratio'])
        # print(nms_indices, nms_indices.type())
        proposals = proposals[nms_indices][:self.nms_param['top_n']]
        scores = scores[nms_indices][:self.nms_param['top_n']]
        proposals = torch.hstack([torch.zeros((proposals.shape[0], 1), device=self.device), proposals])
        if PERF_LOG:
            toc = time.perf_counter()
            print(f"RPN done in {toc-tic:0.4f} seconds!")
        return proposals, scores
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

class DFMBPSROIAlign(nn.Module):
    def __init__(self, dfmb_psroi_pooling_param, device=None):
        super(DFMBPSROIAlign, self).__init__()
        self.device = device
        self.pooled_height = dfmb_psroi_pooling_param['pooled_height']
        self.pooled_width = dfmb_psroi_pooling_param['pooled_width']
        self.anchor_stride = dfmb_psroi_pooling_param['heat_map_a']
        self.sample_per_part = dfmb_psroi_pooling_param['sample_per_part']

        self.channels = 10
        self.width = 34
        self.height = 34

    def forward(self, ft_add_left_right, rois):
        """
        compute the ROI area on the feature map 
        refers to https://github.com/ApolloAuto/apollo/blob/v7.0.0/modules/perception/inference/tensorrt/plugins/dfmb_psroi_align_plugin.cu 
        and https://erdem.pl/2020/02/understanding-region-of-interest-part-2-ro-i-align
        """
        if PERF_LOG:
            tic = time.perf_counter()
        if len(rois.shape) == 4:
            rois = rois.squeeze(2)
            rois = rois.squeeze(2)
        ft_add_left_right = ft_add_left_right[0].reshape(self.channels, self.pooled_height, self.pooled_width, self.height, self.width)
        
        # ROI positions. In the original code, it has calculations with pad_w/h and heat_map_b. 
        # Not sure what they are, there values should be 0 by my analysis, so just ignore them.
        roi_start_w = rois[:,1] / self.anchor_stride
        roi_start_h = rois[:,2] / self.anchor_stride
        roi_end_w   = rois[:,3] / self.anchor_stride
        roi_end_h   = rois[:,4] / self.anchor_stride
        
        roi_height = roi_end_h - roi_start_h
        roi_width  = roi_end_w - roi_start_w
        roi_height = torch.threshold(roi_height, 0.1, 0.1)
        roi_width  = torch.threshold(roi_width,  0.1, 0.1)

        bin_size_h = roi_height / self.pooled_height
        bin_size_w = roi_width  / self.pooled_width
        sub_bin_size_h = bin_size_h / self.sample_per_part
        sub_bin_size_w = bin_size_w / self.sample_per_part

        grid = torch.meshgrid(torch.arange(self.pooled_height, device=self.device), torch.arange(self.pooled_width, device=self.device), indexing='ij')
        phs = grid[0].reshape(-1, 1) # => 49, 1
        pws = grid[1].reshape(-1, 1) # => 49, 1

        hstart = torch.floor(roi_start_h + phs * bin_size_h) # => 49, n
        wstart = torch.floor(roi_start_w + pws * bin_size_w) # => 49, n

        sum_ = torch.zeros((self.channels, self.pooled_height * self.pooled_width, rois.shape[0]), device=self.device)
        count = torch.zeros((self.pooled_height * self.pooled_width, rois.shape[0]), device=self.device)

        for ih in range(self.sample_per_part):
            for iw in range(self.sample_per_part):
                # w and h are the samples
                w = wstart + (iw + 0.5) * sub_bin_size_w
                h = hstart + (ih + 0.5) * sub_bin_size_h

                keep = (w > -1) * (w < self.width) * (h > -1) * (h < self.height)

                # bilinear interpolation
                x1 = torch.floor(w).to(torch.long)
                x2 = torch.ceil(w).to(torch.long)
                y1 = torch.floor(h).to(torch.long)
                y2 = torch.ceil(h).to(torch.long)
                x1valid = (x1 >= 0) * (x1 < self.width)
                x2valid = (x2 >= 0) * (x2 < self.width)
                y1valid = (y1 >= 0) * (y1 < self.height)
                y2valid = (y2 >= 0) * (y2 < self.height)

                x1 = torch.clamp(x1, 0, 33)
                x2 = torch.clamp(x2, 0, 33)
                y1 = torch.clamp(y1, 0, 33)
                y2 = torch.clamp(y2, 0, 33)

                dist_x = w - x1 # => 49, n
                dist_y = h - y1 # => 49, n

                assert x1.shape == x2.shape and x1.shape == y1.shape and x1.shape == y2.shape

                value11 = ft_add_left_right[:, phs, pws, y1, x1] # => 10, 49, n
                value11[:, ~x1valid * y1valid] = 0
                value12 = ft_add_left_right[:, phs, pws, y2, x1] # => 10, 49, n
                value11[:, ~x1valid * y2valid] = 0
                value21 = ft_add_left_right[:, phs, pws, y1, x2] # => 10, 49, n
                value11[:, ~x2valid * y1valid] = 0
                value22 = ft_add_left_right[:, phs, pws, y2, x2] # => 10, 49, n
                value11[:, ~x2valid * y2valid] = 0

                value = (1 - dist_x) * (1 - dist_y) * value11 \
                        + (1 - dist_x) * dist_y * value12 \
                        + dist_x * (1 - dist_y) * value21 \
                        + dist_x * dist_y * value22 # => 10, 49, n
                sum_[:, keep] += value[:, keep]
                count[keep] += 1
        nonzero = count > 0
        sum_[:, nonzero] /= count[nonzero]
        if PERF_LOG:
            toc = time.perf_counter()
            print(f"PSROI done in {toc-tic:0.4f} seconds!")

        return sum_.permute(2, 0, 1)

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
class RCNNProposal(nn.Module):
    def __init__(self, bbox_reg_param, detection_output_ssd_param, device=None):
        super(RCNNProposal, self).__init__()
        self.device = device
        self.bbox_mean = torch.tensor(bbox_reg_param['bbox_mean'], device=self.device)
        self.bbox_std = torch.tensor(bbox_reg_param['bbox_std'], device=self.device)
        self.num_class = detection_output_ssd_param['num_class']
        self.rpn_proposal_output_score = detection_output_ssd_param['rpn_proposal_output_score']
        self.regress_agnostic = detection_output_ssd_param['regress_agnostic']
        self.min_size_h = detection_output_ssd_param['min_size_h']
        self.min_size_w = detection_output_ssd_param['min_size_w']
        self.min_size_mode = detection_output_ssd_param['min_size_mode']
        self.threshold_objectness = detection_output_ssd_param['threshold_objectness']
        self.thresholds = detection_output_ssd_param['thresholds']
        self.refine_out_of_map_bbox = detection_output_ssd_param['refine_out_of_map_bbox']
        self.nms_param = detection_output_ssd_param['nms_param']

    def bbox_transform_inv_rcnn(self, boxes, deltas):
        if len(boxes.shape) == 4:
            boxes = boxes.squeeze(2)
            boxes = boxes.squeeze(2)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1)

        dx = deltas[:, :, 0]
        dy = deltas[:, :, 1]
        dw = deltas[:, :, 2]
        dh = deltas[:, :, 3]

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros(deltas.shape, dtype=deltas.dtype, device=self.device)
        # x1
        pred_boxes[:, :, 0] = pred_ctr_x - 0.5 * (pred_w - 1)
        # y1
        pred_boxes[:, :, 1] = pred_ctr_y - 0.5 * (pred_h - 1)
        # x2
        pred_boxes[:, :, 2] = pred_ctr_x + 0.5 * (pred_w - 1)
        # y2
        pred_boxes[:, :, 3] = pred_ctr_y + 0.5 * (pred_h - 1)
        return pred_boxes

    def clip_boxes(self, boxes, height, width):
        """
        Clip boxes to image boundaries.
        """
        clone = boxes.clone()
        boxes[:, 0] = torch.clamp(clone[:, 0], 0, width - 1)
        boxes[:, 1] = torch.clamp(clone[:, 1], 0, height - 1)
        boxes[:, 2] = torch.clamp(clone[:, 2], 0, width - 1)
        boxes[:, 3] = torch.clamp(clone[:, 3], 0, height - 1)
        return boxes

    def forward(self, cls_score_softmax, bbox_pred, rois, im_info):
        if PERF_LOG:
            tic = time.perf_counter()
        num_rois = rois.shape[1]
        cls_score_softmax_size = num_rois * 4
        bbox_pred_size = num_rois * 4 * 4
        output_size = 9
        origin_height = im_info[0]
        origin_width = im_info[1]
        # normalize the rois
        bbox_pred = (bbox_pred.reshape(-1, 4) * self.bbox_std + self.bbox_mean).reshape(-1, self.num_class + 1, 4)

        # slice_rois
        sliced_rois = rois[:, 1:]
        
        # decode bbox
        decoded_bbox_pred = self.bbox_transform_inv_rcnn(sliced_rois, bbox_pred)

        if self.refine_out_of_map_bbox:
            decoded_bbox_pred = self.clip_boxes(decoded_bbox_pred, origin_height, origin_width)
        
        # filter by objectness
        # bbox_pred dims: [num_box, num_class+1, 4],
        # scores dims: [num_box, num_class+1],
        indices = 1 - cls_score_softmax[:,0] >= self.threshold_objectness
        cls_score_softmax = cls_score_softmax[indices]
        decoded_bbox_pred = decoded_bbox_pred[indices]

        maxes, argmaxes = torch.max(cls_score_softmax[:,1:], 1)
        argmaxes += 1
        indices = maxes > self.thresholds[0]

        # simplified this step. In theory, 3 classes can have different threshold, but in the model definition, they are the same. 
        # So the simplification should not have any affects to the results
        cls_score_softmax = cls_score_softmax[indices]
        decoded_bbox_pred = decoded_bbox_pred[indices]
        decoded_bbox_pred = decoded_bbox_pred.reshape(-1, 4)[argmaxes[indices] + torch.arange(0, decoded_bbox_pred.shape[0], device=self.device) * 4]
        filtered_count = decoded_bbox_pred.shape[0]

        w = decoded_bbox_pred[:, 2] - decoded_bbox_pred[:, 0] + 1
        h = decoded_bbox_pred[:, 3] - decoded_bbox_pred[:, 1] + 1

        if self.min_size_mode == "HEIGHT_OR_WIDTH":
            keep = (w >= self.min_size_w) + (h >= self.min_size_h)
        elif self.min_size_mode == "HEIGHT_AND_WIDTH":
            keep = (w >= self.min_size_w) * (h >= self.min_size_h)
        decoded_bbox_pred = decoded_bbox_pred[keep]
        cls_score_softmax = cls_score_softmax[keep]
        # indices *= keep

        # keep max N candidates
        # top_indices = torch.topk(maxes[indices], min(maxes[indices].shape[0], self.nms_param['max_candidate_n'])).indices
        top_indices = torch.topk(maxes[indices][keep], min(maxes[indices][keep].shape[0], self.nms_param['max_candidate_n'])).indices
        pre_nms_bbox = decoded_bbox_pred[top_indices]
        pre_nms_all_probs = cls_score_softmax[top_indices]
        argmaxes = torch.argmax(pre_nms_all_probs[:,1:], 1) + 1
        pre_nms_score = pre_nms_all_probs.flatten()[argmaxes + 4 * torch.arange(0, len(top_indices), device=self.device)]
        
        # nms_indices = torchvision.ops.nms(pre_nms_bbox, pre_nms_score, self.nms_param['overlap_ratio'])
        # nms_indices = self.nms(pre_nms_bbox, self.nms_param['overlap_ratio'])
        nms_indices = nms(pre_nms_bbox, self.nms_param['overlap_ratio'])
        # print(nms_indices, nms_indices.type())
        boxes = pre_nms_bbox[nms_indices][:self.nms_param['top_n']]
        if PERF_LOG:
            toc = time.perf_counter()
            print(f"RCNN done in {toc-tic:0.4f} seconds!")
        return torch.hstack([torch.zeros((boxes.shape[0], 1), device=self.device), boxes, pre_nms_all_probs[nms_indices][:self.nms_param['top_n']]])
im_info = torch.tensor([270, 270])

class ConvBNScale(nn.Module):
    """
    This is a very common sub-structure in apollo's network: Convolution -> BatchNorm -> Scale.
    Note that there are inconsistencies between the Caffe's BatchNorm and standard BatchNorm. 
    Caffe's BatchNorm ->  Scale is similar to BatchNorm in PyTorch https://github.com/BVLC/caffe/blob/master/include/caffe/layers/batch_norm_layer.hpp#L28.
    Besides, Caffe's BatchNorm has an extra parameter called moving_average_fraction. The solution to handle this is in https://stackoverflow.com/questions/55644109/how-to-convert-batchnorm-weight-of-caffe-to-pytorch-bathnorm. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias)
        self.bn   = nn.BatchNorm2d(num_features=out_channels, affine=False)
        self.gamma = nn.Parameter(torch.FloatTensor(out_channels))
        self.beta  = nn.Parameter(torch.FloatTensor(out_channels))
    
    def forward(self, input):
        return self.bn(self.conv(input)) * self.gamma[None, :, None, None] + self.beta[None, :, None, None]

class FeatureNet(nn.Module):
    """
    This is the model to extract features from the image. I made it a separate network and used it in the TFModel below.
    It is based on ResNet I think.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNScale(in_channels=3, out_channels=16, kernel_size=7, padding=3, stride=2)
        self.res2a_branch1 = ConvBNScale(in_channels=16, out_channels=16, kernel_size=1, padding=0, stride=1)
        self.res2a_branch2a = ConvBNScale(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.res2a_branch2b = ConvBNScale(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.res2b_branch2a = ConvBNScale(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.res2b_branch2b = ConvBNScale(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.res2c_branch2a = ConvBNScale(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.res2c_branch2b = ConvBNScale(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.res3a_branch1 = ConvBNScale(in_channels=16, out_channels=32, kernel_size=1, padding=0, stride=2)
        self.res3a_branch2a = ConvBNScale(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.res3a_branch2b = ConvBNScale(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.res3b_branch2a = ConvBNScale(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.res3b_branch2b = ConvBNScale(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.res3c_branch2a = ConvBNScale(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.res3c_branch2b = ConvBNScale(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.res3d_branch2a = ConvBNScale(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.res3d_branch2b = ConvBNScale(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.res4a_branch1 = ConvBNScale(in_channels=32, out_channels=64, kernel_size=1, padding=0, stride=2)
        self.res4a_branch2a = ConvBNScale(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.res4a_branch2b = ConvBNScale(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.res4b_branch2a = ConvBNScale(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.res4b_branch2b = ConvBNScale(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.res4c_branch2a = ConvBNScale(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.res4c_branch2b = ConvBNScale(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.res4d_branch2a = ConvBNScale(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.res4d_branch2b = ConvBNScale(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.res4e_branch2a = ConvBNScale(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.res4e_branch2b = ConvBNScale(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.res4f_branch2a = ConvBNScale(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.res4f_branch2b = ConvBNScale(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.res5a_branch1 = ConvBNScale(in_channels=64, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.res5a_branch2a = ConvBNScale(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.res5a_branch2b = ConvBNScale(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2, stride=1)
        self.res5b_branch2a = ConvBNScale(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2, stride=1)
        self.res5b_branch2b = ConvBNScale(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2, stride=1)
        self.res5c_branch2a = ConvBNScale(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2, stride=1)
        self.res5c_branch2b = ConvBNScale(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2, stride=1)
        self.rpn_deconv = nn.ConvTranspose2d(in_channels=64, out_channels=256, kernel_size=4, padding=1, dilation=1, stride=2, bias=True)
        self.rpn_cls_score = nn.Conv2d(in_channels=256, out_channels=30, kernel_size=1, padding=0, dilation=1, stride=1, bias=True)
        self.rpn_bbox_pred = nn.Conv2d(in_channels=256, out_channels=60, kernel_size=1, padding=0, dilation=1, stride=1, bias=True)
        self.conv_new = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, padding=1, stride=2, dilation=1, bias=True)
        self.conv_left_kx1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(9, 1), padding=(4, 0), stride=(1, 1), dilation=(1, 1), bias=True)
        self.conv_left_1xk = nn.Conv2d(in_channels=128, out_channels=490, kernel_size=(1, 9), padding=(0, 4), stride=(1, 1), dilation=(1, 1), bias=True)
        self.conv_right_1xk = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 9), padding=(0, 4), stride=(1, 1,), dilation=(1, 1), bias=True)
        self.conv_right_kx1 = nn.Conv2d(in_channels=128, out_channels=490, kernel_size=(9, 1), padding=(4, 0), stride=(1, 1,), dilation=(1, 1,), bias=True)
    def forward(self, x):
        pool1 = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, padding=1, stride=2) # round_mode = 1
        res2a_branch1 = self.res2a_branch1(pool1)
        res2a_branch2a = F.relu(self.res2a_branch2a(pool1))
        res2a_branch2b = self.res2a_branch2b(res2a_branch2a)
        res2a = F.relu(res2a_branch1 + res2a_branch2b)
        res2b_branch2a = F.relu(self.res2b_branch2a(res2a))
        res2b_branch2b = self.res2b_branch2b(res2b_branch2a)
        res2b = F.relu(res2a + res2b_branch2b)
        res2c_branch2a = F.relu(self.res2c_branch2a(res2b))
        res2c_branch2b = self.res2c_branch2b(res2c_branch2a)
        res2c = F.relu(res2b + res2c_branch2b)
        res3a_branch1 = self.res3a_branch1(res2c)
        res3a_branch2a = F.relu(self.res3a_branch2a(res2c))
        res3a_branch2b = self.res3a_branch2b(res3a_branch2a)
        res3a = F.relu(res3a_branch1 + res3a_branch2b)
        res3b_branch2a = F.relu(self.res3b_branch2a(res3a))
        res3b_branch2b = self.res3b_branch2b(res3b_branch2a)
        res3b = F.relu(res3a + res3b_branch2b)
        res3c_branch2a = F.relu(self.res3c_branch2a(res3b))
        res3c_branch2b = self.res3c_branch2b(res3c_branch2a)
        res3c = F.relu(res3b + res3c_branch2b)
        res3d_branch2a = F.relu(self.res3d_branch2a(res3c))
        res3d_branch2b = self.res3d_branch2b(res3d_branch2a)
        res3d = F.relu(res3c + res3d_branch2b)
        res4a_branch1 = self.res4a_branch1(res3d)
        res4a_branch2a = F.relu(self.res4a_branch2a(res3d))
        res4a_branch2b = self.res4a_branch2b(res4a_branch2a)
        res4a = F.relu(res4a_branch1 + res4a_branch2b)
        res4b_branch2a = F.relu(self.res4b_branch2a(res4a))
        res4b_branch2b = self.res4b_branch2b(res4b_branch2a)
        res4b = F.relu(res4a + res4b_branch2b)
        res4c_branch2a = F.relu(self.res4c_branch2a(res4b))
        res4c_branch2b = self.res4c_branch2b(res4c_branch2a)
        res4c = F.relu(res4b + res4c_branch2b)
        res4d_branch2a = F.relu(self.res4d_branch2a(res4c))
        res4d_branch2b = self.res4d_branch2b(res4d_branch2a)
        res4d = F.relu(res4c + res4d_branch2b)
        res4e_branch2a = F.relu(self.res4e_branch2a(res4d))
        res4e_branch2b = self.res4e_branch2b(res4e_branch2a)
        res4e = F.relu(res4d + res4e_branch2b)
        res4f_branch2a = F.relu(self.res4f_branch2a(res4e))
        res4f_branch2b = self.res4f_branch2b(res4f_branch2a)
        res4f = F.relu(res4e + res4f_branch2b)
        res5a_branch1 = self.res5a_branch1(res4f)
        res5a_branch2a = F.relu(self.res5a_branch2a(res4f))
        res5a_branch2b = self.res5a_branch2b(res5a_branch2a)
        res5a = F.relu(res5a_branch1 + res5a_branch2b)
        res5b_branch2a = F.relu(self.res5b_branch2a(res5a))
        res5b_branch2b = self.res5b_branch2b(res5b_branch2a)
        res5b = F.relu(res5a + res5b_branch2b)
        res5c_branch2a = F.relu(self.res5c_branch2a(res5b))
        res5c_branch2b = self.res5c_branch2b(res5c_branch2a)
        res5c = F.relu(res5b + res5c_branch2b)
        rpn_output = F.relu(self.rpn_deconv(res4f))
        rpn_cls_score = self.rpn_cls_score(rpn_output)
        rpn_bbox_pred = self.rpn_bbox_pred(rpn_output)
        rpn_cls_prob = F.softmax(rpn_cls_score.reshape(rpn_cls_score.shape[0], 2, -1, rpn_cls_score.shape[3]), dim=1)
        rpn_cls_prob_reshape = rpn_cls_prob.reshape(rpn_cls_prob.shape[0], 30, -1, rpn_cls_prob.shape[3])
        conv_new = F.relu(self.conv_new(res5c))
        conv_left_kx1 = F.relu(self.conv_left_kx1(conv_new))
        conv_left_1xk = F.relu(self.conv_left_1xk(conv_left_kx1))
        conv_right_1xk = F.relu(self.conv_right_1xk(conv_new))
        conv_right_kx1 = F.relu(self.conv_right_kx1(conv_right_1xk))
        ft_add_left_right = conv_left_1xk + conv_right_kx1
        return rpn_cls_prob_reshape, rpn_bbox_pred, ft_add_left_right

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