#!/bin/bash

# attack
export PYTHONPATH=./code/

# RCNN
# RCNN reg loss only
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_smooth_l1_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_iou_loss

# RCNN cls loss only
python code/attack/nontarget_experiment.py -ds S2TLD720 -t cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -t cls_gt_score_loss

# RCNN reg loss & cls loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_smooth_l1_loss -t cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_smooth_l1_loss -t cls_gt_score_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_iou_loss -t cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_iou_loss -t cls_gt_score_loss

# RPN
# RPN reg loss only
python code/attack/nontarget_experiment.py -ds S2TLD720 -rb box_smooth_l1_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -rb box_iou_loss

# RPN cls only
python code/attack/nontarget_experiment.py -ds S2TLD720 -o cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -o cls_gt_score_loss

# RPN reg loss & cls loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -rb box_smooth_l1_loss -t cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -rb box_smooth_l1_loss -t cls_gt_score_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -rb box_iou_loss -t cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -rb box_iou_loss -t cls_gt_score_loss

# RCNN + RPN
# RCNN + RPN reg loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_smooth_l1_loss -rb box_smooth_l1_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_iou_loss -rb box_iou_loss

# RCNN + RPN cls loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -t cls_nll_loss -o cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -t cls_gt_score_loss -o cls_gt_score_loss

# RCNN + RPN reg + cls
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_smooth_l1_loss -rb box_smooth_l1_loss -t cls_nll_loss -o cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_smooth_l1_loss -rb box_smooth_l1_loss -t cls_gt_score_loss -o cls_gt_score_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_iou_loss -rb box_iou_loss -t cls_nll_loss -o cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_iou_loss -rb box_iou_loss -t cls_gt_score_loss -o cls_gt_score_loss

# RCNN + rec
# RCNN reg + rec cls
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_smooth_l1_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_smooth_l1_loss -c cls_gt_score_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_iou_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_iou_loss -c cls_gt_score_loss

# RCNN cls + rec cls
python code/attack/nontarget_experiment.py -ds S2TLD720 -t cls_nll_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -t cls_gt_score_loss -c cls_gt_score_loss

# RCNN reg + cls, + rec cls
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_smooth_l1_loss -t cls_nll_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_smooth_l1_loss -t cls_gt_score_loss -c cls_gt_score_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_iou_loss -t cls_nll_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_iou_loss -t cls_gt_score_loss -c cls_gt_score_loss

# RPN + rec
# RPN reg + rec cls
python code/attack/nontarget_experiment.py -ds S2TLD720 -rb box_smooth_l1_loss -o cls_nll_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -rb box_smooth_l1_loss -o cls_gt_score_loss -c cls_gt_score_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -rb box_iou_loss -o cls_nll_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -rb box_iou_loss -o cls_gt_score_loss -c cls_gt_score_loss

# RPN cls + rec cls
python code/attack/nontarget_experiment.py -ds S2TLD720 -o cls_nll_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -o cls_gt_score_loss -c cls_gt_score_loss

# RPN reg + cls, + rec cls
python code/attack/nontarget_experiment.py -ds S2TLD720 -rb box_smooth_l1_loss -o cls_nll_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -rb box_smooth_l1_loss -o cls_gt_score_loss -c cls_gt_score_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -rb box_iou_loss -o cls_nll_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -rb box_iou_loss -o cls_gt_score_loss -c cls_gt_score_loss

# RCNN + RPN + rec
# RCNN + RPN, reg + cls, + rec cls
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_smooth_l1_loss -rb box_smooth_l1_loss -t cls_nll_loss -o cls_nll_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_smooth_l1_loss -rb box_smooth_l1_loss -t cls_gt_score_loss -o cls_gt_score_loss -c cls_gt_score_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_iou_loss -rb box_iou_loss -t cls_nll_loss -o cls_nll_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds S2TLD720 -b box_iou_loss -rb box_iou_loss -t cls_gt_score_loss -o cls_gt_score_loss -c cls_gt_score_loss
