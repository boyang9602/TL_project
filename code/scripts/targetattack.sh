#!/bin/bash

# attack
export PYTHONPATH=./code/

# RCNN
# RCNN reg loss only
echo python code/attack/target_experiment.py -ds $1 -b box_smooth_l1_loss
python code/attack/target_experiment.py -ds $1 -b box_smooth_l1_loss
echo python code/attack/target_experiment.py -ds $1 -b box_iou_loss
python code/attack/target_experiment.py -ds $1 -b box_iou_loss

# RCNN cls loss only
echo python code/attack/target_experiment.py -ds $1 -t cls_nll_loss
python code/attack/target_experiment.py -ds $1 -t cls_nll_loss
echo python code/attack/target_experiment.py -ds $1 -t cls_gt_score_loss
python code/attack/target_experiment.py -ds $1 -t cls_gt_score_loss

# RCNN reg loss & cls loss
echo python code/attack/target_experiment.py -ds $1 -b box_smooth_l1_loss -t cls_nll_loss
python code/attack/target_experiment.py -ds $1 -b box_smooth_l1_loss -t cls_nll_loss
echo python code/attack/target_experiment.py -ds $1 -b box_smooth_l1_loss -t cls_gt_score_loss
python code/attack/target_experiment.py -ds $1 -b box_smooth_l1_loss -t cls_gt_score_loss
echo python code/attack/target_experiment.py -ds $1 -b box_iou_loss -t cls_nll_loss
python code/attack/target_experiment.py -ds $1 -b box_iou_loss -t cls_nll_loss
echo python code/attack/target_experiment.py -ds $1 -b box_iou_loss -t cls_gt_score_loss
python code/attack/target_experiment.py -ds $1 -b box_iou_loss -t cls_gt_score_loss
