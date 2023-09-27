#!/bin/bash

# attack
export PYTHONPATH=./code/

# RCNN
# RCNN reg loss only
echo python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss
python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss

# RCNN reg loss & cls loss
echo python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss -t cls_nll_loss
python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss -t cls_nll_loss

# RPN
# RPN reg loss only
echo python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -rb box_smooth_l1_loss
python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -rb box_smooth_l1_loss

# RPN reg loss & cls loss
echo python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -rb box_smooth_l1_loss -o cls_nll_loss
python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -rb box_smooth_l1_loss -o cls_nll_loss

# RCNN + RPN
# RCNN + RPN reg loss
echo python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss -rb box_smooth_l1_loss
python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss -rb box_smooth_l1_loss

# RCNN + RPN reg + cls
echo python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss -t cls_nll_loss -rb box_smooth_l1_loss -o cls_nll_loss
python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss -t cls_nll_loss -rb box_smooth_l1_loss -o cls_nll_loss

# RCNN + rec
# RCNN reg + rec cls
echo python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss -c cls_nll_loss

# RCNN reg + cls, + rec cls
echo python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss -t cls_nll_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss -t cls_nll_loss -c cls_nll_loss

# RPN + rec
# RPN reg + rec cls
echo python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -rb box_smooth_l1_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -rb box_smooth_l1_loss -c cls_nll_loss

# RCNN + RPN + rec
# RCNN + RPN, reg + rec cls
echo python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss -rb box_smooth_l1_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss -rb box_smooth_l1_loss -c cls_nll_loss

# RCNN + RPN, reg + cls, + rec cls
echo python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss -t cls_nll_loss -rb box_smooth_l1_loss -o cls_nll_loss -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -b box_smooth_l1_loss -t cls_nll_loss -rb box_smooth_l1_loss -o cls_nll_loss -c cls_nll_loss

# rec only
echo python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -c cls_nll_loss
python code/attack/nontarget_experiment.py -ds $1 -e $2 -m $3 -c cls_nll_loss
