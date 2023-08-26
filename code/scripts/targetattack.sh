#!/bin/bash

# attack
export PYTHONPATH=./code/

# RCNN cls loss fabricate
echo python code/attack/target_experiment.py -ds $1 -e $2 -m $3 -f cls_nll_loss
python code/attack/target_experiment.py -ds $1 -e $2 -m $3 -f cls_nll_loss

# RCNN cls loss remove
echo python code/attack/target_experiment.py -ds $1 -e $2 -m $3 -r cls_nll_loss
python code/attack/target_experiment.py -ds $1 -e $2 -m $3 -r cls_nll_loss

# RCNN cls loss fabricate +  remove
echo python code/attack/target_experiment.py -ds $1 -e $2 -m $3 -f cls_nll_loss -r cls_nll_loss
python code/attack/target_experiment.py -ds $1 -e $2 -m $3 -f cls_nll_loss -r cls_nll_loss
