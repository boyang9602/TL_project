#!/bin/bash

# attack
export PYTHONPATH=./code/

# RCNN cls loss only
echo python code/attack/target_experiment.py -ds $1 -t cls_nll_loss
python code/attack/target_experiment.py -ds $1 -t cls_nll_loss
