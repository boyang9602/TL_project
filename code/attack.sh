#!/bin/bash

# attack
export PYTHONPATH=./code/

python attack/nontarget_experiment.py -ds S2TLD720 -b box_smooth_l1_loss -t type_nll_loss -c color_nll_loss -o objectiveness_nll_loss
