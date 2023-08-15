#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--output', '-o', action='store', required=False, default='code/scripts/targetattack.sh', help='the output attack bash script location.')
args = parser.parse_args()
sh = """#!/bin/bash

# attack
export PYTHONPATH=./code/
"""

cls_loss_list = [
    'cls_nll_loss',
    'cls_gt_score_loss'
]

command_prefix = f'python code/attack/target_experiment.py -ds $1'

def make_command(commnad):
    return f'echo {commnad}\n{commnad}\n'

sh += '''
# RCNN cls loss only
'''
for cls_loss in cls_loss_list:
    sh += make_command(command_prefix + f' -t {cls_loss}')

with open(args.output, 'w') as f:
    f.write(sh)