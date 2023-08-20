#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--max_iter', '-m', action='store', required=False, default=5, help="the max allowed iteration of attack")
parser.add_argument('--eps', '-e', action='store', required=False, default=16, help="the max allowed iteration of attack")
parser.add_argument('--output', '-o', action='store', required=False, default=None, help='the output attack bash script location.')
args = parser.parse_args()
sh = """#!/bin/bash

# attack
export PYTHONPATH=./code/
"""

box_loss_list = [
    'box_smooth_l1_loss',
    'box_iou_loss'
]

cls_loss_list = [
    'cls_nll_loss',
    # 'cls_gt_score_loss'
]

command_prefix = f'python code/attack/nontarget_experiment.py -ds $1 -m {args.max_iter} -e {args.eps}'

def make_command(commnad):
    return f'echo {commnad}\n{commnad}\n'

sh += '''
# RCNN
# RCNN reg loss only
'''
for box_loss in box_loss_list:
    sh += make_command(command_prefix + f' -b {box_loss}')
sh += '''
# RCNN cls loss only
'''
for cls_loss in cls_loss_list:
    sh += make_command(command_prefix + f' -t {cls_loss}')
sh += '''
# RCNN reg loss & cls loss
'''
for box_loss in box_loss_list:
    for cls_loss in cls_loss_list:
        sh += make_command(command_prefix + f' -b {box_loss} -t {cls_loss}')

sh += '''
# RPN
# RPN reg loss only
'''
for box_loss in box_loss_list:
    sh += make_command(command_prefix + f' -rb {box_loss}')
sh += '''
# RPN cls only
'''
for cls_loss in cls_loss_list:
    sh += make_command(command_prefix + f' -o {cls_loss}')
sh += '''
# RPN reg loss & cls loss
'''
for box_loss in box_loss_list:
    for cls_loss in cls_loss_list:
        sh += make_command(command_prefix + f' -rb {box_loss} -o {cls_loss}')

sh += '''
# RCNN + RPN
# RCNN + RPN reg loss
'''
for box_loss in box_loss_list:
    sh += make_command(command_prefix + f' -b {box_loss} -rb {box_loss}')
sh += '''
# RCNN + RPN cls loss
'''
for cls_loss in cls_loss_list:
    sh += make_command(command_prefix + f' -t {cls_loss} -o {cls_loss}')
sh += '''
# RCNN + RPN reg + cls
'''
for box_loss in box_loss_list:
    for cls_loss in cls_loss_list:
        sh += make_command(command_prefix + f' -b {box_loss} -t {cls_loss} -rb {box_loss} -o {cls_loss}')

sh += '''
# RCNN + rec
# RCNN reg + rec cls
'''
for box_loss in box_loss_list:
    for cls_loss in cls_loss_list:
        sh += make_command(command_prefix + f' -b {box_loss} -c {cls_loss}')
sh += '''
# RCNN cls + rec cls
'''
for cls_loss in cls_loss_list:
    sh += make_command(command_prefix + f' -t {cls_loss} -c {cls_loss}')
sh += '''
# RCNN reg + cls, + rec cls
'''
for box_loss in box_loss_list:
    for cls_loss in cls_loss_list:
        sh += make_command(command_prefix + f' -b {box_loss} -t {cls_loss} -c {cls_loss}')

sh += '''
# RPN + rec
# RPN reg + rec cls
'''
for box_loss in box_loss_list:
    for cls_loss in cls_loss_list:
        sh += make_command(command_prefix + f' -rb {box_loss} -c {cls_loss}')
sh += '''
# RPN cls + rec cls
'''
for cls_loss in cls_loss_list:
    sh += make_command(command_prefix + f' -o {cls_loss} -c {cls_loss}')
sh += '''
# RPN reg + cls, + rec cls
'''
for box_loss in box_loss_list:
    for cls_loss in cls_loss_list:
        sh += make_command(command_prefix + f' -rb {box_loss} -o {cls_loss} -c {cls_loss}')

sh += '''
# RCNN + RPN + rec
# RCNN + RPN, reg + rec cls
'''
for box_loss in box_loss_list:
    for cls_loss in cls_loss_list:
        sh += make_command(command_prefix + f' -b {box_loss} -rb {box_loss} -c {cls_loss}')
sh += '''
# RCNN + RPN + rec
# RCNN + RPN, cls + rec cls
'''
for cls_loss in cls_loss_list:
    sh += make_command(command_prefix + f' -t {cls_loss} -o {cls_loss} -c {cls_loss}')
sh += '''
# RCNN + RPN, reg + cls, + rec cls
'''
for box_loss in box_loss_list:
    for cls_loss in cls_loss_list:
        sh += make_command(command_prefix + f' -b {box_loss} -t {cls_loss} -rb {box_loss} -o {cls_loss} -c {cls_loss}')

output = args.output
if output is None:
    output = f'code/scripts/nontargetattack_{args.eps}_{args.max_iter}.sh'
with open(output, 'w') as f:
    f.write(sh)
