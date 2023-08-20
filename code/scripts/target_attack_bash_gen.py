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

command_prefix = f'python code/attack/target_experiment.py -ds $1'

def make_command(commnad):
    return f'echo {commnad}\n{commnad}\n'

sh += '''
# RCNN cls loss fabricate
'''
sh += make_command(command_prefix + f' -f cls_nll_loss')
sh += '''
# RCNN cls loss remove
'''
sh += make_command(command_prefix + f' -r cls_nll_loss')
sh += '''
# RCNN cls loss fabricate +  remove
'''
sh += make_command(command_prefix + f' -f cls_nll_loss -r cls_nll_loss')

output = args.output
if output is None:
    output = f'code/scripts/targetattack_{args.eps}_{args.max_iter}.sh'
with open(output, 'w') as f:
    f.write(sh)
