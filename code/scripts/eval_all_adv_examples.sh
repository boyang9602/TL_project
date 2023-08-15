#!/bin/bash

datasets=(S2TLD720 S2TLD1080)
attack_types=(target nontarget)

for dataset in ${datasets[@]}; do
    for attack_type in ${attack_types[@]}; do
        path=data/inferences/$dataset/$attack_type
        files=$(ls $path)
        for filename in ${files[@]}; do
            input=data/inferences/$dataset/$attack_type/$filename
            echo python code/eval/eval_adv_examples.py -ds $dataset -f $input
            python code/eval/eval_adv_examples.py -ds $dataset -f $input
        done
    done
done