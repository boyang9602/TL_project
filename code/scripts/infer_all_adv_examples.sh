#!/bin/bash

datasets=(S2TLD720 S2TLD1080)
attack_types=(target nontarget)

for dataset in ${datasets[@]}; do
    for attack_type in ${attack_types[@]}; do
        path=data/adversarial_results/$dataset/$attack_type
        files=$(ls $path)
        for filename in ${files[@]}; do
            input=$path/$filename
            filename="${filename%.*}"
            output=data/inferences/$dataset/$attack_type/$filename"_detection_results".bin
            echo python code/inference/adv_examples_inference.py -ds $dataset -f $input -o $output
            python code/inference/adv_examples_inference.py -ds $dataset -f $input -o $output
        done
    done
done