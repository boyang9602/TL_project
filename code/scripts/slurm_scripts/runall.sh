#!/bin/bash

tar -cf TL_project.tar -C ~/ TL_project/

datasets=(S2TLD720 S2TLD1080)
attack_types=(nontarget target)
epss=(16 8 4 2)
max_iters=(5 10)

for attack_type in ${attack_types[@]}; do
    for dataset in ${datasets[@]}; do
        for eps in ${epss[@]}; do
            for max_iter in ${max_iters[@]}; do
                echo sbatch attack.sh $attack_type $dataset $eps $max_iter
                sbatch attack.sh $attack_type $dataset $eps $max_iter
            done
        done
    done
done
