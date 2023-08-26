#!/bin/bash

#SBATCH --account=b_yang20
#SBATCH --mem=10G
#SBATCH --mail-type=FAIL,END
#SBATCH --gpus=10gb:1
#SBATCH --mail-user=boyang9602@gmail.com

attack_type=$1
dataset=$2
eps=$3
max_iter=$4

cp TL_project.tar $TMPDIR/
cd $TMPDIR/
tar -xf TL_project.tar
cp ~/$dataset".tar" TL_project/
cd TL_project/
tar -xf $dataset".tar"

export PYTHONPATH=./code/
source /etc/profile.d/modules.sh
module load cuda
source ~/.bashrc
conda activate dl

bash ./code/scripts/"$attack_type"attack.sh $dataset $eps $max_iter
mkdir -p ~/data/adversarial_results/
cp -r data/adversarial_results/* ~/data/adversarial_results/

path=data/adversarial_results/$dataset/nontarget
files=$(ls $path)
for filename in ${files[@]}; do
    input=$path/$filename
    filename="${filename%.*}"
    output=data/inferences/$dataset/nontarget/$filename"_detection_results".bin
    echo python code/inference/adv_examples_inference.py -ds $dataset -f $input -o $output
    python code/inference/adv_examples_inference.py -ds $dataset -f $input -o $output
done

mkdir -p ~/data/adversarial_results/
cp -r data/inferences/* ~/data/inferences/
