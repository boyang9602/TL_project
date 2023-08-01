#!/bin/bash

export PYTHONPATH=./code/

echo "python code/eval/eval_prec_recall.py -d data/inferences/S2TLD720_detection_results.bin -o data/inferences/S2TLD720_pre_recall.txt"
python code/eval/eval_prec_recall.py -d data/inferences/S2TLD720_detection_results.bin -o data/inferences/S2TLD720_pre_recall.txt
echo "python code/eval/eval_prec_recall.py -d data/inferences/S2TLD1080_detection_results.bin -o data/inferences/S2TLD1080_pre_recall.txt"
python code/eval/eval_prec_recall.py -d data/inferences/S2TLD1080_detection_results.bin -o data/inferences/S2TLD1080_pre_recall.txt

echo "python code/eval/eval_prec_recall.py -d data/inferences/S2TLD720_detection_results.bin -o data/inferences/S2TLD720_pre_recall.bin"
python code/eval/eval_prec_recall.py -d data/inferences/S2TLD720_detection_results.bin -o data/inferences/S2TLD720_pre_recall.bin
echo "python code/eval/eval_prec_recall.py -d data/inferences/S2TLD1080_detection_results.bin -o data/inferences/S2TLD1080_pre_recall.bin"
python code/eval/eval_prec_recall.py -d data/inferences/S2TLD1080_detection_results.bin -o data/inferences/S2TLD1080_pre_recall.bin