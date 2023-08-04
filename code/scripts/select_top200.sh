#!/bin/bash

export PYTHONPATH=./code/

echo python code/eval/select_topk.py -f data/inferences/S2TLD720_pre_recall.txt -o data/inferences/S2TLD720_top200.txt
python code/eval/select_topk.py -f data/inferences/S2TLD720_pre_recall.txt -o data/inferences/S2TLD720_top200.txt
echo python code/eval/select_topk.py -f data/inferences/S2TLD1080_pre_recall.txt -o data/inferences/S2TLD1080_top200.txt
python code/eval/select_topk.py -f data/inferences/S2TLD1080_pre_recall.txt -o data/inferences/S2TLD1080_top200.txt

echo python code/eval/select_topk.py -f data/inferences/S2TLD720_pre_recall.bin -o data/inferences/S2TLD720_top200.bin
python code/eval/select_topk.py -f data/inferences/S2TLD720_pre_recall.bin -o data/inferences/S2TLD720_top200.bin
echo python code/eval/select_topk.py -f data/inferences/S2TLD1080_pre_recall.bin -o data/inferences/S2TLD1080_top200.bin
python code/eval/select_topk.py -f data/inferences/S2TLD1080_pre_recall.bin -o data/inferences/S2TLD1080_top200.bin