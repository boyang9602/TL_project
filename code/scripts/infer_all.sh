#!/bin/bash

export PYTHONPATH=./code/

echo "python code/inference/dataset_inference.py -ds S2TLD720 -o data/inferences/S2TLD720_detection_results2.bin"
python code/inference/dataset_inference.py -ds S2TLD720 -o data/inferences/S2TLD720_detection_results2.bin
echo "python code/inference/dataset_inference.py -ds S2TLD1080 -o data/inferences/S2TLD1080_detection_results2.bin"
python code/inference/dataset_inference.py -ds S2TLD1080 -o data/inferences/S2TLD1080_detection_results2.bin