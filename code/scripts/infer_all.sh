#!/bin/bash

export PYTHONPATH=./code/

echo "python code/inference/dataset_inference.py -ds S2TLD720"
python code/inference/dataset_inference.py -ds S2TLD720
echo "python code/inference/dataset_inference.py -ds S2TLD1080"
python code/inference/dataset_inference.py -ds S2TLD1080