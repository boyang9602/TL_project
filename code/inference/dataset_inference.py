import time
import pickle
import torch
from tools.dataset import get_dataset
from models.src.pipeline import load_pipeline
import argparse
import os

def infer_all(model, ds):
    results = []
    with torch.no_grad():
        t1 = time.perf_counter()
        for idx in range(len(ds)):
            item = ds[idx]
            image = item['image']
            boxes = item['boxes']
            colors = item['colors']
            try:
                valid_detections, recognitions, assignments, invalid_detections, _ = model(image, boxes)
            except:
                print(idx)
                raise
            results.append((valid_detections, recognitions, assignments, invalid_detections, boxes, colors))
        t2 = time.perf_counter()
        print(f"Run all the cases in {t2-t1:0.4f} seconds!")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference the given dataset.')
    parser.add_argument('--output', '-o', action='store', default=None, required=False, help='the binary file name of the detection results to be written to. Default is the data/inferences/{dataset}_detection_results.bin')
    parser.add_argument('--dataset', '-ds', action='store', required=True, choices=['S2TLD720', 'S2TLD1080'], help='the dataset where the adversarial examples are from')
    args = parser.parse_args()

    if args.output is not None:
        output = args.output
    else:
        output = f'data/inferences/{args.dataset}_detection_results.bin'
    idx = output.rfind('/')
    if idx != -1:
        folder = output[:idx]
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except:
                print('Output folder not exist!!!')
                exit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pipeline(device)
    ds = get_dataset(args.dataset, device)
    results = infer_all(model, ds)

    with open(output, 'wb') as f:
        pickle.dump(results, f)
