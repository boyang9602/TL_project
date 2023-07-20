import time
import sys
import pickle
import torch
from dataset import get_dataset
from pipeline import load_pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

def infer_all(model, ds):
    results = []
    with torch.no_grad():
        t1 = time.perf_counter()
        for idx in range(len(ds)):
            item = ds[idx]
            image = item['image']
            boxes = item['boxes']
            colors = item['colors']
            valid_detections, recognitions, assignments, invalid_detections = model(image, boxes)
            results.append((valid_detections, recognitions, assignments, invalid_detections, boxes, colors))
        t2 = time.perf_counter()
        print(f"Run all the cases in {t2-t1:0.4f} seconds!")
    return results

if __name__ == '__main__':
    model = load_pipeline(device)
    ds = get_dataset(sys.argv[1], device)
    results = infer_all(model, ds)
    with open(f'{sys.argv[1]}_detection_results.bin', 'wb') as f:
        pickle.dump(results, f)
