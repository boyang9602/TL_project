import torch
import time

from models.src.pipeline import load_pipeline
from models.src.pipeline_v2 import load_pipeline as load_pipeline2

from tools.dataset import get_dataset

pl = load_pipeline('cuda')
pl2 = load_pipeline2('cuda')

ds = get_dataset('S2TLD720', 'cuda')

results1 = []
results2 = []

with torch.no_grad():
    t1 = time.perf_counter()
    for i in range(len(ds)):
        item = ds[i]
        results1.append(pl(item['image'], item['boxes']))
    t2 = time.perf_counter()
    print(f"S2TLD720, v1, Run all the cases in {t2-t1:0.4f} seconds!")
    for i in range(len(ds)):
        item = ds[i]
        results2.append(pl2(item['image'], item['boxes']))
    t3 = time.perf_counter()
    print(f"S2TLD720, v2, Run all the cases in {t3-t2:0.4f} seconds!")

correct = []
incorrect = []
for idx, (r1, r2) in enumerate(zip(results1, results2)):
    valid1, reco1, ass1, invalid1, rpn1 = r1
    valid2, reco2, ass2, invalid2, rpn2 = r2
    if valid1.shape == valid2.shape and torch.allclose(valid1, valid2, atol=1e-4) \
        and reco1.shape == reco2.shape and torch.allclose(reco1, reco2, atol=1e-4) \
        and ass1.shape == ass2.shape and torch.allclose(ass1, ass2, atol=1e-4) \
        and invalid1.shape == invalid2.shape and torch.allclose(invalid1, invalid2, atol=1e-4) \
        and rpn1.shape == rpn2.shape and torch.allclose(rpn1, rpn2, atol=1e-4):
        correct.append(idx)
    else:
        incorrect.append(idx)

print(len(correct), len(incorrect))
print(correct)
print(incorrect)

ds = get_dataset('S2TLD1080', 'cuda')

results1 = []
results2 = []

with torch.no_grad():
    t1 = time.perf_counter()
    for i in range(len(ds)):
        item = ds[i]
        results1.append(pl(item['image'], item['boxes']))
    t2 = time.perf_counter()
    print(f"S2TLD1080, v1, Run all the cases in {t2-t1:0.4f} seconds!")
    for i in range(len(ds)):
        item = ds[i]
        results2.append(pl2(item['image'], item['boxes']))
    t3 = time.perf_counter()
    print(f"S2TLD1080, v2, Run all the cases in {t3-t2:0.4f} seconds!")

correct = []
incorrect = []
for idx, (r1, r2) in enumerate(zip(results1, results2)):
    valid1, reco1, ass1, invalid1, rpn1 = r1
    valid2, reco2, ass2, invalid2, rpn2 = r2
    if valid1.shape == valid2.shape and torch.allclose(valid1, valid2, atol=1e-4) \
        and reco1.shape == reco2.shape and torch.allclose(reco1, reco2, atol=1e-4) \
        and ass1.shape == ass2.shape and torch.allclose(ass1, ass2, atol=1e-4) \
        and invalid1.shape == invalid2.shape and torch.allclose(invalid1, invalid2, atol=1e-4) \
        and rpn1.shape == rpn2.shape and torch.allclose(rpn1, rpn2, atol=1e-4):
        correct.append(idx)
    else:
        incorrect.append(idx)

print(len(correct), len(incorrect))
print(correct)
print(incorrect)