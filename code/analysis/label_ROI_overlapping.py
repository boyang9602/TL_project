'''
Separate the cases by if the ROI overlaps with each other,
output:
for each picture, each labelled TL, tell if its ROI overlaps with others
'''
import json
import torch
from tools.dataset import readxml2
from tools.utils import boxes2projections, crop, IoU_multi

def get_annotation_filename(line, dataset):
    if dataset == 'S2TLD720':
        folder, filename = line.strip().split(',')
        return f'{dataset}/{folder}/Annotations/{filename}.xml'
    else:
        filename = line.strip()
        return f'{dataset}/Annotations/{filename}.xml'


datasets = ['S2TLD720', 'S2TLD1080']

for dataset in datasets:
    overlap_info = []
    with open(f'{dataset}/filelist.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            curr = []
            overlap_info.append(curr)
            bboxes, colors = readxml2(get_annotation_filename(line, dataset))
            projections = boxes2projections(bboxes)
            ROIs = []
            for projection in projections:
                xl, xr, yt, yb = crop((720, 1280), projection)
                ROIs.append([xl, yt, xr, yb])
            ious = IoU_multi(torch.tensor(ROIs).reshape(-1, 4), torch.tensor(ROIs).reshape(-1, 4))
            for iou in ious:
                overlapped_ids, = torch.nonzero(iou, as_tuple=True)
                curr.append(overlapped_ids.numpy().tolist())
    with open(f'{dataset}/ROI_overlap_info.json', 'w') as f:
        json.dump(overlap_info, f)
