import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import utils
import dataset
from models.src.pipeline import load_pipeline
import torch

def plot_ground_truth(dataset_name, idx):
    ds = dataset.get_dataset(dataset_name)
    case = ds[idx]
    image = utils.bgr2rgb(case['image'])
    boxes = case['boxes']
    colors = case['colors']
    pl = load_pipeline()
    recognizer, shape = pl.classifiers[0]
    means_rec = pl.means_rec
    
    valid, rec, assignments, invalid, _ = pl(case['image'], boxes)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for i, (box, text) in enumerate(zip(boxes, colors)):
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor="none")
        ax.add_patch(rect)
        assignment = assignments[assignments[:, 0] == i][0]
        tl_box = utils.preprocess4rec(image, box, shape, means_rec)
        output = recognizer(tl_box.permute(2, 0, 1).unsqueeze(0))
        rec_vec = rec[assignment[1]]
        rec_color = ["off", "red", "yellow", "green"][torch.argmax(rec_vec)]
        rec_color2 = ["off", "red", "yellow", "green"][torch.argmax(output[0])]
        ax.text(xmin, ymin-2, f'{i}:{text}:{rec_color}:{rec_color2}', c='r', fontsize='large')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot cases.')
    parser.add_argument('--dataset', '-ds', action='store', required=True, choices=['S2TLD720', 'S2TLD1080'], help='the dataset')
    parser.add_argument('--index', '-i', action='store', required=True, type=int, help='the item idx')
    args = parser.parse_args()
    plot_ground_truth(args.dataset, args.index)