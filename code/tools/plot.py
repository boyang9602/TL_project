import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import tools.utils as utils
import dataset
from models.pipeline import load_pipeline
import torch
TL_TYPE_TEXTS = ['VERT', 'QUAD', 'HORI']
REC_COLORS = ["off", "red", "yellow", "green"]
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

def viz_pipeline_results(image, valid_detections=None, recognitions=None, assignments=None, invalid_detections=None, projections=None, fig_width=20.48, fig_height=15.36):
    """
      results is a n*13 tensor, n TL instances, 
      for each instance,
      index 0: pass
      index 1-4: bbox
      index 5-8: TL type softmax
      index 9-end: classification softmax
    """
    fig, ax = plt.subplots(1)
    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_height)
    ax.imshow(image)
    if valid_detections != None:
        for i, result in enumerate(valid_detections):
            xmin = int(result[1].item())
            ymin = int(result[2].item())
            xmax = int(result[3].item())
            ymax = int(result[4].item())
            edge_color = 'r'
            if i in assignments[:, 1]:
                edge_color = 'g'
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=edge_color, facecolor="none")
            ax.add_patch(rect)
            tl_type = torch.argmax(result[5:9]).item() - 1
            assert tl_type >= 0 # tl_type should be not unknown or the classification scores should be all -1s. 
            tl_color = torch.argmax(recognitions[i])
            ax.text(xmin, ymin-2, f'{TL_TYPE_TEXTS[tl_type]}:{REC_COLORS[tl_color]}', c='r', fontsize='large')
    if invalid_detections != None:
        for i, result in enumerate(invalid_detections):
            xmin = int(result[1].item())
            ymin = int(result[2].item())
            xmax = int(result[3].item())
            ymax = int(result[4].item())
            edge_color = 'r'
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=edge_color, facecolor="none")
            ax.add_patch(rect)
    if projections != None:
        for projection in projections:
            # print(projection)
            rect = patches.Rectangle((projection.x, projection.y), projection.w, projection.h, linewidth=2, edgecolor='c', facecolor="none")
            ax.add_patch(rect)
            coor = utils.crop(image.shape, projection)
            rect = patches.Rectangle((coor[0], coor[2]), coor[1] - coor[0], coor[3] - coor[2], linewidth=2, edgecolor='y', facecolor="none")
            ax.add_patch(rect)
    return fig, ax

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot cases.')
    parser.add_argument('--dataset', '-ds', action='store', required=True, choices=['S2TLD720', 'S2TLD1080'], help='the dataset')
    parser.add_argument('--index', '-i', action='store', required=True, type=int, help='the item idx')
    args = parser.parse_args()
    plot_ground_truth(args.dataset, args.index)