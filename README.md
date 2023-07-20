# TL project

This is a project for doing adversarial attack against Baidu Apollo's TL detection, selection and recognition pipeline. 

## File lists

```
detector.py: the TL detector code which detects the TL boxes and their types.
selector.py: the TL selection code, which uses Hungarian Algorithm to select TLs detected by TL detector.
recognizer.py: the color recognizer which recognizes the color of detected TL boxes.
pipeline.py: the code that combines the detector, selector and recognizer, following the Baidu Apollo's code.
utils.py: the utils functions, including reading annotation files, visualizing TL detection and recognition results, NMS, etc. Most functions are self-explained and commented. 
hungarian_optimizer: the replicated hungarian algorithm implementation of Baidu Apollo's Hungarian Optimizer. https://github.com/b0yan9/hungarian_optimizer 
adversarial.py: the adversarial attack code and objective function. All the parameters are commented and explained in the code.
main.py: the code to infer the given dataset and save the detection results into a file. 
eval.py: the code to evaluate the results of inference on S2TLD dataset. 
models/: this folder contains the weights files of the networks. The example code below shows how to use them. 
top200perfect.py: the code to find 200 pictures whose avg IoUs are top among all of the perfect cases. 
perfect_precision3.bin: the list of picture where all the detected TLs are correct, i.e., they have a 100% precision for detection, setting IoU threshold to 0.5.
perfect_recall3.bin: the list of picture where all the ground truth TLs are detected, i.e., they have a 100% recall for detection, setting IoU threshold to 0.5.
perfect3.bin: the list of pictures which have a 100% precision and recall for detection, setting IoU threshold to 0.5.
top200avg.bin: the 200 pictures which are top 200 avg IoUs. 
```

## Example code
To set up the detection, selection and recognition pipeline 
```python
import cv2
import time
import pickle
import hungarian_optimizer
import utils
import torch
from adversarial import adversarial, create_objective, objective
from detector import TFModel
from recognizer import Recognizer
from pipeline import Pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device is {device}')
TL_TYPE_TEXTS = ['VERT', 'QUAD', 'HORI']
REC_COLORS = ["Black", "Red", "Yellow", "Green"]
means_det = torch.Tensor([102.9801, 115.9465, 122.7717]).to(device)
means_rec = torch.Tensor([69.06, 66.58, 66.56]).to(device)

# create detector
detector = TFModel()
detector.load_state_dict(torch.load('models/tl.torch'))
detector = detector.to(device)
detector.eval();

# create recognizor
quad_pool_params = {'kernel_size': (4, 4), 'stride': (4, 4)}
hori_pool_params = {'kernel_size': (2, 6), 'stride': (2, 6)}
vert_pool_params = {'kernel_size': (6, 2), 'stride': (6, 2)}
quad_recognizer = Recognizer(quad_pool_params)
hori_recognizer = Recognizer(hori_pool_params)
vert_recognizer = Recognizer(vert_pool_params)

quad_recognizer.load_state_dict(torch.load('models/quad.torch'))
quad_recognizer = quad_recognizer.to(device)
quad_recognizer.eval();

hori_recognizer.load_state_dict(torch.load('models/hori.torch'))
hori_recognizer = hori_recognizer.to(device)
hori_recognizer.eval();

vert_recognizer.load_state_dict(torch.load('models/vert.torch'))
vert_recognizer = vert_recognizer.to(device)
vert_recognizer.eval();
classifiers = [(vert_recognizer, (96, 32, 3)), (quad_recognizer, (64, 64, 3)), (hori_recognizer, (32, 96, 3))]

# create selector algorithm
ho = hungarian_optimizer.HungarianOptimizer()

# create the pipeline
pipeline = Pipeline(detector, classifiers, ho, means_det, means_rec)
```

To run the inference
```python
image_file = <file path>
annot_file = <annotation path>
# read the image
image = torch.from_numpy(cv2.imread(image_file)).to(device)
# read the ground truth labels
boxes = utils.readxml(annot_file)
# inference, detection is the detection result, assignment is the matching result.
detection, assignments = pipeline(image, boxes)
```

To do the adversarial attack:
```python
# create the objective function, with a max budget of 5 steps.
obj_fn = create_objective(objective, 5)
# run the adversarial attack, the adv_img is the found adversarial example.
adv_img = adversarial(pipeline, image, boxes, obj_fn, 3, 16)
```

To load the bin files:
```python
# the loaded data is a tensor with the shape (200, 2), the 2nd col is the id of the image and picture file. Ignore the first col.
# e.g., top200_cases[0, 1] gets the case with the highest avg IoU. If the number is 66, then it corresponds to the image 000066.jpg and the annotation 000066.xml from the S2TLD dataset.
with open('top200avg.bin', 'rb') as f:
    top200_cases = pickle.load(f)
```

## Dataset.
Please follow this github repo to get the dataset. 
https://github.com/Thinklab-SJTU/S2TLD