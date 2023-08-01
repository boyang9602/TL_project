# TL project

This is a project for doing adversarial attack against Baidu Apollo's TL detection, selection and recognition pipeline. 

## File lists

1. code/models/: the replicated models as well as the weights 
2. code/attack/: the adversarial attack related code files
3. code/eval/: the evaluation code
4. recognizer_eval: the code for test the robustness of/attack the recognizer standalone
5. code/tools/: the utilites used in other modules.

## Example code
You need to set the PYTHONPATH first. 
1. Open your terminal,
2. Go to the root of the project
3. Command

`export PYTHONPATH=./code/`


To set up the detection, selection and recognition pipeline 

```python
from models.src.pipeline import load_pipeline
torch.manual_seed(42) # for reproductibility, not necessary
device = "cuda" if torch.cuda.is_available() else "cpu"
pl = load(device=device)
valid_detections, recognitions, assignments, invalid_detections, rpn_data = pl(image, boxes) # image is the image file in bgr format, boxes is a list of single box, which is [xmin, ymin, xmax, ymax].
```
### Pipeline output
1. valid_detections is a n * 9 tensor. The first column is useless in this project. 1:5 are the bounding boxes, 5:9 are the TL type scores vector.
2. recognitions are the recognition scores vector.
3. assignments is a n * 2 tensor. Each row is match between the [projection](https://github.com/ApolloAuto/apollo/blob/v7.0.0/docs/specs/traffic_light.md#pre-process) and the valid detection. The first col is the idx of a projection of TLs and the second col is the idx of a valid detection. 
4. invalid_detections are discarded in Apollo and rpn_data is the intermediate data in RPN layer. They are used for the attack in this project.



## Dataset.
Please follow this github repo to get the dataset. 
https://github.com/Thinklab-SJTU/S2TLD

tools/dataset.py is a simple dataset loader. It only supports the S2TLD dataset for now. It needs to generates a filelist.txt before using. Open your terminal and command

`python tools/dataset.py <S2TLD 720 * 1280 path> <S2TLD 1080 * 1920 path>`

To use the dataset,

```python
from tools.dataset import get_dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
ds = get_dataset(<'S2TLD720'|'S2TLD1080'>, device=device)

item = ds[idx]
'''
{
    'image': ..., bgr format
    'boxes': ...,
    'colors': ...,
    ...

}
'''
```