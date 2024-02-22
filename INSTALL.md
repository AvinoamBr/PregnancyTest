# Installation

## Set python Environment

Download and extract repository into ```<my_path\>```

set python vitual environment, and source into it:
```commandline
python -m venv venv
source venv/bin/activate
```
Install python requirements
```commandline
pip3 install -r requirements.txt
```

## Object Detection Model
Object detection is based on [YOLOV8](https://github.com/ultralytics/ultralytics).\
You can use pretrain weights. copy *---add link---* to ```<my_path\yolo_weights\>```\
### Train Object Detection Model
To Train your own model: *---add explain here---*

## Markers Extract Model
Markers model is a classical computer vision model, and is coded with it's params in code / configuration files.

## Markers Classifier Model
Markers classifier model is declared in [markers_model.py](markers_model.py).\
You can use pretrained weights.  copy *---add link---* to ```<my_path\marker_model_weights\>```\



