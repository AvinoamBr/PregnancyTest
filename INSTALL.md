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
### Pretrained model
You can use pretrain weights. copy [yolo weight](https://drive.google.com/file/d/1qIvYesOrsOLhxdvXpcI5WabeRUOx91wN/view?usp=drive_link)
to ```<my_path/window_detector/weights/>```\
### Train Object Detection Model
To Train your objet detection model: 
- Get dataset of pregnant test sticks
- Annotate for: Sticks, window, inner_window.
- Export data in yolov8 format and extract into ```<your_train_object_detection_path>```
- run ```markers_classifier/train.py --data_path <your_train_object_detection_path>```

## Markers Extract Model
Markers model is a classical computer vision model, and is coded with it's params in code / configuration files.

## Markers Classifier Model
Markers classifier model is declared in [markers_model.py](markers_classifier/markers_model.py).\
### Use pretrained weights
You can use pretrained weights.  copy [marker model weight](https://drive.google.com/file/d/1ONR1RJvs0xhDTuqwK-0vPlWCO9OA8fdX/view?usp=drive_link) 
to ```<my_path/markers_classifier/checkpoints/>```\
### Train model
#### prepare training data
- Collect images of positive/negative sticks and save in separate folders
-  Generate patterns by this script:\
- run for positive and negative:
```utils/markers_dataset_generator.py --data_path <path_to_[positive/negative]_sticks> -o <path_to_save_[positive/negative]_markers>```
- you will get .jpg files, with same names as source, representing: a. alligned crop of the window. b. markers.
- file system structure:\
--> positive\
----> markers\
----> window\
--> negative\
----> markers\
----> window

- split to positve and negative to test/train by this script: ```utils/split_train_test.py```\
- You will get this file structure:\
-> train\
---> positive\
---> negative\
-> test\
---> positive\
---> negative\
#### Train 
```markers_classifier/train.py --data_path /home/avinoam/workspace/Salignostics/data/patterns/train```

## Run the model

Now, that all parts of the model are ready:

### Run on single image
Select image file and run:
``` pregnant_detector/pregnant_detector.py -file <file image of stick to detect>```

### Evaluate on test data:
```pregnant_detector/test_markers_model.py --data <path_to_test_pre_prepared_data> ```



