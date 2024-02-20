from ultralytics import YOLO
import os

data_path = '/home/avinoam/workspace/Salignostics/data/roboflow_20240219/data.yaml'
n_epochs = 100
bs = 4
gpu_id = 0
verbose = True
rng = 0
validate = True

# Specify the save directory for training runs
save_dir = '/home/avinoam/workspace/Salignostics/pythonProject/bounding_box_detection/weights/'
os.makedirs(save_dir, exist_ok=True)
cp = "/home/avinoam/Desktop/autobrains/DL_Engineer/assignment_files/runs/detect/train13/weights/best.pt"

model = YOLO('yolov8n.pt')
model.load(cp)
results = model.train(
    data=data_path,
    epochs=n_epochs,
    batch=bs,
    augment=True,
    lr0=0.05,
    verbose=verbose,
    seed=rng,
    val=validate,
    save_dir=save_dir
)