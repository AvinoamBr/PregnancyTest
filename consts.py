# ENERAl

class TextColors(object):
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

text_color = TextColors()


# yolo model arguments
YOLO_CP = "/home/avinoam/Desktop/autobrains/DL_Engineer/assignment_files/runs/detect/train14/weights/best.pt"

STICKS = 'Sticks'
INNER_WINDOW = "inner_window"
WINDOW = 'window'
OBJECT_DETECTION_CLASS_NAMES = [STICKS, INNER_WINDOW, WINDOW]

# markers model arguments
CLASS_NAMES = ['positive','negative']
TRAIN_DATA_PATH = "../data/patterns/train"
TEST_DATA_PATH = "../data/patterns/test"

EPOCHS = 100
INPUT_IMAGE_SHAPE = (80,80)

CHECKPOINT_PATH = "markers_classifier/checkpoints/"

CLASS_CONF_THRESH = 0.8
