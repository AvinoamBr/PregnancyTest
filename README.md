# Diagnotsitc Stick Calssifier

This repositoty contain an end to end application
that receive image of detection stick, and return Positve/Negative status for the detection.

## Three Steps Model

A - Object detection:  Locate the stick and the exact window were markers are expected. Then
aligning the window, so that it is directed horizontally.

B - Markers Detection - Running pattern match algorithms to enhance pixels were it might be a marker.

C - Classification - Given the map of the potential markers, and their intensity,\
classifying the test to be positive or negative. 

# Visual example
|                 | Positive                                                        | Negative                                                         |
|-----------------|-----------------------------------------------------------------|------------------------------------------------------------------|
| source          | ![](images_for_readme/True_Positive/20230616_070451_source.jpg) | ![](images_for_readme/True_Negative/20230615_125100_source.jpg)  |
| alligned window | ![](images_for_readme/True_Positive/20230616_070451_window.jpg) | ![](images_for_readme/True_Negative/20230615_125100_window.jpg)  |
| markers         | ![](images_for_readme/True_Positive/20230616_070451_markers.jpg) | ![](images_for_readme/True_Negative/20230615_125100_markers.jpg) |
|model response   | Positive, confidence = 1.0                                      | Negative, confidence = 0.92 | 

## installation and usage
 Please follow instructions in INSTALL.md 
