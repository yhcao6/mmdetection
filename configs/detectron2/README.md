# Benchmark

## Introduction

We adopt the same setting as those in Detectron2 and compare the accuracy and speed. We convert and use the backbone used in Detectron2 into the format MMDetection 1.1 uses.


## Results and Models

### Faster RCNN
|Code base| Backbone  | Lr schd |train time (s/iter)|inference time (s/im)|train mem (GB) | box AP | Download |
|:------:|:-----:|:----:|:-----:|:-----:|:-------:|:------:|:--------:|
|MMDet| R-50      | 1x      |     |   |   | 38.2  |     -    |
|Detectron2| R-50      | 1x      |     |   |   | 37.9   |     -    |
|MMDet | R-50      | 3x      |     |   |   | 40.2   |     -    |
|Detectron2| R-50      | 3x      |     |   |   | 40.2   |     -    |

### Mask RCNN
|Code base| Backbone  | Lr schd |train time (s/iter)|inference time (s/im)|train mem (GB) | box AP | mask AP | Download |
|:------:|:-----:|:----:|:-----:|:-----:|:-------:|:------:|:------:|:------:|
|MMDet| R-50      | 1x      |     |   |   | 38.6  |35.2|     -    |
|Detectron2| R-50      | 1x      |     |   |   | 38.6   |35.2|     -    |
|MMDet | R-50      | 3x      |     |   |   | |  |   -    |
|Detectron2| R-50      | 3x      |     |   |   | 41.0   |37.2|     -    |

### RetinaNet
|Code base| Backbone  | Lr schd |train time (s/iter)|inference time (s/im)|train mem (GB) | box AP | Download |
|:------:|:-----:|:----:|:-----:|:-----:|:-------:|:------:|:--------:|
|MMDet| R-50      | 1x      |     |   |   | 36.7  |     -    |
|Detectron2| R-50      | 1x      |     |   |   | 36.5   |     -    |
|MMDet | R-50      | 3x      |     |   |   ||     -    |
|Detectron2| R-50      | 3x      |     |   |   ||     -    |
