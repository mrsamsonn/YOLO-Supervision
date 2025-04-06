# YOLO-Supervision

This project demonstrates YOLO-based object detection, optimized for real-time supervision using a custom dataset. It is designed to work efficiently even on low-end hardware using OpenVino. The system detects objects, performs inventory checks, and handles movement-dependent logic.

## Table of Contents
- [Overview](#overview)
- [YouTube Shorts](#youtube-shorts)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Output](#output)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [License](#license)

## Overview

The YOLO-Supervision project utilizes YOLOv8 for real-time object detection and tracking. With OpenVino optimization, the system can be deployed on low-end hardware with minimal performance loss. The project handles custom datasets and performs tasks such as inventory checking and movement detection.

## YouTube Shorts

Here are a few YouTube Shorts that showcase the algorithm in action:

- [Learning the wheel - 05/11/24](https://www.youtube.com/shorts/5BKpS3-Ndds)
- [Training on Custom Dataset - 05/11/24](https://www.youtube.com/shorts/2pcbaSQviZU)
- [Building Custom Dataset - 05/14/24](https://www.youtube.com/shorts/XflusQ3jpDI)

## Algorithm Explanation

### 1. **Data Preparation**

The first step involves preparing your custom dataset. The dataset needs to be annotated and split into training and validation sets. The `train_model.py` script handles this process, where you define the path to your dataset and the annotations format.

```python
import os
from yolov5.utils import dataset

def prepare_dataset(data_path, annotations_path):
    # Load and prepare dataset for training
    dataset = dataset.CustomDataset(data_path, annotations_path)
    return dataset
```

2. Model Setup and Optimization

We use the YOLOv8 model, and OpenVino optimization is applied to improve inference performance on lower-end hardware. Here’s how you initialize and optimize the model:

```python
from openvino.inference_engine import IECore
from yolov8 import YOLO

def load_model(model_path):
    # Load YOLOv8 model
    model = YOLO(model_path)
    # OpenVino optimization for faster inference
    ie = IECore()
    network = ie.read_network(model=model_path)
    return network
```

3. Real-Time Detection

Once the model is ready, the system detects objects in real-time. The detect_objects.py script handles the detection logic, reading frames and processing them using the optimized model.

```python
import cv2
import numpy as np
from yolov8 import YOLO

def detect_objects(model, video_source=0):
    # Open video source (camera or video file)
    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Pre-process frame and detect objects
        detections = model.detect(frame)
        for detection in detections:
            # Draw bounding boxes and labels
            cv2.rectangle(frame, detection['box'], (0, 255, 0), 2)
            cv2.putText(frame, detection['label'], detection['box'][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Display the frame
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
```

4. Movement Detection and Inventory Checks

The system also handles tasks like inventory checking and movement detection, where detected objects are tracked across frames.

```python
from collections import deque

def track_objects(detections, previous_detections):
    # Simple tracking algorithm to monitor object movement
    tracked_objects = deque(maxlen=100)
    for detection in detections:
        label = detection['label']
        if label not in previous_detections:
            tracked_objects.append(detection)
    return tracked_objects
```

Setup Instructions

Prerequisites
	•	Python 3.x
	•	OpenVINO for model optimization
	•	Jupyter Notebook (optional for analysis)

Install Dependencies

```
pip install torch torchvision opencv-python openvino
```

Clone the Repository

```
git clone https://github.com/mrsamsonn/YOLO-Supervision.git
cd YOLO-Supervision
```

Load Custom Dataset

To train on your custom dataset, follow the provided examples in train_model.py.

Usage

Run the YOLO model

Execute the model to start object detection:

```
python detect_objects.py
```

Modify Configuration

You can adjust model parameters, input paths, and detection logic through the configuration file.

Output

The system outputs detection results as follows:
	•	Bounding boxes around detected objects
	•	Class labels and confidence scores
	•	Real-time tracking of detected objects

Project Structure

```
YOLO-Supervision/
│
├── detect_objects.py          # Script for running object detection
├── train_model.py            # Script for training with custom dataset
├── config.yaml               # Configuration file for the project
├── model_weights/            # Directory for model weights
├── README.md                 # Project documentation
└── data/                     # Directory for dataset
```

Future Work
	•	Enhance Real-time Performance: Further optimizations for faster inference on low-end hardware.
	•	Expand Dataset: Support for additional custom datasets for different object categories.
	•	Advanced Detection Logic: Improve detection under various environmental conditions.
