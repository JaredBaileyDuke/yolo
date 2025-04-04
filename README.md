# YOLO - Train Neural Networks & Run Pretrained Models

This project uses YOLOv8 to train neural networks and run pretrained models for classification on images and videos.

---

## 📁 Data Files

- All data should be placed in the `assets/` folder.
- You can add your own datasets to this folder for training or testing.

---

## 📂 Repository Structure

- assets/ - Contains all data files.
- models/ - Pretrained and trained YOLO models are stored here.
- src/ - Source code for using models and training.
  - use_models/ - Scripts for classification using pretrained models.
- train.py - Script to train custom YOLO networks.
- config.yaml - Configuration for training parameters.

---

## 🚀 Use YOLO Out of the Box

YOLO models are automatically downloaded the first time you run the script. The model will be placed in the `models/` folder for future use.

### 🔍 Classification (Pretrained)

#### Overview
Use pre-trained YOLOv8 models to classify either **images** or **videos**. Classification results are shown in real-time with a blue box and label in the top-left corner of each image/frame.

#### Run the Code

```bash
python src/use_models/classification.py
```

### 🎯 Object Detection (Pretrained)

#### Overview
Detect objects in images or videos using pre-trained YOLOv8 object detection models. Detected objects appear with bounding boxes and labels.

#### Run the Code
```bash
python src/use_models/detection.py
```

### 🗺️ OBB (Oriented Bounding Boxes)

#### Overview
Utilize YOLOv8 models for detecting oriented bounding boxes, ideal for applications requiring rotated detection outputs.

#### Run the Code
```bash
python src/use_models/obb_detection.py
```

### 💃 Pose Estimation

#### Overview
Extract human poses from images or videos using pre-trained YOLOv8 pose models. Keypoints are overlaid to visualize body positions.

#### Run the Code
```bash
python src/use_models/pose.py
```

### 🎨 Segmentation

#### Overview
Segment images into various objects and backgrounds with the help of YOLOv8 segmentation models. Each segmented area is highlighted accordingly.

#### Run the Code
```bash
python src/use_models/segmentation.py
```

---

## Train Neural Networks

### Classification

#### Overview
Train custom YOLO models on your dataset by configuring the training parameters.

#### Run the Code
Execute: 
```bash
python train.py --config config.yaml
```