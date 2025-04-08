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

## 🧪 Train Neural Networks

### 📌 Overview
Train custom YOLOv8 models for classification, object detection, segmentation, pose estimation, or OBB using your own data. You can start with existing datasets or annotate your own from scratch.

---

### 🔍 Where to Get Training Data

#### 🪐 Roboflow Universe
Browse thousands of publicly available datasets on [Roboflow Universe](https://universe.roboflow.com/). You can download datasets in the YOLOv8-compatible format (select **YOLOv8** when exporting).

Example steps:
1. Go to [https://universe.roboflow.com](https://universe.roboflow.com)
2. Search for a dataset (e.g., “birds”, “vehicles”, “medical”, etc.)
3. Export the dataset in **YOLOv8** format
4. Unzip the dataset into the `assets/` folder

---

#### ✍️ Annotate Your Own Images with CVAT

If you prefer to label your own data:
1. Go to [https://cvat.org](https://cvat.org) and create a free account.
2. Upload your images and create an annotation task.
3. Annotate using bounding boxes, polygons, keypoints, etc.
4. Export the labeled dataset as **YOLO format**.
5. Move the exported data to your `assets/` folder.

CVAT supports:
- Object detection
- Segmentation
- Pose/keypoints
- OBB (via rotated boxes or polygons)

---

### 🚀 Train a Custom Model

#### Step-by-Step (Google Colab Recommended)

Use the provided Jupyter notebook for training in the cloud using GPUs on Google Colab.