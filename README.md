# YOLO - Train Neural Networks & Run Pretrained Models

This project uses YOLOv8 to train neural networks and run pretrained models for classification on images and videos.

---

## 📁 Data Files

- All data should be placed in the `assets/` folder.
- You can add your own datasets to this folder for training or testing.

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



## Train Neural Networks

### Classification

#### Overview
Train custom YOLO models on your dataset by configuring the training parameters.

#### Run the Code
Execute: "python train.py --config config.yaml"