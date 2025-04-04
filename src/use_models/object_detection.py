"""
IMAGE & VIDEO OBJECT DETECTION WITH YOLOv8 NANO

This script uses the YOLOv8 Nano object detection model to detect objects in real time
for both images and video files. It draws bounding boxes and labels on detected objects.

If the model does not exist in the 'models/' folder, it is downloaded and moved
there from the current working directory or Ultralytics cache.
"""

from ultralytics import YOLO
import cv2
import os
import shutil
import glob
import numpy as np
from random import randint


def load_model():
    """
    Loads the YOLOv8 Nano object detection model.
    Returns:
        YOLO: The loaded detection model.
    """
    model_dir = "models"
    model_filename = "yolov8n.pt"
    model_path = os.path.join(model_dir, model_filename)
    cwd_model_path = os.path.abspath(model_filename)

    if os.path.exists(model_path):
        print("Model already exists in models/ folder.")
        return YOLO(model_path)

    print("Model not found in models/. Downloading with YOLO...")
    _ = YOLO(model_filename)

    if os.path.exists(cwd_model_path):
        os.makedirs(model_dir, exist_ok=True)
        shutil.move(cwd_model_path, model_path)
    else:
        cache_root = os.path.expanduser("~/.cache/ultralytics")
        matching_files = list(glob.iglob(f"{cache_root}/**/{model_filename}", recursive=True))
        if matching_files:
            os.makedirs(model_dir, exist_ok=True)
            shutil.move(matching_files[0], model_path)
        else:
            raise RuntimeError("Model was downloaded but not found in current dir or cache.")

    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        raise RuntimeError("Model not found. Could not load or move it successfully.")


def get_color_map(names):
    """
    Generates a consistent color map for all class names.
    Returns:
        dict: Mapping from class name to color tuple.
    """
    return {
        name: (randint(30, 255), randint(30, 255), randint(30, 255))
        for name in names.values()
    }


def draw_detections(image, result, color_map):
    """
    Draws bounding boxes and class labels on the image.
    
    Args:
        image (np.array): The input image.
        result: YOLO result object.
        color_map (dict): Class name to BGR color.

    Returns:
        np.array: Image with detections drawn.
        list of str: Class names detected.
    """
    detected_classes = set()
    boxes = result.boxes
    names = result.names

    for i in range(len(boxes)):
        box = boxes.xyxy[i].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
        class_id = int(boxes.cls[i].item())
        class_name = names[class_id]
        color = color_map[class_name]
        detected_classes.add(class_name)

        # Draw bounding box
        cv2.rectangle(image, box[:2], box[2:], color, 2)

        # Draw label
        label = f"{class_name}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_bg_top_left = (box[0], box[1] - h - 10)
        label_bg_bottom_right = (box[0] + w + 10, box[1])
        cv2.rectangle(image, label_bg_top_left, label_bg_bottom_right, color, -1)
        cv2.putText(image, label, (box[0] + 5, box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image, sorted(detected_classes)


def draw_class_labels(image, class_names, color_map):
    """
    Draws stacked class labels in the top-left corner.

    Args:
        image (np.array): The image to draw on.
        class_names (list of str): Class labels to show.
        color_map (dict): Mapping from class label to BGR color.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    margin = 10
    y_offset = margin

    for class_name in class_names:
        label = class_name
        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        box_coords = ((margin - 5, y_offset - 5), (margin + w + 5, y_offset + h + 5))

        cv2.rectangle(image, box_coords[0], box_coords[1], color_map[class_name], thickness=-1)
        cv2.putText(image, label, (margin, y_offset + h), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        y_offset += h + 10


def detect_image(model, image_path):
    """
    Runs object detection on a single image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    results = model(image)
    if not results:
        return

    result = results[0]
    color_map = get_color_map(result.names)
    overlay, class_names = draw_detections(image.copy(), result, color_map)
    draw_class_labels(overlay, class_names, color_map)

    cv2.imshow("YOLOv8 Object Detection - Image", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_video(model, video_path):
    """
    Runs object detection on each frame of a video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video from {video_path}")

    color_map = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        if not results:
            continue

        result = results[0]
        if color_map is None:
            color_map = get_color_map(result.names)

        overlay, class_names = draw_detections(frame.copy(), result, color_map)
        draw_class_labels(overlay, class_names, color_map)

        cv2.imshow("YOLOv8 Object Detection - Video", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Prompts user for file and runs object detection.
    """
    print("Loading YOLOv8 Nano detection model...")
    model = load_model()

    file_path = input("Enter path to an image or video file: ").strip()
    if not os.path.exists(file_path):
        print("Error: File does not exist.")
        return

    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            print("Running image detection...")
            detect_image(model, file_path)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            print("Running video detection...")
            detect_video(model, file_path)
        else:
            print("Unsupported file type.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
