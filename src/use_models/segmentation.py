"""
IMAGE & VIDEO SEGMENTATION WITH YOLOv8 NANO

This script uses the YOLOv8 Nano segmentation model to identify objects and draw
segmentation masks in real time for images and videos.

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
    Loads the YOLOv8 Nano segmentation model.
    Returns:
        YOLO: The loaded segmentation model.
    """
    model_dir = "models"
    model_filename = "yolov8n-seg.pt"
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


def apply_segmentation_overlay(image, result, color_map):
    """
    Applies colored masks to the image for each class instance.
    Args:
        image (np.array): Original image.
        result: YOLO result object.
        color_map (dict): Class name to BGR color.
    Returns:
        np.array: Image with masks overlaid.
        list of str: Class names detected in the frame.
        dict: Class label -> color
    """
    overlay = image.copy()
    detected_classes = set()
    mask_data = result.masks.data.cpu().numpy() if result.masks else []
    class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []

    for i, mask in enumerate(mask_data):
        class_id = class_ids[i] if i < len(class_ids) else 0
        class_name = result.names[class_id]
        color = color_map[class_name]
        detected_classes.add(class_name)

        # Resize mask to match the input image size
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_resized > 0.5).astype(np.uint8)

        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[:, :, 0] = binary_mask * color[0]
        colored_mask[:, :, 1] = binary_mask * color[1]
        colored_mask[:, :, 2] = binary_mask * color[2]

        # Blend the original image with the colored mask using the mask
        mask_indices = binary_mask.astype(bool)
        overlay[mask_indices] = cv2.addWeighted(
            overlay[mask_indices].astype(np.uint8),
            0.5,
            colored_mask[mask_indices].astype(np.uint8),
            0.5,
            0,
        )

    return overlay, sorted(detected_classes), color_map


def draw_class_labels(image, class_names, color_map):
    """
    Draws stacked class labels in the top-left corner, each with a colored background.

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

        # Draw filled rectangle with class color
        cv2.rectangle(image, box_coords[0], box_coords[1], color_map[class_name], thickness=-1)

        # Draw the class name on top of it
        cv2.putText(image, label, (margin, y_offset + h), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        y_offset += h + 10  # Move down for next class


def segment_image(model, image_path):
    """
    Segments a single image and overlays results.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    results = model(image)
    if not results:
        return

    result = results[0]
    color_map = get_color_map(result.names)
    overlay, class_names, _ = apply_segmentation_overlay(image, result, color_map)
    draw_class_labels(overlay, class_names, color_map)

    cv2.imshow("YOLOv8 Image Segmentation", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segment_video(model, video_path):
    """
    Segments each frame of a video with mask overlays.
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

        overlay, class_names, _ = apply_segmentation_overlay(frame, result, color_map)
        draw_class_labels(overlay, class_names, color_map)

        cv2.imshow("YOLOv8 Video Segmentation", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("Loading YOLOv8 Nano segmentation model...")
    model = load_model()

    file_path = input("Enter path to an image or video file: ").strip()
    if not os.path.exists(file_path):
        print("Error: File does not exist.")
        return

    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            print("Running image segmentation...")
            segment_image(model, file_path)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            print("Running video segmentation...")
            segment_video(model, file_path)
        else:
            print("Unsupported file type.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
