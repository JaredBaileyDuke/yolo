"""
OBJECT DETECTION WITH YOLOv8 NANO: IMAGE, VIDEO, OR WEBCAM

This script uses the YOLOv8 Nano object detection model to:
- Detect objects in a single image
- Detect objects in a video file
- Detect objects from a live webcam stream

Features:
- Supports optional saving of processed image or video (not webcam)
- Draws labeled bounding boxes and class summary
- Auto-downloads the model if not found in the models/ folder
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
    Loads the YOLOv8 Nano object detection model from 'models/'.
    Downloads it if not already present.

    Returns:
        YOLO: The YOLOv8 object detection model.
    """
    model_dir = "models"
    model_filename = "yolov8n.pt"
    model_path = os.path.join(model_dir, model_filename)

    if os.path.exists(model_path):
        print("✅ Model loaded from models/")
        return YOLO(model_path)

    print("📥 Model not found. Downloading...")
    _ = YOLO(model_filename)

    cwd_path = os.path.abspath(model_filename)
    if os.path.exists(cwd_path):
        os.makedirs(model_dir, exist_ok=True)
        shutil.move(cwd_path, model_path)
    else:
        cache_root = os.path.expanduser("~/.cache/ultralytics")
        matches = list(glob.iglob(f"{cache_root}/**/{model_filename}", recursive=True))
        if matches:
            os.makedirs(model_dir, exist_ok=True)
            shutil.move(matches[0], model_path)
        else:
            raise RuntimeError("Model downloaded but not found in expected locations.")

    return YOLO(model_path)


def get_color_map(names):
    """
    Generates random but consistent colors for each class name.

    Args:
        names (dict): Mapping of class indices to names.

    Returns:
        dict: Mapping of class names to BGR color tuples.
    """
    return {
        name: (randint(30, 255), randint(30, 255), randint(30, 255))
        for name in names.values()
    }


def draw_detections(image, result, color_map):
    """
    Draws bounding boxes and labels on detected objects in an image.

    Args:
        image (np.array): Original image or frame.
        result: YOLO result object.
        color_map (dict): Class-to-color mapping.

    Returns:
        image (np.array): Annotated image.
        class_names (list): List of unique class names detected.
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

        # Draw bounding box and label
        cv2.rectangle(image, box[:2], box[2:], color, 2)
        label = f"{class_name}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (box[0], box[1] - h - 10), (box[0] + w + 10, box[1]), color, -1)
        cv2.putText(image, label, (box[0] + 5, box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image, sorted(detected_classes)


def draw_class_labels(image, class_names, color_map):
    """
    Displays detected class names stacked in the top-left corner.

    Args:
        image (np.array): The image to draw on.
        class_names (list): List of class names to show.
        color_map (dict): Class-to-color mapping.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    margin = 10
    y_offset = margin

    for class_name in class_names:
        label = class_name
        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(image, (margin - 5, y_offset - 5),
                      (margin + w + 5, y_offset + h + 5),
                      color_map[class_name], -1)
        cv2.putText(image, label, (margin, y_offset + h),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y_offset += h + 10


def detect_image(model, image_path, save_output=False, output_path="detected_image.jpg"):
    """
    Runs object detection on an image and displays/saves it.

    Args:
        model (YOLO): YOLOv8 model.
        image_path (str): Path to input image.
        save_output (bool): Save the output image.
        output_path (str): Output image path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Could not load image from {image_path}")

    results = model(image)
    result = results[0]
    color_map = get_color_map(result.names)
    overlay, class_names = draw_detections(image.copy(), result, color_map)
    draw_class_labels(overlay, class_names, color_map)

    cv2.imshow("YOLOv8 Object Detection - Image", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_output:
        cv2.imwrite(output_path, overlay)
        print(f"💾 Output saved to: {output_path}")


def detect_video(model, source, save_output=False, output_path="detected_video.mp4"):
    """
    Runs object detection on a video file or webcam stream.

    Args:
        model (YOLO): The YOLOv8 detection model.
        source (str or int): Path to video file or webcam index.
        save_output (bool): Whether to save the output.
        output_path (str): Output filename.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"❌ Could not open video source: {source}")

    writer = None
    frame_written = False
    color_map = None

    if save_output:
        # Auto-select codec based on file extension
        ext = os.path.splitext(output_path)[1].lower()
        codec = 'avc1' if ext == '.mp4' else 'XVID'
        fourcc = cv2.VideoWriter_fourcc(*codec)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30  # fallback

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width == 0 or height == 0:
            raise RuntimeError("❌ Unable to get video dimensions.")

        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Output video will be saved to: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        result = results[0]
        if color_map is None:
            color_map = get_color_map(result.names)

        overlay, class_names = draw_detections(frame.copy(), result, color_map)
        draw_class_labels(overlay, class_names, color_map)

        if save_output and writer:
            writer.write(overlay)
            frame_written = True

        cv2.imshow("YOLOv8 Object Detection - Video/Webcam", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

    cap.release()
    if writer:
        writer.release()
        if not frame_written:
            print("⚠️ Warning: No frames written. Output video may not be valid.")
    cv2.destroyAllWindows()


def main():
    """
    Main CLI loop:
    - Prompts user to choose between image, video, webcam
    - Offers saving options for image/video only
    - Runs YOLOv8 detection accordingly
    """
    print("🔍 Loading YOLOv8 Nano detection model...")
    model = load_model()

    print("\nSelect input type:")
    print("1 - Image")
    print("2 - Video")
    print("3 - Webcam (not saved)")

    choice = input("Your choice [1/2/3]: ").strip()

    if choice == '1':
        path = input("Enter path to image: ").strip()
        if not os.path.isfile(path):
            print("❌ Error: Image file not found.")
            return
        save = input("Save output image? (y/n): ").strip().lower() == 'y'
        output_path = "detected_image.jpg"
        if save:
            custom_name = input("Enter output filename (or press Enter for default): ").strip()
            if custom_name:
                output_path = custom_name
        detect_image(model, path, save_output=save, output_path=output_path)

    elif choice == '2':
        path = input("Enter path to video: ").strip()
        if not os.path.isfile(path):
            print("❌ Error: Video file not found.")
            return
        save = input("Save output video? (y/n): ").strip().lower() == 'y'
        output_path = "detected_video.mp4"
        if save:
            custom_name = input("Enter output filename (or press Enter for default): ").strip()
            if custom_name:
                output_path = custom_name
        detect_video(model, path, save_output=save, output_path=output_path)

    elif choice == '3':
        print("🎥 Starting webcam (live only, not saved)...")
        detect_video(model, 0, save_output=False)

    else:
        print("❌ Invalid selection. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
