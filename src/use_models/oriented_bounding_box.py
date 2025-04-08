"""
OBB DETECTION WITH YOLOv8 NANO: IMAGE, VIDEO, OR WEBCAM

This script uses the YOLOv8 Nano OBB (Oriented Bounding Box) model to detect
and draw rotated bounding boxes on images, videos, or webcam streams.

Features:
- Automatically downloads model if not found locally
- Supports saving annotated outputs for images and videos
- Webcam is live only (not saved)
- Draws class names with consistent color coding
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
    Loads the YOLOv8 Nano OBB model from disk or downloads it if missing.

    Returns:
        YOLO: The loaded detection model.
    """
    model_dir = "models"
    model_filename = "yolov8n-obb.pt"
    model_path = os.path.join(model_dir, model_filename)

    # Check if the model file already exists in the models/ directory
    if os.path.exists(model_path):
        print("✅ Model found in models/ folder.")
        return YOLO(model_path)

    print("📥 Model not found. Downloading with YOLO...")

    # Trigger model download via Ultralytics
    _ = YOLO(model_filename)

    # Move downloaded model to models/ folder
    cwd_path = os.path.abspath(model_filename)
    if os.path.exists(cwd_path):
        os.makedirs(model_dir, exist_ok=True)
        shutil.move(cwd_path, model_path)
    else:
        # Check the ultralytics cache directory for model file
        cache_root = os.path.expanduser("~/.cache/ultralytics")
        matches = list(glob.iglob(f"{cache_root}/**/{model_filename}", recursive=True))
        if matches:
            os.makedirs(model_dir, exist_ok=True)
            shutil.move(matches[0], model_path)
        else:
            raise RuntimeError("Model downloaded but not found.")

    return YOLO(model_path)


def get_color_map(names):
    """
    Assigns a unique random color to each class for consistent labeling.

    Args:
        names (dict): Mapping of class indices to names.

    Returns:
        dict: Mapping of class name to BGR color tuple.
    """
    return {
        name: (randint(30, 255), randint(30, 255), randint(30, 255))
        for name in names.values()
    }


def draw_oriented_boxes(image, result, color_map):
    """
    Draws rotated bounding boxes and class labels on the image.

    Args:
        image (np.array): Input image/frame to annotate.
        result: YOLO model output for a frame.
        color_map (dict): Mapping of class names to color.

    Returns:
        np.array: Annotated image.
        list: List of detected class names.
    """
    detected_classes = set()
    boxes = result.obb
    names = result.names

    # If there are no boxes, return original image
    if boxes is None or boxes.xywhr is None:
        return image, []

    # Extract box data and class IDs
    xywhr = boxes.xywhr.cpu().numpy()  # [x_center, y_center, width, height, rotation]
    class_ids = boxes.cls.cpu().numpy().astype(int)

    # Loop through each detection
    for i in range(len(xywhr)):
        x, y, w, h, angle = xywhr[i]
        class_id = class_ids[i]
        class_name = names[class_id]
        color = color_map[class_name]
        detected_classes.add(class_name)

        # Get box corner points from rotated box data
        rect = ((x, y), (w, h), angle)
        box_points = cv2.boxPoints(rect).astype(int)

        # Draw the rotated bounding box
        cv2.drawContours(image, [box_points], 0, color, 2)

        # Create and draw a background box for the label
        label = class_name
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_pos = tuple(box_points[1])
        label_bg = (label_pos[0], label_pos[1] - h_text - 5), (label_pos[0] + w_text + 10, label_pos[1])
        cv2.rectangle(image, label_bg[0], label_bg[1], color, -1)

        # Draw the label text
        cv2.putText(image, label, (label_pos[0] + 5, label_pos[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image, sorted(detected_classes)


def draw_class_labels(image, class_names, color_map):
    """
    Draws a list of detected class names in the top-left corner.

    Args:
        image (np.array): Image to annotate.
        class_names (list): Detected class names.
        color_map (dict): Color assigned to each class.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    margin = 10
    y_offset = margin

    for name in class_names:
        (w, h), _ = cv2.getTextSize(name, font, scale, thickness)
        box = ((margin - 5, y_offset - 5), (margin + w + 5, y_offset + h + 5))
        cv2.rectangle(image, box[0], box[1], color_map[name], -1)
        cv2.putText(image, name, (margin, y_offset + h),
                    font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y_offset += h + 10


def detect_image(model, path, save_output=False, output_path="obb_image.jpg"):
    """
    Detects objects in a single image.

    Args:
        model (YOLO): YOLOv8 detection model.
        path (str): Path to the image.
        save_output (bool): Whether to save the output image.
        output_path (str): Path to save the output.
    """
    image = cv2.imread(path)
    if image is None:
        raise ValueError("❌ Unable to load image.")

    results = model(image)
    result = results[0]
    color_map = get_color_map(result.names)

    # Draw boxes and labels
    overlay, class_names = draw_oriented_boxes(image.copy(), result, color_map)
    draw_class_labels(overlay, class_names, color_map)

    # Display output
    cv2.imshow("YOLOv8 OBB Detection - Image", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save image
    if save_output:
        cv2.imwrite(output_path, overlay)
        print(f"💾 Output saved to: {output_path}")


def detect_video(model, source, save_output=False, output_path="obb_video.mp4"):
    """
    Detects objects in a video file or webcam stream.

    Args:
        model (YOLO): YOLOv8 detection model.
        source (str or int): File path or webcam index.
        save_output (bool): Whether to save the annotated video.
        output_path (str): Path to save output.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError("❌ Unable to open video source.")

    color_map = None
    writer = None
    frame_written = False

    # Setup video writer if saving is enabled
    if save_output:
        ext = os.path.splitext(output_path)[1].lower()
        codec = 'avc1' if ext == '.mp4' else 'XVID'
        fourcc = cv2.VideoWriter_fourcc(*codec)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Saving output to: {output_path}")

    # Frame-by-frame processing
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        result = results[0]
        if color_map is None:
            color_map = get_color_map(result.names)

        overlay, class_names = draw_oriented_boxes(frame.copy(), result, color_map)
        draw_class_labels(overlay, class_names, color_map)

        # Save frame if required
        if save_output and writer:
            writer.write(overlay)
            frame_written = True

        # Display frame
        cv2.imshow("YOLOv8 OBB Detection - Video/Webcam", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if writer:
        writer.release()
        if not frame_written:
            print("⚠️ No frames were written.")
    cv2.destroyAllWindows()


def main():
    """
    CLI Entry point:
    - Ask user for input type (image, video, webcam)
    - Ask to save image or video if applicable
    - Run detection accordingly
    """
    print("🔍 Loading YOLOv8 Nano OBB model...")
    model = load_model()

    # Ask user for input type
    print("\nSelect input type:")
    print("1 - Image")
    print("2 - Video")
    print("3 - Webcam (live only, not saved)")

    choice = input("Your choice [1/2/3]: ").strip()

    if choice == '1':
        # Handle image input
        path = input("Enter image path: ").strip()
        if not os.path.isfile(path):
            print("❌ Image not found.")
            return
        save = input("Save output image? (y/n): ").strip().lower() == 'y'
        output_path = "obb_image.jpg"
        if save:
            user_path = input("Output filename (or press Enter for default): ").strip()
            if user_path:
                output_path = user_path
        detect_image(model, path, save_output=save, output_path=output_path)

    elif choice == '2':
        # Handle video input
        path = input("Enter video path: ").strip()
        if not os.path.isfile(path):
            print("❌ Video not found.")
            return
        save = input("Save output video? (y/n): ").strip().lower() == 'y'
        output_path = "obb_video.mp4"
        if save:
            user_path = input("Output filename (or press Enter for default): ").strip()
            if user_path:
                output_path = user_path
        detect_video(model, path, save_output=save, output_path=output_path)

    elif choice == '3':
        # Handle webcam input
        print("🎥 Starting webcam detection (not saved)...")
        detect_video(model, 0, save_output=False)

    else:
        print("❌ Invalid selection. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
