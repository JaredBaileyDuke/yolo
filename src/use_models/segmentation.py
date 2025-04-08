"""
IMAGE, VIDEO, & WEBCAM SEGMENTATION WITH YOLOv8 NANO

This script uses the YOLOv8 Nano segmentation model to:
- Identify and color objects in an image, video, or webcam stream.
- Display object labels with custom colors.
- Optionally save output for images and video.
- Uses safe video codecs for compatibility.

Press 'Q' during webcam/video to quit.
"""

from ultralytics import YOLO  # Load YOLOv8 models
import cv2  # OpenCV for image/video processing
import os  # File system ops
import shutil  # Move downloaded models
import glob  # Pattern matching for file search
import numpy as np  # NumPy for masks
from random import randint  # Random colors for classes


def load_model():
    """
    Load the YOLOv8 Nano segmentation model.
    Downloads the model if not found locally.

    Returns:
        YOLO: Loaded segmentation model.
    """
    model_dir = "models"
    model_filename = "yolov8n-seg.pt"
    model_path = os.path.join(model_dir, model_filename)

    if os.path.exists(model_path):
        print("✅ Model loaded from models/")
        return YOLO(model_path)

    print("📥 Model not found. Downloading using YOLO API...")
    _ = YOLO(model_filename)  # Triggers download

    # Move model to 'models/' directory
    if os.path.exists(model_filename):
        os.makedirs(model_dir, exist_ok=True)
        shutil.move(model_filename, model_path)
    else:
        # Try to retrieve from cache
        cache_root = os.path.expanduser("~/.cache/ultralytics")
        found = list(glob.iglob(f"{cache_root}/**/{model_filename}", recursive=True))
        if found:
            os.makedirs(model_dir, exist_ok=True)
            shutil.move(found[0], model_path)
        else:
            raise RuntimeError("❌ Model not found after download.")

    return YOLO(model_path)


def get_color_map(names):
    """
    Assign a unique random color for each class label.

    Args:
        names (dict): Mapping of class index to class name.

    Returns:
        dict: Class name to BGR color.
    """
    return {name: (randint(30, 255), randint(30, 255), randint(30, 255)) for name in names.values()}


def apply_segmentation_overlay(image, result, color_map):
    """
    Overlay segmentation masks on an image.

    Args:
        image (np.array): Original image.
        result: YOLO result object.
        color_map (dict): Class name to color.

    Returns:
        np.array: Image with overlays.
        list: List of detected class names.
        dict: Color map used.
    """
    overlay = image.copy()
    detected_classes = set()

    # Retrieve masks and class IDs from result
    masks = result.masks.data.cpu().numpy() if result.masks else []
    class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []

    for i, mask in enumerate(masks):
        class_id = class_ids[i] if i < len(class_ids) else 0
        class_name = result.names[class_id]
        color = color_map[class_name]
        detected_classes.add(class_name)

        # Resize mask to image size and threshold it
        resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        binary = (resized > 0.5).astype(np.uint8)

        # Build color mask
        color_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            color_mask[:, :, c] = binary * color[c]

        # Apply color mask to overlay using transparency
        mask_indices = binary.astype(bool)
        overlay[mask_indices] = cv2.addWeighted(overlay[mask_indices], 0.5, color_mask[mask_indices], 0.5, 0)

    return overlay, sorted(detected_classes), color_map


def draw_class_labels(image, class_names, color_map):
    """
    Draws a color-coded list of class names in the top-left.

    Args:
        image (np.array): The image to annotate.
        class_names (list): Sorted list of detected classes.
        color_map (dict): Class name to BGR color.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    y_offset = 10

    for name in class_names:
        (w, h), _ = cv2.getTextSize(name, font, scale, thickness)
        cv2.rectangle(image, (5, y_offset - 5), (10 + w, y_offset + h + 5), color_map[name], -1)
        cv2.putText(image, name, (10, y_offset + h), font, scale, (255, 255, 255), thickness)
        y_offset += h + 10


def segment_image(model, path, save_output=False, output_path="segmented_image.jpg"):
    """
    Segments a single image and optionally saves output.

    Args:
        model (YOLO): Loaded YOLOv8 segmentation model.
        path (str): Image file path.
        save_output (bool): Whether to save the output image.
        output_path (str): Save path for the output image.
    """
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"❌ Could not read image: {path}")

    results = model(image)
    result = results[0]
    color_map = get_color_map(result.names)
    overlay, class_names, _ = apply_segmentation_overlay(image, result, color_map)
    draw_class_labels(overlay, class_names, color_map)

    # Show the result
    cv2.imshow("YOLOv8 Image Segmentation", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_output:
        cv2.imwrite(output_path, overlay)
        print(f"💾 Image saved to: {output_path}")


def segment_video(model, source, save_output=False, output_path="segmented_video.mp4"):
    """
    Segments a video file or webcam feed.

    Args:
        model (YOLO): YOLOv8 segmentation model.
        source (str or int): File path or webcam index.
        save_output (bool): Whether to save the video output.
        output_path (str): Where to save the output video.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"❌ Could not open source: {source}")

    writer = None
    frame_written = False
    color_map = None

    # Setup writer if saving output
    if save_output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ext = os.path.splitext(output_path)[1].lower()
        codec = 'avc1' if ext == '.mp4' else 'XVID'
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Saving video to: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        result = results[0]

        # Generate consistent colors
        if color_map is None:
            color_map = get_color_map(result.names)

        overlay, class_names, _ = apply_segmentation_overlay(frame, result, color_map)
        draw_class_labels(overlay, class_names, color_map)

        # Show the result
        cv2.imshow("YOLOv8 Segmentation - Video/Webcam", overlay)

        if save_output and writer:
            writer.write(overlay)
            frame_written = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

    cap.release()
    if writer:
        writer.release()
        if not frame_written:
            print("⚠️ No frames written. Output file may be empty.")
    cv2.destroyAllWindows()


def main():
    """
    Command-line user interface:
    - Select input type: image, video, webcam
    - Choose to save image/video output
    - Run segmentation accordingly
    """
    print("🔍 Loading YOLOv8 Nano segmentation model...")
    model = load_model()

    print("\nChoose input type:")
    print("1 - Image")
    print("2 - Video")
    print("3 - Webcam (live only, not saved)")

    choice = input("Your choice [1/2/3]: ").strip()

    if choice == '1':
        path = input("Enter image path: ").strip()
        if not os.path.isfile(path):
            print("❌ File not found.")
            return
        save = input("Save output image? (y/n): ").strip().lower() == 'y'
        output_path = "segmented_image.jpg"
        if save:
            custom = input("Enter output filename (or press Enter for default): ").strip()
            if custom:
                output_path = custom
        segment_image(model, path, save_output=save, output_path=output_path)

    elif choice == '2':
        path = input("Enter video path: ").strip()
        if not os.path.isfile(path):
            print("❌ File not found.")
            return
        save = input("Save output video? (y/n): ").strip().lower() == 'y'
        output_path = "segmented_video.mp4"
        if save:
            custom = input("Enter output filename (or press Enter for default): ").strip()
            if custom:
                output_path = custom
        segment_video(model, path, save_output=save, output_path=output_path)

    elif choice == '3':
        print("🎥 Running segmentation on live webcam (not saved)...")
        segment_video(model, 0, save_output=False)

    else:
        print("❌ Invalid selection.")


if __name__ == "__main__":
    main()
