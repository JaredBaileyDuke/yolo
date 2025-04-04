"""
IMAGE, VIDEO, AND WEBCAM CLASSIFICATION WITH YOLOv8 NANO

This script uses the YOLOv8 Nano classification model to perform object classification
on images, video files, or live webcam input. Classification results are overlaid on the frame.

Features:
- Automatically downloads the model if not present
- Supports saving video or webcam output with overlaid labels
- Shows classification confidence scores

Press 'Q' during video or webcam streaming to exit.
"""

from ultralytics import YOLO
import cv2
import os
import shutil
import glob
import numpy as np


def load_model():
    """
    Loads the YOLOv8 Nano classification model.

    Returns:
        YOLO: Loaded classification model.

    Workflow:
    - Checks if model is in 'models/' folder.
    - If not, downloads it and moves to the folder.
    """
    model_dir = "models"
    model_filename = "yolov8n-cls.pt"
    model_path = os.path.join(model_dir, model_filename)
    cwd_model_path = os.path.abspath(model_filename)

    if os.path.exists(model_path):
        print("Model already exists in models/ folder.")
        return YOLO(model_path)

    print("Model not found in models/. Downloading with YOLO...")
    _ = YOLO(model_filename)

    # First check if model landed in current directory
    if os.path.exists(cwd_model_path):
        os.makedirs(model_dir, exist_ok=True)
        shutil.move(cwd_model_path, model_path)
    else:
        # Else check the Ultralytics cache
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


def draw_classification_label(image, label):
    """
    Draws a classification label in the top-left corner of the image.

    Args:
        image (np.array): Image or frame to draw on.
        label (str): Label text (e.g., "cat (0.95)").
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    margin = 10

    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Draw background rectangle
    cv2.rectangle(
        image,
        (margin - 5, margin - 5),
        (margin + text_width + 5, margin + text_height + 5),
        (255, 0, 0),  # Blue background
        thickness=-1
    )

    # Overlay white text
    cv2.putText(
        image,
        label,
        (margin, margin + text_height),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        lineType=cv2.LINE_AA
    )


def classify_image(model, image_path):
    """
    Classifies a single image and displays the result.

    Args:
        model (YOLO): YOLOv8 model object.
        image_path (str): Path to the image file.

    Returns:
        list: YOLO classification results.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    results = model(image)

    for result in results:
        class_name = result.names[result.probs.top1]
        confidence = result.probs.top1conf
        label = f"{class_name} ({confidence:.2f})"
        draw_classification_label(image, label)

    # Display result
    cv2.imshow("YOLOv8 Image Classification", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results


def classify_video(model, source, save_output=False, output_path="output_classified.mp4"):
    """
    Performs classification on video file or webcam stream frame-by-frame.

    Args:
        model (YOLO): YOLOv8 classification model.
        source (str or int): Path to video file or webcam index (e.g., 0).
        save_output (bool): Whether to save the output video.
        output_path (str): Path to save the output video if enabled.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")

    writer = None
    if save_output:
        # Set default FPS in case it's 0 (webcam issue)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Ensure frame size is valid
        if width == 0 or height == 0:
            raise RuntimeError("Unable to get video frame dimensions.")

        # Use mp4 codec for compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            class_name = result.names[result.probs.top1]
            confidence = result.probs.top1conf
            label = f"{class_name} ({confidence:.2f})"
            draw_classification_label(frame, label)

        cv2.imshow("YOLOv8 Video Classification", frame)

        if save_output and writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def show_results(results):
    """
    Prints the classification results to console.

    Args:
        results (list): YOLO result objects.
    """
    for result in results:
        class_name = result.names[result.probs.top1]
        confidence = result.probs.top1conf
        print(f"Classified as: {class_name} ({confidence:.2f})")


def main():
    """
    Main entry point for the script.

    - Loads YOLOv8 model.
    - Prompts user to select input type.
    - Handles classification accordingly.
    """
    print("Loading YOLOv8 Nano classification model...")
    model = load_model()

    print("\nSelect input type:")
    print("1 - Image file")
    print("2 - Video file")
    print("3 - Webcam (live only, no saving)")

    choice = input("Your choice: ").strip()

    if choice == '1':
        # Image classification
        path = input("Enter path to image: ").strip()
        if not os.path.exists(path):
            print("Error: File does not exist.")
            return
        results = classify_image(model, path)
        show_results(results)

    elif choice == '2':
        # Video classification
        path = input("Enter path to video: ").strip()
        if not os.path.exists(path):
            print("Error: File does not exist.")
            return
        save = input("Save output video? (y/n): ").strip().lower() == 'y'
        output_path = "output_classified.mp4"
        if save:
            user_path = input("Enter output filename (or press enter to use default): ").strip()
            if user_path:
                output_path = user_path
        classify_video(model, path, save_output=save, output_path=output_path)

    elif choice == '3':
        # Webcam (live only)
        print("Running webcam classification (live only, not saving)...")
        classify_video(model, 0, save_output=False)

    else:
        print("Invalid selection. Exiting.")


if __name__ == "__main__":
    main()
