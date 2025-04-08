"""
IMAGE, VIDEO, AND WEBCAM CLASSIFICATION WITH YOLOv8 NANO

This script uses the YOLOv8 Nano classification model to perform object classification
on images, video files, or live webcam input. Classification results are overlaid on the frame.

Features:
- Automatically downloads the model if not present
- Supports saving output for images and videos (not webcam)
- Displays classification confidence scores on each frame

Controls:
- Press 'Q' during webcam or video playback to exit.
"""

from ultralytics import YOLO
import cv2
import os
import shutil
import glob
import numpy as np


def load_model():
    """
    Loads the YOLOv8 Nano classification model from 'models/' folder.
    Downloads and moves it there if not already present.

    Returns:
        YOLO: Loaded YOLOv8 model.
    """
    model_dir = "models"
    model_filename = "yolov8n-cls.pt"
    model_path = os.path.join(model_dir, model_filename)
    cwd_model_path = os.path.abspath(model_filename)

    if os.path.exists(model_path):
        print("✅ Model found in models/ folder.")
        return YOLO(model_path)

    print("📥 Model not found. Downloading using Ultralytics...")
    _ = YOLO(model_filename)

    # Try moving from current directory
    if os.path.exists(cwd_model_path):
        os.makedirs(model_dir, exist_ok=True)
        shutil.move(cwd_model_path, model_path)
    else:
        # Try cache directory
        cache_root = os.path.expanduser("~/.cache/ultralytics")
        matching_files = list(glob.iglob(f"{cache_root}/**/{model_filename}", recursive=True))
        if matching_files:
            os.makedirs(model_dir, exist_ok=True)
            shutil.move(matching_files[0], model_path)
        else:
            raise RuntimeError("❌ Model downloaded but not found in current or cache directory.")

    return YOLO(model_path)


def draw_classification_label(image, label):
    """
    Draws classification label on top-left corner of an image or video frame.

    Args:
        image (np.array): Image/frame.
        label (str): Label text, e.g., "dog (0.98)".
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    margin = 10

    (text_width, text_height), _ = cv2.getTextSize(label, font, scale, thickness)

    # Blue background box
    cv2.rectangle(
        image,
        (margin - 5, margin - 5),
        (margin + text_width + 5, margin + text_height + 5),
        (255, 0, 0),
        thickness=-1
    )

    # White text
    cv2.putText(
        image,
        label,
        (margin, margin + text_height),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA
    )


def classify_image(model, image_path, save_output=False, output_path="classified_image.jpg"):
    """
    Classifies a single image and shows the result with overlay.
    
    Args:
        model (YOLO): YOLOv8 model.
        image_path (str): Path to image file.
        save_output (bool): Whether to save the output.
        output_path (str): Where to save the output image.
    
    Returns:
        list: YOLO classification results.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Could not load image: {image_path}")

    results = model(image)

    for result in results:
        class_name = result.names[result.probs.top1]
        confidence = result.probs.top1conf
        label = f"{class_name} ({confidence:.2f})"
        draw_classification_label(image, label)

    # Show the image
    cv2.imshow("YOLOv8 Image Classification", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save if requested
    if save_output:
        cv2.imwrite(output_path, image)
        print(f"💾 Image saved to: {output_path}")

    return results



def classify_video(model, source, save_output=False, output_path="output_classified.mp4"):
    """
    Performs classification on video or webcam frame-by-frame.
    
    Args:
        model (YOLO): YOLOv8 model.
        source (str or int): File path or webcam index.
        save_output (bool): Save the output video.
        output_path (str): File path to save output.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"❌ Could not open video source: {source}")

    writer = None
    frame_written = False

    if save_output:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width == 0 or height == 0:
            raise RuntimeError("❌ Invalid video dimensions.")

        ext = os.path.splitext(output_path)[1].lower()
        fourcc = cv2.VideoWriter_fourcc(*('XVID' if ext == '.avi' else 'avc1'))  # safe fallback
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Saving output to: {output_path}")

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
            frame_written = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

    cap.release()
    if writer:
        writer.release()
        if not frame_written:
            print("⚠️ No frames were written. Output file may be invalid.")
    cv2.destroyAllWindows()



def show_results(results):
    """
    Prints classification results to console.

    Args:
        results (list): YOLO result objects.
    """
    for result in results:
        class_name = result.names[result.probs.top1]
        confidence = result.probs.top1conf
        print(f"🧠 Classified as: {class_name} ({confidence:.2f})")


def main():
    """
    CLI-based main function to handle:
    - Loading model
    - Prompting for image/video/webcam
    - Saving outputs where applicable (not webcam)
    """
    print("🔍 Loading YOLOv8 Nano classification model...")
    model = load_model()

    print("\nSelect input type:")
    print("1 - Image file")
    print("2 - Video file")
    print("3 - Webcam (live stream, not saved)")

    choice = input("Your choice [1/2/3]: ").strip()

    if choice == '1':
        path = input("Enter path to image file: ").strip()
        if not os.path.isfile(path):
            print("❌ Error: Image file not found.")
            return
        save = input("Save output image? (y/n): ").strip().lower() == 'y'
        output_path = "classified_image.jpg"
        if save:
            user_path = input("Enter output filename (press Enter for default): ").strip()
            if user_path:
                output_path = user_path
        results = classify_image(model, path, save_output=save, output_path=output_path)
        show_results(results)


    elif choice == '2':
        path = input("Enter path to video file (.mp4, .avi): ").strip()
        if not os.path.isfile(path):
            print("❌ Error: Video file not found.")
            return
        save = input("Save output video? (y/n): ").strip().lower() == 'y'
        output_path = "output_classified.mp4"
        if save:
            user_path = input("Enter output filename (press Enter for default): ").strip()
            if user_path:
                output_path = user_path
        classify_video(model, path, save_output=save, output_path=output_path)

    elif choice == '3':
        print("🎥 Starting webcam classification (not saving)...")
        classify_video(model, 0, save_output=False)

    else:
        print("❌ Invalid selection. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
