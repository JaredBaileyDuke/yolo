"""
IMAGE & VIDEO CLASSIFICATION WITH YOLOv8 NANO

This script allows you to classify objects using the YOLOv8 Nano classification model.
It supports both single image and video input. The classification results are shown
in a blue rectangle with white text in the top-left corner of each image/frame.

If the model does not exist in the 'models/' folder, it is downloaded and moved
there from Ultralytics' internal cache.
"""

from ultralytics import YOLO
import cv2
import os
import shutil


def load_model():
    """
    Loads the YOLOv8 Nano classification model.

    - If the model already exists in the 'models/' folder, loads it from there.
    - If not, calls YOLO to trigger download (which may download to root).
    - Moves the downloaded model to 'models/'.
    - Ensures no leftover model copies are in the root directory.

    Returns:
        YOLO: The loaded YOLO model from the 'models/' directory.
    """
    import glob

    model_dir = "models"
    model_filename = "yolov8n-cls.pt"
    model_path = os.path.join(model_dir, model_filename)
    cwd_model_path = os.path.abspath(model_filename)

    # Step 1: If the model already exists in models/, load directly
    if os.path.exists(model_path):
        print("Model already exists in models/ folder.")
        return YOLO(model_path)

    # Step 2: If not, trigger YOLO to download it (likely to current dir or cache)
    print("Model not found in models/. Downloading with YOLO...")
    _ = YOLO(model_filename)

    # Step 3: Check current directory first (Ultralytics often downloads here)
    if os.path.exists(cwd_model_path):
        print(f"Found downloaded model in current directory: {cwd_model_path}")
        os.makedirs(model_dir, exist_ok=True)
        shutil.move(cwd_model_path, model_path)
        print(f"Moved model to: {model_path}")

    # Step 4: Else look in Ultralytics cache (~/.cache/ultralytics/**/)
    else:
        cache_root = os.path.expanduser("~/.cache/ultralytics")
        matching_files = list(glob.iglob(f"{cache_root}/**/{model_filename}", recursive=True))

        if matching_files:
            print(f"Found downloaded model in cache: {matching_files[0]}")
            os.makedirs(model_dir, exist_ok=True)
            shutil.move(matching_files[0], model_path)
            print(f"Moved model to: {model_path}")
        else:
            raise RuntimeError("Model was downloaded but not found in current dir or cache.")

    # Step 5: Confirm and load from models/
    if os.path.exists(model_path):
        print("Loading model from models/ folder...")
        return YOLO(model_path)
    else:
        raise RuntimeError("Model not found. Could not load or move it successfully.")


def draw_classification_label(image, label):
    """
    Draws a labeled blue rectangle with white text in the upper-left corner of an image.

    Args:
        image (np.array): The image or frame to annotate.
        label (str): The classification result to display.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX       # Font style for the text
    font_scale = 0.8                      # Font size
    font_thickness = 2                    # Thickness of the text
    margin = 10                           # Padding from the top-left corner

    # Measure the size of the text so we can size the box around it
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Draw a filled blue rectangle behind the label text
    cv2.rectangle(
        image,
        (margin - 5, margin - 5),
        (margin + text_width + 5, margin + text_height + 5),
        (255, 0, 0),                      # Blue box (BGR color format)
        thickness=-1                     # Filled rectangle
    )

    # Put white text on top of the blue rectangle
    cv2.putText(
        image,
        label,
        (margin, margin + text_height),
        font,
        font_scale,
        (255, 255, 255),                 # White text
        font_thickness,
        lineType=cv2.LINE_AA            # Anti-aliased line (smoother text)
    )


def classify_image(model, image_path):
    """
    Loads and classifies a single image using the provided YOLO model.

    - Reads the image from disk.
    - Runs YOLO classification.
    - Draws the result label on the image.
    - Displays the image in a window.

    Args:
        model (YOLO): A YOLO model instance.
        image_path (str): Path to the image file.

    Returns:
        list: Classification results from the model.
    """
    image = cv2.imread(image_path)                  # Load the image
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    results = model(image)                          # Run classification

    # Draw label on the image
    for result in results:
        class_name = result.names[result.probs.top1]        # Most likely class
        confidence = result.probs.top1conf                  # Confidence score
        label = f"{class_name} ({confidence:.2f})"          # Format label
        draw_classification_label(image, label)             # Draw it

    # Display the image with the label
    cv2.imshow("YOLOv8 Image Classification", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results


def classify_video(model, video_path):
    """
    Classifies each frame of a video using YOLOv8 Nano.

    - Opens the video file.
    - Processes each frame with YOLO.
    - Draws the predicted class + confidence on the frame.
    - Displays the video in real-time.

    Args:
        model (YOLO): A YOLO model instance.
        video_path (str): Path to the video file.
    """
    cap = cv2.VideoCapture(video_path)           # Open the video
    if not cap.isOpened():
        raise ValueError(f"Could not open video from {video_path}")

    while True:
        ret, frame = cap.read()                  # Read one frame
        if not ret:
            break                                # End of video

        results = model(frame)                   # Classify this frame
        for result in results:
            class_name = result.names[result.probs.top1]
            confidence = result.probs.top1conf
            label = f"{class_name} ({confidence:.2f})"
            draw_classification_label(frame, label)

        cv2.imshow("YOLOv8 Video Classification", frame)  # Show frame

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def show_results(results):
    """
    Prints classification results to the console.

    Args:
        results (list): List of YOLO classification result objects.
    """
    for result in results:
        class_name = result.names[result.probs.top1]
        confidence = result.probs.top1conf
        print(f"Classified as: {class_name} ({confidence:.2f})")


def main():
    """
    Main entry point.

    - Loads the YOLOv8 Nano classification model.
    - Asks the user for an image or video file path.
    - Classifies the file accordingly.
    """
    print("Loading YOLOv8 Nano model...")
    model = load_model()                         # Load the model

    file_path = input("Enter path to an image or video file: ").strip()

    # Check if the file exists
    if not os.path.exists(file_path):
        print("Error: File does not exist.")
        return

    # Determine file type from extension
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            print("Running image classification...")
            results = classify_image(model, file_path)
            show_results(results)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            print("Running video classification...")
            classify_video(model, file_path)
        else:
            print("Unsupported file type. Please use an image or video.")
    except Exception as e:
        print(f"An error occurred during classification: {e}")


if __name__ == "__main__":
    main()
