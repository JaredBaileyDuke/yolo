"""
YOLOv8 NANO POSE ESTIMATION (Image, Video, Webcam)

This script performs human pose estimation using the YOLOv8n-pose model.
- Accepts input from image, video file, or live webcam.
- Optionally saves results for image or video.
- Displays keypoints and skeleton lines for 17 COCO-style joints.
"""

from ultralytics import YOLO
import cv2
import os
import shutil
import glob
import numpy as np


def load_model():
    """
    Loads the YOLOv8 pose estimation model from 'models/' directory.
    Downloads it if not already present.

    Returns:
        YOLO: Loaded YOLOv8 model for pose estimation.
    """
    model_dir = "models"
    model_filename = "yolov8n-pose.pt"
    model_path = os.path.join(model_dir, model_filename)

    # If the model already exists, use it
    if os.path.exists(model_path):
        print("✅ Model loaded from models/")
        return YOLO(model_path)

    print("📥 Model not found. Downloading using Ultralytics...")
    _ = YOLO(model_filename)  # This downloads the model to working dir or cache

    # Try moving the downloaded model to 'models/' folder
    if os.path.exists(model_filename):
        os.makedirs(model_dir, exist_ok=True)
        shutil.move(model_filename, model_path)
    else:
        # Try locating in cache
        cache_root = os.path.expanduser("~/.cache/ultralytics")
        matches = list(glob.iglob(f"{cache_root}/**/{model_filename}", recursive=True))
        if matches:
            os.makedirs(model_dir, exist_ok=True)
            shutil.move(matches[0], model_path)
        else:
            raise RuntimeError("❌ Model not found after download.")

    return YOLO(model_path)


def draw_poses(image, result):
    """
    Draws keypoints and skeleton connections on the image.

    Args:
        image (np.array): Input image.
        result: YOLO result containing keypoints.

    Returns:
        np.array: Image with keypoints and skeleton overlaid.
    """
    if not result or result.keypoints is None:
        return image

    kpts = result.keypoints.data.cpu().numpy()  # shape: [num_people, 17, 3]
    
    # Define COCO skeleton structure (pair of keypoints)
    skeleton = [
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 12), (5, 11), (6, 12)
    ]

    for person in kpts:
        # Draw joints
        for x, y, conf in person:
            if conf > 0.5:
                cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)

        # Draw bones (lines between keypoints)
        for a, b in skeleton:
            xa, ya, ca = person[a]
            xb, yb, cb = person[b]
            if ca > 0.5 and cb > 0.5:
                cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)

    return image


def pose_image(model, image_path, save_output=False, output_path="pose_image.jpg"):
    """
    Runs pose estimation on a single image.

    Args:
        model (YOLO): YOLOv8 pose model.
        image_path (str): Path to the input image.
        save_output (bool): If True, save the output image.
        output_path (str): Path to save the annotated image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Could not read image from {image_path}")

    results = model(image)
    if not results:
        return

    result = results[0]
    overlay = draw_poses(image.copy(), result)

    cv2.imshow("YOLOv8 Pose Estimation - Image", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_output:
        cv2.imwrite(output_path, overlay)
        print(f"💾 Output image saved to: {output_path}")


def pose_video(model, video_source, save_output=False, output_path="output_pose.mp4"):
    """
    Runs pose estimation on a video or webcam stream and optionally saves output.

    Args:
        model (YOLO): YOLOv8 pose model.
        video_source (str or int): File path or webcam index (e.g., 0).
        save_output (bool): Whether to save the output video.
        output_path (str): Where to save the video.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"❌ Could not open video source: {video_source}")

    writer = None
    frame_written = False
    color = (255, 0, 0)

    # Only setup writer if user wants to save
    if save_output:
        # Get width, height, and FPS from the video source
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30  # fallback if FPS not detected

        # Choose a compatible codec based on file extension
        ext = os.path.splitext(output_path)[1].lower()
        codec = 'avc1' if ext == '.mp4' else 'XVID'
        fourcc = cv2.VideoWriter_fourcc(*codec)

        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Saving output to: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run model prediction
        results = model(frame)
        if not results:
            continue

        result = results[0]
        overlay = draw_poses(frame.copy(), result)

        # Show frame in real-time
        cv2.imshow("YOLOv8 Pose Estimation - Video/Webcam", overlay)

        # Write to file if needed
        if save_output and writer:
            writer.write(overlay)
            frame_written = True

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

    cap.release()
    if writer:
        writer.release()
        if not frame_written:
            print("⚠️ Warning: No frames were written. Output video may be invalid.")
    cv2.destroyAllWindows()


def main():
    """
    Command-line interface to select input type and run pose estimation:
    - Image: optional output saving
    - Video: optional output saving
    - Webcam: live view only (no saving)
    """
    print("🔍 Loading YOLOv8 Nano pose model...")
    model = load_model()

    # Ask user for input type
    print("\nSelect input type:")
    print("1 - Image file")
    print("2 - Video file (e.g., .mp4)")
    print("3 - Webcam (live only, no saving)")

    choice = input("Your choice [1/2/3]: ").strip()

    if choice == '1':
        # Image input
        path = input("Enter path to image: ").strip()
        if not os.path.exists(path):
            print("❌ Image not found.")
            return
        save = input("Save output image? (y/n): ").strip().lower() == 'y'
        output_path = "pose_image.jpg"
        if save:
            custom_name = input("Output filename (press Enter for default): ").strip()
            if custom_name:
                output_path = custom_name
        pose_image(model, path, save_output=save, output_path=output_path)

    elif choice == '2':
        # Video input
        path = input("Enter path to video: ").strip()
        if not os.path.exists(path):
            print("❌ Video not found.")
            return
        save = input("Save output video? (y/n): ").strip().lower() == 'y'
        output_path = "output_pose.mp4"
        if save:
            custom_name = input("Output filename (press Enter for default): ").strip()
            if custom_name:
                output_path = custom_name
        pose_video(model, path, save_output=save, output_path=output_path)

    elif choice == '3':
        # Webcam input - live view only, no saving
        print("🎥 Starting live webcam pose estimation (not saved)...")
        pose_video(model, 0, save_output=False)

    else:
        print("❌ Invalid selection. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
