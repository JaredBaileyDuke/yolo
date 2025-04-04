"""
YOLOv8 NANO POSE ESTIMATION (Image, Video, Webcam)

This script detects human poses using the YOLOv8n-pose.pt model.
- Works on images, videos, or live webcam feed.
- Optionally saves annotated video output.

Pose output includes 17 keypoints and connecting skeleton lines.
"""

from ultralytics import YOLO
import cv2
import os
import shutil
import glob
import numpy as np


def load_model():
    model_dir = "models"
    model_filename = "yolov8n-pose.pt"
    model_path = os.path.join(model_dir, model_filename)

    if os.path.exists(model_path):
        print("Model loaded from models/")
        return YOLO(model_path)

    print("Model not found in models/. Downloading with YOLO...")
    _ = YOLO(model_filename)

    if os.path.exists(model_filename):
        os.makedirs(model_dir, exist_ok=True)
        shutil.move(model_filename, model_path)
    else:
        cache_root = os.path.expanduser("~/.cache/ultralytics")
        matches = list(glob.iglob(f"{cache_root}/**/{model_filename}", recursive=True))
        if matches:
            os.makedirs(model_dir, exist_ok=True)
            shutil.move(matches[0], model_path)
        else:
            raise RuntimeError("Model not found after download.")

    return YOLO(model_path)


def draw_poses(image, result):
    """
    Draw keypoints and COCO skeleton connections on the image.
    """
    if not result or result.keypoints is None:
        return image

    kpts = result.keypoints.data.cpu().numpy()  # [num_people, 17, 3]
    skeleton = [
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 12), (5, 11), (6, 12)
    ]

    for person in kpts:
        for x, y, conf in person:
            if conf > 0.5:
                cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)

        for a, b in skeleton:
            xa, ya, ca = person[a]
            xb, yb, cb = person[b]
            if ca > 0.5 and cb > 0.5:
                cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)

    return image


def pose_image(model, image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    results = model(image)
    if not results:
        return

    result = results[0]
    overlay = draw_poses(image.copy(), result)

    cv2.imshow("YOLOv8 Pose Estimation - Image", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pose_video(model, video_source, save_output=False, output_path="output_pose.mp4"):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source {video_source}")

    writer = None
    if save_output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        if not results:
            continue

        result = results[0]
        overlay = draw_poses(frame.copy(), result)

        cv2.imshow("YOLOv8 Pose Estimation - Video", overlay)
        if save_output:
            writer.write(overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    print("Loading YOLOv8 Nano pose model...")
    model = load_model()

    print("\nSelect input type:")
    print("1 - Image")
    print("2 - Video file (e.g., .mp4)")
    print("3 - Webcam")

    choice = input("Your choice: ").strip()

    if choice == '1':
        path = input("Enter path to image: ").strip()
        if not os.path.exists(path):
            print("Image not found.")
            return
        pose_image(model, path)

    elif choice == '2':
        path = input("Enter path to video: ").strip()
        if not os.path.exists(path):
            print("Video not found.")
            return
        save = input("Save output video? (y/n): ").strip().lower() == 'y'
        pose_video(model, path, save_output=save)

    elif choice == '3':
        save = input("Save webcam output video? (y/n): ").strip().lower() == 'y'
        pose_video(model, 0, save_output=save)

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
