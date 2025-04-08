"""
ROCK PAPER SCISSORS DETECTOR VIA WEBCAM

This script uses a YOLOv8 model to detect rock, paper, or scissors in real time
from your webcam. The model file is located at ../../models/best.pt.

Features:
- Webcam input only
- Bounding boxes and prediction labels
- Confidence score for each prediction

Requirements:
- ultralytics
- opencv-python
"""

from ultralytics import YOLO
import cv2
import os


def load_model():
    """
    Load the YOLOv8 model from a relative path.

    Returns:
        YOLO: The loaded YOLOv8 model.
    """
    model_path = os.path.abspath(os.path.join(__file__, "../../..", "models/best.pt"))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    print(f"✅ Model loaded from {model_path}")
    return YOLO(model_path)


def draw_predictions(frame, result):
    """
    Draw bounding boxes and labels for each detection on the frame.

    Args:
        frame (np.array): The current video frame.
        result: YOLO result object.

    Returns:
        np.array: Frame with annotations drawn.
    """
    boxes = result.boxes
    names = result.names

    if boxes is None or boxes.cls is None:
        return frame

    for i in range(len(boxes)):
        # Get box coordinates (x1, y1, x2, y2)
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        class_id = int(boxes.cls[i])
        confidence = float(boxes.conf[i])
        label = f"{names[class_id]} ({confidence:.2f})"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 255, 0), -1)

        # Draw label text
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


def main():
    """
    Run real-time Rock-Paper-Scissors detection using webcam.
    """
    model = load_model()

    cap = cv2.VideoCapture(0)  # Open default webcam
    if not cap.isOpened():
        print("❌ Could not access webcam.")
        return

    print("📷 Running Rock-Paper-Scissors detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame)
        result = results[0]

        # Draw bounding boxes and labels
        annotated_frame = draw_predictions(frame.copy(), result)

        # Display the result
        cv2.imshow("Rock-Paper-Scissors Detection", annotated_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
