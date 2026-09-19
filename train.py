"""
train.py

Trains a custom object detector using YOLOv8 (via the Ultralytics library)
on the dataset described by dataset/data.yaml.

Requirements:
    pip install -r requirements.txt

Run:
    python train.py

This will:
  1. Download a small pretrained YOLOv8 base model (yolov8n.pt) - requires
     internet access the first time it runs.
  2. Fine-tune it on your dataset (dataset/data.yaml).
  3. Save the trained weights under runs/detect/train/weights/best.pt
"""

from ultralytics import YOLO

DATA_CONFIG = "dataset/data.yaml"
BASE_MODEL = "yolov8n.pt"   # smallest/fastest YOLOv8 variant, good for learning
EPOCHS = 30
IMG_SIZE = 320


def main():
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=8,
        name="custom_detector",
    )

    metrics = model.val()
    print("\nValidation metrics:")
    print(metrics)

    print("\nTraining complete.")
    print("Best weights saved to: runs/detect/custom_detector/weights/best.pt")


if __name__ == "__main__":
    main()
