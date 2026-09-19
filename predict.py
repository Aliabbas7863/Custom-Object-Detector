"""
predict.py

Runs the trained model on one or more images and saves the results
(with drawn bounding boxes and labels) so you can visually verify
that training worked.

Run:
    python predict.py --weights runs/detect/custom_detector/weights/best.pt --source dataset/images/val
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="runs/detect/custom_detector/weights/best.pt",
        help="Path to trained model weights (.pt file)",
    )
    parser.add_argument(
        "--source",
        default="dataset/images/val",
        help="Path to an image, folder of images, or video to run detection on",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Minimum confidence threshold to show a detection",
    )
    args = parser.parse_args()

    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,          # saves annotated images to runs/detect/predict/
        show_labels=True,
        show_conf=True,
    )

    print(f"\nRan detection on: {args.source}")
    print("Annotated results saved under: runs/detect/predict/")

    for r in results:
        names = [model.names[int(c)] for c in r.boxes.cls] if r.boxes is not None else []
        print(f"{r.path}: detected {names}")


if __name__ == "__main__":
    main()
