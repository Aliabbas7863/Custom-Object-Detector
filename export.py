"""
export.py

Exports the trained PyTorch model (.pt) into formats you can use outside
of Python, including a browser-compatible TensorFlow.js format for
plugging into a web app like object-scanner.html.

Run:
    python export.py --weights runs/detect/custom_detector/weights/best.pt --format onnx
    python export.py --weights runs/detect/custom_detector/weights/best.pt --format tfjs

Notes:
  - 'onnx' is a good universal intermediate format, widely supported.
  - 'tfjs' produces a model directory you can load in the browser with
    TensorFlow.js, similar to how object-scanner.html loads COCO-SSD.
    Note: exporting to tfjs requires additional packages (tensorflowjs)
    and may require converting through 'saved_model' first depending on
    your Ultralytics version - check the console output for guidance if
    it fails.
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
        "--format",
        default="onnx",
        choices=["onnx", "tfjs", "torchscript", "saved_model"],
        help="Export format",
    )
    args = parser.parse_args()

    model = YOLO(args.weights)
    exported_path = model.export(format=args.format)

    print(f"\nExported model to: {exported_path}")
    if args.format == "tfjs":
        print(
            "\nTo use this in a web app, copy the exported model folder into "
            "your project and load it with TensorFlow.js's tf.loadGraphModel() "
            "or the appropriate loader for your export version."
        )


if __name__ == "__main__":
    main()
