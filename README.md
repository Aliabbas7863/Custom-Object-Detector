# Custom Object Detector — Training Pipeline

A complete, runnable pipeline for training your own object detection model with **YOLOv8** (via Ultralytics). Comes with a small **synthetic sample dataset already generated** (colored shapes: circle, square, triangle) so you can run the whole pipeline immediately and confirm everything works — then swap in a real dataset from Kaggle once you're ready.

This is a **separate, follow-up project** to the webcam `object-scanner.html` app — it produces a *new* trained model. Plugging that custom model into the web app is a further integration step described at the bottom.

---

## 📁 What's in this project

```
custom-detector-project/
├── generate_dataset.py     # Creates the included synthetic sample dataset
├── train.py                 # Trains a YOLOv8 model on dataset/data.yaml
├── predict.py                # Runs the trained model on images to test it
├── export.py                  # Exports the trained model (ONNX / TF.js / etc.)
├── requirements.txt         # Python packages needed
├── dataset/
│   ├── data.yaml            # Class names + paths (points training at your data)
│   ├── images/
│   │   ├── train/            # 80 sample training images (auto-generated)
│   │   └── val/               # 20 sample validation images (auto-generated)
│   └── labels/
│       ├── train/            # YOLO-format .txt label files for each train image
│       └── val/               # YOLO-format .txt label files for each val image
└── README.md
```

The sample dataset (80 train / 20 val images of colored shapes) already exists in `dataset/` — you don't need to run `generate_dataset.py` again unless you want to regenerate it or study how it works.

---

## 🚀 Quick Start (using the included sample dataset)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (downloads a small pretrained YOLOv8 base model the first time — needs internet)
python train.py

# 3. Test the trained model on validation images
python predict.py --weights runs/detect/custom_detector/weights/best.pt --source dataset/images/val

# 4. Export for other uses (e.g. browser deployment)
python export.py --weights runs/detect/custom_detector/weights/best.pt --format onnx
```

After step 2, check `runs/detect/custom_detector/` for training charts and metrics. After step 3, check `runs/detect/predict/` for images with predicted bounding boxes drawn on them — this is how you visually confirm training worked.

> **Note:** Steps 2 and 4 need internet access (to download the base YOLOv8 weights and any export dependencies). Run these on your own machine, in Google Colab, or in a Kaggle Notebook — anywhere with internet and ideally a GPU.

---

## 🔁 Swapping in a Real Dataset from Kaggle

The included dataset is intentionally simple (synthetic shapes) so you can verify the *pipeline* works before dealing with a real, messier dataset. To use a real one:

1. **Find a Kaggle dataset already in YOLO format** — look for datasets that include a `data.yaml` and `labels/*.txt` files alongside the images (search Kaggle for "YOLO format" + your object of interest, e.g. "pothole detection YOLO format").
2. **Replace the contents of `dataset/`** with the downloaded dataset's images and labels, keeping the same folder structure (`images/train`, `images/val`, `labels/train`, `labels/val`).
3. **Update `dataset/data.yaml`** to match the new dataset's class names and image counts (the downloaded dataset usually already includes its own `data.yaml` you can use directly — just update the `path:` line to point to your local folder).
4. **Re-run `train.py`** — nothing else needs to change, since it just reads whatever `dataset/data.yaml` points to.

If your chosen Kaggle dataset is **not** already in YOLO format (e.g. it uses Pascal VOC XML or COCO JSON annotations), it will need to be converted first — Roboflow (roboflow.com) has a free tool that converts between annotation formats and can export directly to YOLO format if you'd rather not write a converter script by hand.

---

## 🌐 Using Your Trained Model in the Web App

The `object-scanner.html` app from the other part of this project is currently wired to expect **COCO-SSD's** specific output format. A custom YOLOv8 model:
- Has different class names (whatever you trained on, e.g. "circle", "square", "triangle" instead of COCO's 80 categories)
- May need a different loading method in the browser depending on the exported format (e.g. `tf.loadGraphModel()` instead of `cocoSsd.load()`)
- May output boxes in a different array shape and require some extra decoding logic in JavaScript

This means plugging in a custom model isn't a drop-in file swap — it requires updating the detection code in `object-scanner.html`. Once you've trained and exported a model you're happy with, share the exported model format and I'll write the updated JavaScript loading/decoding code to match.

---

## 🧠 What Each Script Actually Does

| File | What it does |
|---|---|
| `generate_dataset.py` | Draws random shapes (circle/square/triangle) on images and calculates their bounding boxes, saving both the image and a YOLO-format `.txt` label file for each one. This is what "labeled data" looks like under the hood. |
| `train.py` | Loads a small pretrained YOLOv8 model and fine-tunes it on your dataset — this is the actual "training" step. |
| `predict.py` | Loads your trained weights and runs them on new images, saving copies with predicted boxes/labels drawn on so you can see how well it learned. |
| `export.py` | Converts the trained PyTorch model into other formats (ONNX, TensorFlow.js, etc.) for use outside of Python. |

---

## 📄 Understanding YOLO Label Format

Each `.txt` label file has one line per object in the matching image:

```
class_id x_center y_center width height
```

All values except `class_id` are normalized between 0 and 1 (relative to image width/height). For example:

```
0 0.512500 0.481250 0.281250 0.281250
```

means: object of class `0` (circle), centered near the middle of the image, taking up about 28% of the image's width and height.
