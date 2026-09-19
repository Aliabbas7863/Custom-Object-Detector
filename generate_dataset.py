"""
generate_dataset.py

Generates a small synthetic object-detection dataset (colored shapes on
random backgrounds) in YOLO format, so the training pipeline can be run
and verified end-to-end without needing to download anything first.

Classes: circle, square, triangle

Run:
    python generate_dataset.py

Output:
    dataset/images/train/*.jpg
    dataset/images/val/*.jpg
    dataset/labels/train/*.txt
    dataset/labels/val/*.txt
    dataset/data.yaml
"""

import os
import random
from PIL import Image, ImageDraw

random.seed(42)

CLASSES = ["circle", "square", "triangle"]
IMG_SIZE = 320
TRAIN_COUNT = 80
VAL_COUNT = 20
MIN_SHAPES = 1
MAX_SHAPES = 3
MIN_SHAPE_SIZE = 40
MAX_SHAPE_SIZE = 90

OUT_DIR = "dataset"

BG_COLORS = [
    (245, 245, 240), (230, 235, 245), (240, 230, 220),
    (225, 240, 230), (235, 225, 235), (250, 250, 250),
]

SHAPE_COLORS = [
    (220, 80, 60), (60, 120, 200), (60, 170, 100),
    (230, 170, 40), (150, 90, 200), (40, 40, 40),
]


def random_bg():
    return random.choice(BG_COLORS)


def draw_circle(draw, cx, cy, size, color):
    r = size / 2
    bbox = [cx - r, cy - r, cx + r, cy + r]
    draw.ellipse(bbox, fill=color)
    return bbox


def draw_square(draw, cx, cy, size, color):
    half = size / 2
    bbox = [cx - half, cy - half, cx + half, cy + half]
    draw.rectangle(bbox, fill=color)
    return bbox


def draw_triangle(draw, cx, cy, size, color):
    half = size / 2
    points = [
        (cx, cy - half),
        (cx - half, cy + half),
        (cx + half, cy + half),
    ]
    draw.polygon(points, fill=color)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    bbox = [min(xs), min(ys), max(xs), max(ys)]
    return bbox


DRAW_FUNCS = {
    "circle": draw_circle,
    "square": draw_square,
    "triangle": draw_triangle,
}


def generate_image(index, split):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), random_bg())
    draw = ImageDraw.Draw(img)

    n_shapes = random.randint(MIN_SHAPES, MAX_SHAPES)
    labels = []

    for _ in range(n_shapes):
        class_name = random.choice(CLASSES)
        class_id = CLASSES.index(class_name)
        size = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
        cx = random.randint(size, IMG_SIZE - size)
        cy = random.randint(size, IMG_SIZE - size)
        color = random.choice(SHAPE_COLORS)

        bbox = DRAW_FUNCS[class_name](draw, cx, cy, size, color)

        x_min, y_min, x_max, y_max = bbox
        x_center = ((x_min + x_max) / 2) / IMG_SIZE
        y_center = ((y_min + y_max) / 2) / IMG_SIZE
        w = (x_max - x_min) / IMG_SIZE
        h = (y_max - y_min) / IMG_SIZE

        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    img_path = os.path.join(OUT_DIR, "images", split, f"{split}_{index:04d}.jpg")
    label_path = os.path.join(OUT_DIR, "labels", split, f"{split}_{index:04d}.txt")

    img.save(img_path, quality=90)
    with open(label_path, "w") as f:
        f.write("\n".join(labels) + "\n")


def main():
    for split, count in [("train", TRAIN_COUNT), ("val", VAL_COUNT)]:
        os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)
        for i in range(count):
            generate_image(i, split)
        print(f"Generated {count} images for '{split}'")

    data_yaml = f"""# Auto-generated dataset config for YOLO training
path: {os.path.abspath(OUT_DIR)}
train: images/train
val: images/val

names:
  0: circle
  1: square
  2: triangle
"""
    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write(data_yaml)

    print(f"\nDataset ready in ./{OUT_DIR}/")
    print(f"Config written to ./{OUT_DIR}/data.yaml")


if __name__ == "__main__":
    main()
