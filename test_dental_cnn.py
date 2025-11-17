#!/usr/bin/env python3
# ------------------------------------------------------------
# Predict Dental Diseases using a trained CNN model
#
# - Looks in: C:\Users\Waqas Mehmood\Downloads\Dental diseases_Model\test_images
# - Shows one image at a time (close window to go next)
# - Displays only the top prediction with probability
# ------------------------------------------------------------

import argparse, os, json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
from PIL import Image

# --- Performance setup ---
K.set_image_data_format("channels_last")
mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--image",
        type=str,
        help="Path to a single image to classify (overrides --dir)."
    )
    p.add_argument(
        "--dir",
        type=str,
        default=r"C:\Users\Waqas Mehmood\Downloads\Dental diseases_Model\test_images",
        help="Folder of images to classify one-by-one (recursive)."
    )
    p.add_argument(
        "--model",
        type=str,
        default=r"C:\Users\Waqas Mehmood\Downloads\artifacts\best_model.keras",
        help="Path to saved model (.keras)."
    )
    p.add_argument(
        "--labels",
        type=str,
        default=r"C:\Users\Waqas Mehmood\Downloads\artifacts\labels.json",
        help="Path to labels.json from training."
    )
    return p.parse_args()

# ---------- Utils ----------
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def load_class_names(labels_path: Path):
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found at: {labels_path}")
    mapping = json.loads(labels_path.read_text())
    first_key = next(iter(mapping.keys()))
    if isinstance(first_key, str):
        class_names = [mapping[str(i)] for i in range(len(mapping))]
    else:
        class_names = [mapping[i] for i in range(len(mapping))]
    return class_names

def infer_img_size(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return int(shape[1])

def load_and_preprocess(path: str, img_size: int):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, 0)

def predict_one(model, class_names, img_path, img_size):
    x = load_and_preprocess(img_path, img_size)
    probs = model.predict(x, verbose=0)[0].astype(np.float32)
    idx = int(np.argmax(probs))
    top_label = class_names[idx]
    top_prob = float(probs[idx])
    return top_label, top_prob

def walk_images(folder: Path):
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            yield str(p)

def show_image_with_pred(img_path: str, label: str, prob: float, title_prefix="Prediction"):
    img = Image.open(img_path).convert("RGB")
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{title_prefix}\n{label}: {prob:.4f}", fontsize=14)
    plt.show()  # Wait until window closed

# ---------- Main ----------
def main():
    args = parse_args()

    print(f">>> Loading model: {args.model}")
    model = keras.models.load_model(args.model)
    img_size = infer_img_size(model)
    class_names = load_class_names(Path(args.labels))
    print(f">>> Classes: {class_names}")
    print(f">>> Model input size: {img_size}x{img_size}")

    # Single image mode
    if args.image:
        img_path = args.image
        if not Path(img_path).exists():
            raise FileNotFoundError(img_path)
        label, prob = predict_one(model, class_names, img_path, img_size)
        print(f"\nImage: {img_path}\nPrediction: {label} ({prob:.4f})")
        show_image_with_pred(img_path, label, prob)
        return

    # Folder mode
    folder = Path(args.dir)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    print(f">>> Scanning folder: {folder}")
    paths = list(walk_images(folder))
    if not paths:
        raise SystemExit("No images found in the folder.")
    print(f">>> Found {len(paths)} images.\n")

    for i, img_path in enumerate(paths, 1):
        label, prob = predict_one(model, class_names, img_path, img_size)
        print(f"[{i}/{len(paths)}] {img_path}\n  Prediction: {label} ({prob:.4f})")
        show_image_with_pred(img_path, label, prob, title_prefix=f"Image {i}/{len(paths)}")

if __name__ == "__main__":
    main()
