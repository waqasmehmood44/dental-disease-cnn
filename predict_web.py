#!/usr/bin/env python3
# ------------------------------------------------------------
# Web App: Dental Disease Classifier (Two-Stage Model)
#
# Stage 1 → Teeth Detector (MobileNetV2, 160x160)
# Stage 2 → Disease Classifier (Your CNN)
#
# ------------------------------------------------------------

import os
import json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision

from flask import Flask, request, redirect, url_for, render_template_string, flash
from werkzeug.utils import secure_filename

# ------------------------------------------------------------
# PERFORMANCE SETTINGS
# ------------------------------------------------------------
K.set_image_data_format("channels_last")
mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)

# ------------------------------------------------------------
# MODEL PATHS
# ------------------------------------------------------------
BASE_PROJECT = Path(r"D:\Machine Learning Projects\dental_diseases_detection")

DISEASE_MODEL_PATH = BASE_PROJECT / "artifacts" / "best_model.keras"
LABELS_PATH        = BASE_PROJECT / "artifacts" / "labels.json"
TEETH_MODEL_PATH   = BASE_PROJECT / "artifacts" / "teeth_detector.keras"

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# UTIL FUNCTIONS
# ------------------------------------------------------------

def load_class_names(labels_path: Path):
    mapping = json.loads(labels_path.read_text())
    return [mapping[str(i)] for i in range(len(mapping))]

def infer_img_size(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return int(shape[1])

def preprocess_image(path: str, img_size: int):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, 0)

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS

# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------
print(">>> Loading Disease Model...")
DISEASE_MODEL = keras.models.load_model(DISEASE_MODEL_PATH)
DISEASE_IMG_SIZE = infer_img_size(DISEASE_MODEL)
CLASS_NAMES = load_class_names(LABELS_PATH)

print(">>> Loading Teeth Detector...")
TEETH_MODEL = keras.models.load_model(TEETH_MODEL_PATH)
TEETH_IMG_SIZE = infer_img_size(TEETH_MODEL)

print(f"✔ Disease model input size: {DISEASE_IMG_SIZE}")
print(f"✔ Teeth detector input size: {TEETH_IMG_SIZE}")

# ------------------------------------------------------------
# FLASK APP
# ------------------------------------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.secret_key = "change-this-key"

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dental Disease Classifier</title>
  <style>
    body { font-family: Arial; background:#f5f5f5; margin:0; padding:0; }
    .container {
      max-width:800px; margin:auto; background:white; padding:2rem;
      margin-top:2rem; border-radius:10px;
      box-shadow:0 2px 8px rgba(0,0,0,0.1);
    }
    h1 { text-align:center; }
    .upload-form { text-align:center; margin-top:1.5rem; }
    .result { margin-top:2rem; text-align:center; }
    img { max-width:100%; border-radius:8px; margin-top:1rem; }
    .pred-box {
      display:inline-block; padding:0.6rem 1rem; background:#e8f5e9;
      border-radius:8px; font-weight:bold;
    }
    .error-box {
      display:inline-block; padding:0.6rem 1rem; background:#ffebee;
      border-radius:8px; font-weight:bold; color:#b71c1c;
    }
  </style>
</head>
<body>
<div class="container">
  <h1>Dental Disease Classifier</h1>

  <form class="upload-form" method="post" action="{{ url_for('predict') }}" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*" required>
    <button type="submit">Upload & Predict</button>
  </form>

  {% if filename %}
  <div class="result">
    <h2>Result</h2>

    {% if is_error %}
      <div class="error-box">{{ label }} ({{ prob|round(4) }})</div>
    {% else %}
      <div class="pred-box">{{ label }} ({{ prob|round(4) }})</div>
    {% endif %}

    <img src="{{ url_for('static', filename='uploads/' + filename) }}">
  </div>
  {% endif %}
</div>
</body>
</html>
"""

# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE_TEMPLATE, filename=None)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file uploaded.")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file type.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    save_path = UPLOAD_FOLDER / filename
    file.save(save_path)

    # ------------------------------------------------------------
    # STAGE 1 — TEETH DETECTION
    # ------------------------------------------------------------
    x_teeth = preprocess_image(str(save_path), TEETH_IMG_SIZE)
    det_prob = float(TEETH_MODEL.predict(x_teeth, verbose=0)[0][0])

    print(f"[Teeth Detector] Probability: {det_prob:.4f}")

    # Threshold can be tuned (0.60 recommended)
    if det_prob < 0.60:
        return render_template_string(
            PAGE_TEMPLATE,
            filename=filename,
            label="❌ No teeth detected — cannot classify disease",
            prob=det_prob,
            is_error=True
        )

    # ------------------------------------------------------------
    # STAGE 2 — DISEASE CLASSIFICATION
    # ------------------------------------------------------------
    x_disease = preprocess_image(str(save_path), DISEASE_IMG_SIZE)
    probs = DISEASE_MODEL.predict(x_disease, verbose=0)[0].astype(np.float32)
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    prob = float(probs[idx])

    return render_template_string(
        PAGE_TEMPLATE,
        filename=filename,
        label=label,
        prob=prob,
        is_error=False
    )

# ------------------------------------------------------------
# RUN APP
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
