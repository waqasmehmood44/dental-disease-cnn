#!/usr/bin/env python3
"""
Train Dental Diseases CNN (MobileNetV2 + mixed precision)

- Mixed precision + XLA JIT for speed
- Optional fast mode (smaller img, fewer epochs, cap steps, freeze base)
- Stratified train/val/test split
- EarlyStopping + ModelCheckpoint + CSVLogger
- Saves artifacts in --output (model, logs, labels)
"""

import argparse, os, json, random, itertools, csv
from pathlib import Path
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

# --- Performance switches ---
K.set_image_data_format("channels_last")
mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)

# --- Fast-mode defaults ---
FAST_DEFAULTS = dict(
    img_size=128,
    epochs=3,
    batch_size=128,
    patience=1,
    freeze_base=True,
    train_steps=100,
    val_steps=25,
)

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    default_data_root = r"C:\Users\Waqas Mehmood\Downloads\Dental diseases_Model"
    p.add_argument("--data-root", type=str, default=default_data_root,
                   help=f'Root folder with one subfolder per class (default: "{default_data_root}").')
    p.add_argument("--val-split", type=float, default=0.1, help="Validation fraction (stratified).")
    p.add_argument("--test-split", type=float, default=0.1, help="Test fraction (stratified).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    p.add_argument("--img-size", type=int, default=160, help="Square image size (pixels).")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    # UPDATED default output directory -> Downloads\artifacts
    p.add_argument("--output", type=str, default=r"C:\Users\Waqas Mehmood\Downloads\artifacts",
                   help="Directory for saved models/metrics.")
    p.add_argument("--freeze-base", action="store_true",
                   help="Warmup with base frozen then fine-tune (or train head-only if --fast).")
    # Speed flags
    p.add_argument("--fast", action="store_true",
                   help="Fast mode: fewer epochs, smaller images, freeze base, cap steps.")
    p.add_argument("--train-steps", type=int, default=None,
                   help="Limit training batches per epoch (overrides fast defaults).")
    p.add_argument("--val-steps", type=int, default=None,
                   help="Limit validation batches per epoch (overrides fast defaults).")
    p.add_argument("--patience", type=int, default=2,
                   help="EarlyStopping patience (epochs without val_accuracy improvement).")
    return p.parse_args()

# ---------------------------
# Data utilities
# ---------------------------
def list_images_with_labels(root: Path):
    root = Path(root)
    classes = [d.name for d in sorted(root.iterdir()) if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No class folders found under: {root}")
    paths, labels = [], []
    for idx, cls in enumerate(classes):
        cls_dir = root / cls
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"):
            for f in cls_dir.rglob(ext):
                paths.append(str(f))
                labels.append(idx)
    if not paths:
        raise RuntimeError(f"No images found under: {root}")
    return classes, paths, labels

def stratified_split(paths, labels, val_split, test_split, seed):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for p, y in zip(paths, labels):
        by_class[y].append(p)

    train, val, test = [], [], []
    for y, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * test_split))) if n > 10 else max(1, int(n * test_split))
        n_val  = max(1, int(round(n * val_split)))  if n > 10 else max(1, int(n * val_split))
        n_test = min(n_test, n - 2) if n >= 3 else min(n_test, n - 1)
        n_val  = min(n_val,  n - n_test - 1) if n - n_test >= 2 else 0

        test_items = items[:n_test]
        val_items  = items[n_test:n_test+n_val]
        train_items= items[n_test+n_val:]
        test.extend((p,y) for p in test_items)
        val.extend((p,y) for p in val_items)
        train.extend((p,y) for p in train_items)

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test

def make_dataset(samples, img_size, batch, shuffle=True):
    paths = [p for p,_ in samples]
    labels = [y for _,y in samples]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.cast(label, tf.int32)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 2000), seed=123)

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)

    return ds.batch(batch, drop_remainder=False).cache().prefetch(tf.data.AUTOTUNE)

# ---------------------------
# Model
# ---------------------------
def build_model(num_classes, img_size, lr, freeze_base):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="images_rgb")

    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    x = aug(inputs)

    try:
        base = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet",
            input_shape=(img_size, img_size, 3)
        )
    except ValueError as e:
        print("[WARN] ImageNet weights mismatch. Using random init. Details:", e)
        base = keras.applications.MobileNetV2(
            include_top=False, weights=None,
            input_shape=(img_size, img_size, 3)
        )

    base.trainable = not freeze_base
    x = base(x, training=not freeze_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model, base

def save_history_csv(history, out_csv):
    keys = list(history.history.keys())
    rows = zip(*[history.history[k] for k in keys])
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for r in rows:
            writer.writerow(r)

# ---------------------------
# Main (train only)
# ---------------------------
def main():
    args = parse_args()

    # Fast defaults (allow CLI overrides)
    if args.fast:
        args.img_size   = FAST_DEFAULTS["img_size"]    if args.img_size == 160 else args.img_size
        args.epochs     = FAST_DEFAULTS["epochs"]      if args.epochs == 8 else args.epochs
        args.batch_size = FAST_DEFAULTS["batch_size"]  if args.batch_size == 64 else args.batch_size
        args.patience   = FAST_DEFAULTS["patience"]    if args.patience == 2 else args.patience
        if not args.freeze_base:
            args.freeze_base = FAST_DEFAULTS["freeze_base"]
        if args.train_steps is None:
            args.train_steps = FAST_DEFAULTS["train_steps"]
        if args.val_steps is None:
            args.val_steps = FAST_DEFAULTS["val_steps"]

    # Reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_root = Path(args.data_root)
    classes, paths, labels = list_images_with_labels(data_root)
    print(f"Found {len(paths)} images across {len(classes)} classes: {classes}")
    print("Keras image data format:", K.image_data_format())
    print("Mixed precision policy:", mixed_precision.global_policy())
    print("XLA JIT:", tf.config.optimizer.get_jit())

    train, val, _test_unused = stratified_split(paths, labels, args.val_split, args.test_split, args.seed)
    print(f"Split sizes -> train: {len(train)}, val: {len(val)} (test kept for separate script)")

    train_ds = make_dataset(train, args.img_size, args.batch_size, shuffle=True)
    val_ds   = make_dataset(val,   args.img_size, args.batch_size, shuffle=False)

    if args.train_steps: train_ds = train_ds.take(args.train_steps)
    if args.val_steps:   val_ds   = val_ds.take(args.val_steps)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/"labels.json","w") as f:
        json.dump({i: name for i, name in enumerate(classes)}, f, indent=2)

    model, base = build_model(len(classes), args.img_size, args.lr, args.freeze_base)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=args.patience, monitor="val_accuracy",
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(out_dir/"best_model.keras",
                                        monitor="val_accuracy", save_best_only=True),
        keras.callbacks.CSVLogger(out_dir/"training_log.csv"),
    ]

    total_epochs = args.epochs
    if args.freeze_base:
        print("Training with base frozen (fast mode or --freeze-base)…")
        history = model.fit(train_ds, validation_data=val_ds,
                            epochs=total_epochs, callbacks=callbacks, verbose=1)
    else:
        half = max(1, total_epochs//2)
        print("Warmup with base frozen…")
        base.trainable = False
        hist1 = model.fit(train_ds, validation_data=val_ds, epochs=half, callbacks=callbacks, verbose=1)
        base.trainable = True
        model.compile(optimizer=keras.optimizers.Adam(args.lr/10),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        print("Fine-tuning base…")
        hist2 = model.fit(train_ds, validation_data=val_ds,
                          epochs=total_epochs - len(hist1.history["loss"]),
                          callbacks=callbacks, verbose=1)
        history = hist1
        for k, v in hist2.history.items():
            history.history[k].extend(v)

    # Save final model & history
    model.save(out_dir/"final_model.keras")
    save_history_csv(history, out_dir/"history.csv")

    print(f"\nArtifacts saved to: {out_dir.resolve()}")
    print(" - best_model.keras / final_model.keras")
    print(" - labels.json, history.csv, training_log.csv")
    print("NOTE: Run test_dental_cnn.py to evaluate on the held-out test split.")

if __name__ == "__main__":
    main()
