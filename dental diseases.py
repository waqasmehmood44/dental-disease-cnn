#!/usr/bin/env python3
"""
Improved Dental Disease CNN Training Script
-------------------------------------------
✔ Adds support for "no_teeth" class (prevents false predictions)
✔ Stronger augmentation
✔ Label smoothing (reduces overconfident wrong predictions)
✔ Dropout + regularization for robustness
✔ Mixed precision + XLA for fast performance
✔ Stratified split + early stopping
"""

import argparse, os, json, random, csv
from pathlib import Path
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision

# -----------------------------
# PERFORMANCE SETTINGS
# -----------------------------
K.set_image_data_format("channels_last")
mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)

FAST_DEFAULTS = dict(
    img_size=128,
    epochs=3,
    batch_size=128,
    patience=1,
    freeze_base=True,
    train_steps=100,
    val_steps=25,
)

# -----------------------------
# ARGUMENT PARSER
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    default_data_root = r"C:\Users\Waqas Mehmood\Downloads\Dental diseases_Model"

    p.add_argument("--data-root", type=str, default=default_data_root)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--test-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--output", type=str, default=r"C:\Users\Waqas Mehmood\Downloads\artifacts")

    p.add_argument("--freeze-base", action="store_true")
    p.add_argument("--fast", action="store_true")

    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--val-steps", type=int, default=None)
    p.add_argument("--patience", type=int, default=2)

    return p.parse_args()

# -----------------------------
# DATA LOADING
# -----------------------------
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
        raise RuntimeError("No images found under dataset root!")

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

        n_test = max(1, int(n * test_split))
        n_val = max(1, int(n * val_split))

        test_items = items[:n_test]
        val_items = items[n_test:n_test+n_val]
        train_items = items[n_test+n_val:]

        test.extend((i, y) for i in test_items)
        val.extend((i, y) for i in val_items)
        train.extend((i, y) for i in train_items)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# -----------------------------
# DATA PIPELINE
# -----------------------------
def make_dataset(samples, img_size, batch, num_classes, shuffle=True):
    paths = [p for p,_ in samples]
    labels = [y for _,y in samples]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32) / 255.0

        # One-hot label (required for label smoothing)
        label = tf.one_hot(label, depth=num_classes)

        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 2000))

    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# -----------------------------
# MODEL BUILDING
# -----------------------------
def build_model(num_classes, img_size, lr, freeze_base):

    inputs = keras.Input(shape=(img_size, img_size, 3))

    # Stronger augmentation
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ])
    x = aug(inputs)

    # MobileNetV2 backbone
    base = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet",
        input_shape=(img_size, img_size, 3)
    )
    base.trainable = not freeze_base

    x = base(x, training=not freeze_base)
    x = layers.GlobalAveragePooling2D()(x)

    # Regularization to reduce overconfident predictions
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        kernel_regularizer=keras.regularizers.l2(1e-5)
    )(x)

    model = keras.Model(inputs, outputs)

    # Label smoothing greatly improves generalization
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    return model, base


# -----------------------------
# MAIN TRAINING LOGIC
# -----------------------------
def main():
    args = parse_args()

    if args.fast:
        args.img_size = FAST_DEFAULTS["img_size"]
        args.epochs = FAST_DEFAULTS["epochs"]
        args.batch_size = FAST_DEFAULTS["batch_size"]
        args.train_steps = FAST_DEFAULTS["train_steps"]
        args.val_steps = FAST_DEFAULTS["val_steps"]
        args.freeze_base = True

    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_root = Path(args.data_root)

    classes, paths, labels = list_images_with_labels(data_root)
    print("\nDetected classes:", classes)

    train, val, _test = stratified_split(paths, labels, args.val_split, args.test_split, args.seed)

    num_classes = len(classes)

    train_ds = make_dataset(train, args.img_size, args.batch_size, num_classes, shuffle=True)
    val_ds   = make_dataset(val, args.img_size, args.batch_size, num_classes, shuffle=False)

    if args.train_steps:
        train_ds = train_ds.take(args.train_steps)
    if args.val_steps:
        val_ds = val_ds.take(args.val_steps)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir/"labels.json", "w") as f:
        json.dump({i: name for i, name in enumerate(classes)}, f, indent=2)

    model, base = build_model(num_classes, args.img_size, args.lr, args.freeze_base)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=args.patience, monitor="val_accuracy", restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(out_dir/"best_model.keras", monitor="val_accuracy", save_best_only=True),
        keras.callbacks.CSVLogger(out_dir/"training_log.csv"),
    ]

    print("\nTraining Model...\n")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=callbacks, verbose=1
    )

    model.save(out_dir/"final_model.keras")

    print("\nTraining complete!")
    print(f"Artifacts saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
