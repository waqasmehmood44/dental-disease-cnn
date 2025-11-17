# 🦷 Dental Disease Classification CNN (MobileNetV2 Training Pipeline)

A high‑performance, optimized deep‑learning training pipeline for classifying dental diseases using TensorFlow/Keras. This project includes mixed‑precision training, XLA acceleration, stratified dataset splitting, data augmentation, and full training artifact logging.

---

## 🚀 Features

* **MobileNetV2** backbone for efficient image classification
* **Mixed precision training** (`float16` compute + `float32` outputs)
* **XLA JIT acceleration** for faster execution
* **Stratified train/validation/test splitting**
* **Advanced augmentations** (flip, rotation, zoom, contrast)
* **EarlyStopping**, **ModelCheckpoint**, and **CSVLogger**
* **Fast mode** for rapid experimentation
* **Automatic artifact saving:**

  * `best_model.keras`
  * `final_model.keras`
  * `labels.json`
  * `history.csv`
  * `training_log.csv`

---

## 📂 Project Structure

```
project-folder/
│
├── train_dental_cnn.py
├── test_dental_cnn.py  (optional, separate evaluation script)
│
├── dental_dataset/     (one folder per class)
│    ├── class_1/
│    ├── class_2/
│    └── ...
│
└── artifacts/
     ├── best_model.keras
     ├── final_model.keras
     ├── labels.json
     ├── history.csv
     └── training_log.csv
```

---

## 🧠 How the Training Script Works

### **1. Load Dataset**

Images are loaded from a root folder where **each class has its own subfolder**.
All standard image formats are supported (`.jpg`, `.png`, `.bmp`, `.webp`, etc.).

### **2. Stratified Split**

Dataset is split into:

* **Training**
* **Validation**
* (Test split is created but used in a separate evaluation script)

### **3. Data Pipeline**

* TensorFlow `tf.data.Dataset`
* Resizing & normalization
* Parallel loading
* Caching & prefetching

### **4. Model Architecture**

* MobileNetV2 (ImageNet‑pretrained when possible)
* GlobalAveragePooling
* Dropout
* Dense softmax output

### **5. Mixed Precision + XLA**

Enabled automatically for:

* Faster GPU usage
* Lower memory footprint

### **6. Training Modes**

#### Normal Mode

Full‑scale training using provided parameters.

#### Fast Mode (`--fast`)

* Fewer epochs
* Smaller image size
* Fewer batches per epoch
* Frozen base model
* Designed for experimentation or low‑end GPUs

### **7. Checkpoints & Logs**

* Saves best model based on `val_accuracy`
* Logs each epoch to CSV
* Saves training history separately

---

## ▶️ Usage

### **Basic Training Command**

```bash
python train_dental_cnn.py --data-root "path/to/dataset" --output "artifacts/"
```

### **Recommended Run (Normal Mode)**

```bash
python train_dental_cnn.py --data-root "./dental_dataset" --epochs 10 --img-size 160 --batch-size 64
```

### **Fast Mode (quick testing)**

```bash
python train_dental_cnn.py --data-root "./dental_dataset" --fast
```

### **Freeze Base Model**

```bash
python train_dental_cnn.py --freeze-base
```

### **Custom Learning Rate**

```bash
python train_dental_cnn.py --lr 0.00005
```

---

## 📄 Output Artifacts

Saved inside your `--output` directory:

* **best_model.keras** – best model based on validation accuracy
* **final_model.keras** – final epoch model
* **labels.json** – index-to-class mapping
* **training_log.csv** – epoch logs
* **history.csv** – clean training history

---

## 🖼 Dataset Requirements

```
root_directory/
│
├── Class_A/
│    ├── img001.jpg
│    ├── img002.jpg
│    └── ...
├── Class_B/
│    └── ...
└── Class_C/
```

Each folder = one dental disease category.

---

## 🧪 Evaluation

A separate script such as `test_dental_cnn.py` (not included here) should load and evaluate the **held-out test set**.

If you'd like, I can generate that script as well.

---

## 🔧 Customization

You can modify:

* Backbone model (ResNet, EfficientNet, etc.)
* Augmentation strength
* Loss function
* Mixed precision options
* Learning rate schedule
* Save more metrics (F1, precision, recall)

Just ask and I’ll help update the code.

---

## 📜 License

This project is intended for educational and research use.

---

## 💬 Support

Need help customizing the pipeline, adding TensorBoard support, exporting to TFLite, or creating a full inference script?

Just let me know!
