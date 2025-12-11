# 📌 **Bird Sound Classification Using Deep Learning**

### **VGG19 • ResNet50 • Inception-V1 • Vision Transformer (ViT-B/16)**

---

## 📖 **Project Overview**

This project focuses on **automatic bird species classification** using **audio recordings** transformed into **Mel-spectrogram images**.
We evaluated **four deep learning architectures**—VGG19, ResNet50, Inception-V1, and ViT (Vision Transformer)—to determine which model performs best for audio spectrogram classification.

The project includes:

* Data cleaning and filtering
* Audio preprocessing
* Log-Mel spectrogram extraction
* Feature engineering (Δ and Δ² channels for Inception)
* Data augmentation (SpecAugment, MixUp)
* Training four deep learning models
* Full evaluation (Accuracy, F1, Confusion Matrix, AUC)
* Deployment-ready saved weights

---

## 🎧 **Dataset Description**

**Source:** BirdCLEF audio dataset (Xeno-Canto recordings)

### **Final Dataset After Filtering**

* **10 species**
* **Training set:** up to **500–1000 samples per class** (after balancing)
* **Validation set:** 968 samples
* **Test set:** 995 samples
* Each audio clip was standardized to:

  * **32 kHz sampling rate**
  * **Mono channel**
  * **5-second duration**
  * **RMS-normalized**

### **Feature Representation**

Each audio file was converted into:

✔ **128-band Log-Mel Spectrogram**
✔ For Inception: **3-channel representation**

* Original Mel
* Delta (Δ)
* Delta-Delta (Δ²)

---

## 🛠 **Preprocessing Pipeline**

### **1. Audio Standardization**

* Resampling → 32 kHz
* Converting to mono
* Padding/cropping → 5 seconds
* RMS normalization

### **2. Spectrogram Generation**

* Mel-spectrogram (128 × T)
* Log scaling
* Normalized to [0,1]

### **3. Data Augmentation**

* **SpecAugment** (ViT only):

  * Time masking
  * Frequency masking
* **MixUp** (optional for ViT)
* Random rotations & flips for image-based loaders
* Balanced classes up to the max class size

---

## 🧠 **Models Trained**

### **1. VGG19 (Custom 1-Channel Version)**

A deep convolutional baseline adapted for single-channel spectrograms.

### **2. ResNet50 (Fine-Tuned)**

* Pretrained on ImageNet
* Backbone frozen except Layer3
* Works with 3-channel spectrogram images

### **3. Inception-V1 (GoogLeNet)**

* 3-channel Mel + Delta + Delta²
* Multi-scale convolutions ideal for audio textures

### **4. Vision Transformer (ViT-B/16)**

* Strong global attention
* Added:

  * SpecAugment
  * Dropout
  * Label smoothing
  * AdamW + weight decay
  * ReduceLROnPlateau
  * Early stopping

---

## 📊 **Model Performance (Test Set)**

| Model            | Test Accuracy | Macro AUC      | Strength                                              |
| ---------------- | ------------- | -------------- | ----------------------------------------------------- |
| **VGG19**        | **70.0%**     | **~0.90–0.92** | Strong baseline, but limited generalization           |
| **ResNet50**     | **77.19%**    | **0.9670**     | Stable and powerful deep CNN                          |
| **Inception-V1** | **76.88%**    | **0.969**      | Multi-scale features help identify frequency patterns |
| **ViT-B/16**     | **78.79%**    | **0.969**      | Best accuracy, strong global attention                |

**🏆 Best Overall Model → ViT-B/16**
**🥈 Best CNN Model → ResNet50**


---

## 🧪 **Evaluation Metrics**

* Accuracy
* Precision / Recall / F1
* Macro AUC and ROC curves
* Confusion Matrix
* Per-class accuracy distribution
* Prediction histograms

---

## 🧩 **Key Findings**

* **ViT consistently outperformed all CNN models** thanks to its global attention.
* **Spectrogram augmentation (SpecAugment) boosted ViT accuracy significantly.**
* **ResNet50 generalized well** with minimal fine-tuning.
* **Inception-V1 performed strongly** due to its multi-scale filters.
* **VGG19 is a good baseline but does not scale well** with complex audio textures.

---
