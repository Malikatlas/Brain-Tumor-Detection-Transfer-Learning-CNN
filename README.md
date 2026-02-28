# Brain Tumor Detection using CNN and Transfer Learning

Comparative performance analysis of a baseline CNN trained from scratch versus pre-trained deep learning models (ResNet50, DenseNet121, EfficientNet-B0) for multi-class brain tumor detection using MRI images.

---

## 🧠 Problem Statement

Accurate brain tumor classification from MRI scans is critical for early diagnosis and treatment planning.

This study evaluates:

- Baseline CNN (trained from scratch)
- ResNet50 (Transfer Learning)
- DenseNet121 (Transfer Learning)
- EfficientNet-B0 (Transfer Learning)

---

## 📊 Dataset

- Total Images: 2176 MRI scans
- Classes:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor
- Data Split: 70% Train / 15% Validation / 15% Test
- Image Size: 224×224
- Balanced dataset (no class imbalance)

---

## 🏗 Model Architectures

### 🔹 Baseline CNN
- 3 Convolutional layers (16 → 32 → 64 filters)
- MaxPooling layers
- Fully connected layer (512 neurons)
- Output layer (4 classes)

Observed heavy overfitting despite early stopping.

---

### 🔹 Transfer Learning Models
All pre-trained on ImageNet:

- ResNet50
- DenseNet121
- EfficientNet-B0

Training Strategy:
- Freeze backbone
- Replace classification head (4 classes)
- Train only final FC layer
- Optimizer: Adam
- Loss: CrossEntropy
- Epochs: 40
- Early stopping: patience = 5

---

## 📈 Results

### 🔥 Test Performance

| Model | Accuracy | F1-Score |
|--------|----------|----------|
| EfficientNet | **0.9271** | **0.9264** |
| DenseNet | 0.9174 | 0.9169 |
| ResNet | 0.9144 | 0.9133 |
| Baseline CNN | 0.9144 | 0.9138 |

EfficientNet achieved the highest validation accuracy (0.9297) and best generalization.

---

## 📊 Class-wise Observations

- **Pituitary** class achieved highest F1-scores across models.
- **Meningioma** consistently showed lowest precision/recall due to subtle visual overlap.
- EfficientNet demonstrated strongest and most stable class-wise performance.

---

## 🚀 Top-k Accuracy

EfficientNet:
- Top-1: 0.9633
- Top-2: 0.9969
- Top-3: 0.9969

Indicates extremely reliable predictions.

---

## ⚡ Efficiency Analysis

| Model | Trainable Params | Train Time | Inference Time |
|--------|----------------|------------|----------------|
| Baseline CNN | 25.7M | Slowest | Slowest |
| EfficientNet | 5K | Fast | Fastest |

Transfer learning drastically reduced computational cost while improving accuracy.

---

## 🧪 Explainable AI (XAI)

Implemented:

- Saliency Maps
- Grad-CAM Visualizations

Findings:
- EfficientNet focuses on tumor regions correctly in true positives.
- Misclassifications often occur due to feature overlap or ambiguous regions.

---

## 📂 Repository Structure

```
Brain-Tumor-Detection-Transfer-Learning-CNN/
│
├── Performance Analysis of Baseline CNN and Pre-Trained Deep learning models for multi-class Brain Tumor Detection.ipynb
├── Performance Analysis of Baseline CNN and Pre-Trained Deep learning models for multi-class Brain Tumor Detection.pdf
├── README.md
└── LICENSE
```

---

## 🛠 Tech Stack

- Python
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn

---

## 🎯 Key Contributions

✔ Comparative baseline vs transfer learning study  
✔ Demonstrated overfitting in scratch CNN  
✔ EfficientNet superior generalization  
✔ Computational efficiency analysis  
✔ Grad-CAM explainability integration  

---

## 🔮 Future Work

- Fine-tune deeper layers
- Add Vision Transformers (ViT)
- 3D MRI volume classification
- Ensemble methods
- Hybrid CNN-ViT models

---

## 📜 License

MIT License

---

## ⚠ Disclaimer

For academic and research purposes only.
