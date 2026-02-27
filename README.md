# 🤚 Hand Gesture Classification
### Using MediaPipe Landmarks from the HaGRID Dataset

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-brightgreen)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Overview

This project classifies **18 hand gestures** in real time using **21 hand landmarks** extracted by [MediaPipe Hands](https://mediapipe.dev/) from the [HaGRID dataset](https://github.com/hukenovs/hagrid). Three machine learning models are trained and compared, with the best model deployed in a live webcam inference pipeline.

---

## 🤟 Gesture Classes

| | | | | | |
|---|---|---|---|---|---|
| call | dislike | fist | four | like | mute |
| no_gesture | ok | one | palm | peace | peace_inverted |
| rock | stop | stop_inverted | three | two_up | two_up_inverted |

---

## 🗂️ Repository Structure

```
Hand-Gesture-Classification/
│
├── ML1.ipynb                      # Main ML pipeline notebook
├── ML1_landmarks.ipynb            # Landmark extraction & labeling notebook
├── labeling.ipynb                 # Dataset labeling utilities
├── hand_landmark_visualization.py # Standalone landmark visualization script
├── hand_landmarks_data.csv        # Extracted landmark dataset (25,675 samples)
├── best_gesture_model.pkl         # Saved best model (SVM)
├── label_encoder.pkl              # Saved label encoder
└── .gitignore
```

---

## 🔄 Pipeline

```
HaGRID Images
      ↓
MediaPipe Hand Landmark Extraction  (21 landmarks × 3 axes = 63 features)
      ↓
Normalization  (translate to wrist origin → scale by fingertip distance)
      ↓
SMOTE Oversampling  (balance 6 minority classes)
      ↓
Model Training & Comparison  (Random Forest / SVM / KNN)
      ↓
Best Model: SVM  (99.08% test accuracy)
      ↓
Real-Time Webcam Inference  (20-frame sliding window)
```

---

## 📊 Model Performance

| Model | Train Acc | Test Acc | Overfit Gap | F1-Score |
|-------|-----------|----------|-------------|----------|
| Random Forest | 99.99% | 97.94% | 2.05% | 0.98 |
| **SVM (selected)** | **99.56%** | **99.08%** | **0.47%** | **0.99** |
| KNN | 100.00% | 98.11% | 1.89% | 0.98 |

> SVM with RBF kernel achieved the best test accuracy and the smallest overfit gap across all 18 classes.

---

## 🧠 Preprocessing

Two normalization steps are applied to every hand before training and inference:

1. **Translation** — Subtract the wrist landmark (index 0) from all X and Y coordinates, recentering the hand at the origin.
2. **Scale** — Divide all X and Y values by the mean Euclidean distance from the wrist to the 4 fingertips (landmarks 8, 12, 16, 20). This makes hands comparable regardless of distance from the camera.

> Z coordinates are used as-is — MediaPipe already normalizes depth relative to the wrist.

---

## 🚀 Getting Started

### Prerequisites

1. Clone the repository:
   ```bash
   git clone https://github.com/Ahmed-Tarek1/Hand-Gesture-Classification.git
   cd Hand-Gesture-Classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run the Notebook

3. Open `ML1.ipynb` in Jupyter or Google Colab and run all cells.

3. The trained model is saved as `best_gesture_model.pkl` and `label_encoder.pkl`.

### Real-Time Inference

The last section of `ML1.ipynb` opens your webcam and classifies gestures live:

- Predictions are **stabilized** using a 20-frame sliding window (mode of last 20 frames).
- Press **ESC** to exit.

---

## 📁 Dataset

- **Source:** [HaGRID](https://github.com/hukenovs/hagrid) — Hand Gesture Recognition Image Dataset
- **Landmarks:** Extracted using MediaPipe Hands (`ML1_landmarks.ipynb`)
- **Samples:** 25,675 rows × 64 columns (63 landmark features + 1 label)
- **Features:** `x1–x21`, `y1–y21`, `z1–z21` for the 21 hand landmarks

---

## 📈 Visualizations

### Hand Landmark Skeletons

One representative skeleton per gesture class (normalized):

> Run `hand_landmark_visualization.py` or the visualization cell in `ML1.ipynb` to generate the landmark plots.

### Confusion Matrix

Heatmap of true vs. predicted classes for the best model (SVM) on the test set.

---

## 🏗️ Key Design Decisions

- **SMOTE** was applied to 6 underrepresented classes (fist, mute, one, dislike, two_up, two_up_inverted), each upsampled to 1,600 samples, to address class imbalance.
- **GridSearchCV** with 5-fold cross-validation was used to tune the SVM hyperparameters (`C`, `gamma`).
- The **test set uses the original (unaugmented) distribution** to reflect real-world performance.
- The **inference pipeline mirrors the training normalization exactly** to avoid train/test mismatch.

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `mediapipe` | Hand landmark extraction |
| `opencv-python` | Webcam capture & frame processing |
| `scikit-learn` | Model training, evaluation, GridSearchCV |
| `imbalanced-learn` | SMOTE oversampling |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `joblib` | Model serialization |

---

## 👤 Author

**Ahmed Tarek**  
Hand Gesture Classification Project — MediaPipe + HaGRID  
