# Real-Time Emotion Detection

A real-time facial emotion recognition system that detects faces from a webcam feed and classifies the emotion on each face into one of seven categories. A Convolutional Neural Network (CNN) trained on the FER-2013 dataset powers the predictions, while OpenCV handles live video capture and face detection.

## Demo

!![Uploading image.png…]()


> _Replace `demo.png` with a screenshot of the detector running on your own webcam feed._

## Tech Stack

- **Python** — core language
- **OpenCV** — webcam capture, Haar Cascade face detection, on-screen drawing
- **TensorFlow / Keras** — building, training, and running the CNN
- **CNN (Convolutional Neural Network)** — the model architecture
- **NumPy** — array handling and prediction smoothing

## How It Works

### 1. Data Preprocessing

- Images are loaded from per-emotion folders using Keras' `ImageDataGenerator`.
- Every image is converted to **grayscale** and resized to **48×48 pixels**.
- Pixel values are **rescaled from 0–255 to 0–1** so the network trains stably.
- Training images are **augmented** on the fly (random rotation, shifts, zoom, and horizontal flips) to expose the model to more variety and reduce overfitting.
- **Class weights** are computed to counter dataset imbalance (e.g. far more "happy" samples than "disgust"), so rare emotions aren't ignored during training.

### 2. CNN Architecture

The model is a `Sequential` stack of **three convolutional blocks** followed by a classifier head:

| Block | Layers | Filters |
|-------|--------|---------|
| Conv Block 1 | Conv2D → BatchNorm → MaxPooling → Dropout(0.25) | 64 |
| Conv Block 2 | Conv2D → BatchNorm → MaxPooling → Dropout(0.25) | 128 |
| Conv Block 3 | Conv2D → BatchNorm → MaxPooling → Dropout(0.25) | 256 |
| Classifier | Flatten → Dense(256) → BatchNorm → Dropout(0.5) → Dense(7, softmax) | — |

Each block learns progressively more complex patterns: **edges → shapes → facial features**. `BatchNormalization` stabilizes training, `MaxPooling` shrinks the feature maps while keeping the strongest signals, and `Dropout` randomly disables neurons to prevent the model from memorizing the training set. The final `softmax` layer outputs a probability for each of the 7 emotions.

The model is compiled with the **Adam** optimizer (learning rate `0.0001`) and **categorical cross-entropy** loss. Two callbacks keep training efficient: `EarlyStopping` (halts and restores the best weights when validation loss stops improving) and `ReduceLROnPlateau` (lowers the learning rate when progress stalls).

### 3. Training

- **Dataset:** [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Training images:** 28,709
- **Test images:** 7,178
- **Classes:** 7 emotions

### 4. Real-Time Inference

`emotion_detector.py` opens the webcam, detects faces with OpenCV's Haar Cascade, and for each face replicates the exact training preprocessing (grayscale → 48×48 → rescale). It then feeds the face to the model and draws the predicted emotion with a confidence percentage. Predictions are **smoothed over the last several frames** so the label stays steady instead of flickering, and the live **FPS** is displayed for performance monitoring.

## Model Accuracy

The model reaches approximately **56% validation accuracy** on FER-2013.

This is a solid result for this dataset — FER-2013 is notoriously difficult due to low-resolution 48×48 grayscale images, mislabeled samples, and genuinely ambiguous expressions. For reference, human accuracy on FER-2013 is estimated at around 65%.

## Emotions Detected

The model classifies faces into these **7 emotions**:

😠 Angry · 🤢 Disgust · 😨 Fear · 😄 Happy · 😐 Neutral · 😢 Sad · 😲 Surprise

## Installation & Usage

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Real-Time-Emotion-Detection
```

### 2. Create and activate a virtual environment

```bash
# Create the environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

Place the FER-2013 dataset so that `train/` and `test/` each contain one sub-folder per emotion, then run:

```bash
python train_model.py
```

This produces `emotion_model.h5` (the trained model) and `training_results.png` (accuracy/loss graphs).

> A pre-trained `emotion_model.h5` is already included, so you can skip this step and go straight to running the detector.

### 5. Run the real-time detector

```bash
python emotion_detector.py
```

A window opens showing your webcam feed with detected faces and predicted emotions. Press **`q`** to quit.

## Project Structure

```
Real-Time-Emotion-Detection/
├── train_model.py          # Builds and trains the CNN
├── emotion_detector.py     # Real-time webcam emotion detection
├── emotion_model.h5        # Trained model weights
├── training_results.png    # Accuracy/loss training graphs
├── requirements.txt        # Python dependencies
├── train/                  # Training images (7 emotion sub-folders)
└── test/                   # Test images (7 emotion sub-folders)
```

## Future Improvements

- **Deeper architectures** — experiment with ResNet or other deeper/residual networks to push accuracy beyond the current baseline.
- **Better datasets** — train on richer, higher-quality datasets such as AffectNet for more robust, real-world performance.
- **Face landmark features** — incorporate facial landmarks (eyes, eyebrows, mouth) as additional input to give the model more explicit cues about expression.
