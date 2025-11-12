# Real-Time Facial Sentiment Analysis using CNN

A complete end-to-end system for real-time facial sentiment analysis that classifies expressions as "happy" or "sad" using a custom CNN trained on the FER2013 dataset.

## 🎯 Project Overview

This research-grade system combines computer vision and deep learning to:
- Train a custom CNN from scratch for binary sentiment classification
- Achieve **>85% validation accuracy** on FER2013 dataset
- Perform **real-time sentiment detection** via webcam with face detection
- Log predictions with timestamps for analysis
- Generate comprehensive training metrics and visualizations

**Optimized for MacBook** (CPU/MPS) with efficient architecture and conservative training parameters.

---

## 📋 Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
  - [Training](#training)
  - [Real-time Inference](#real-time-inference)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Results & Metrics](#results--metrics)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

### Training Pipeline
- ✅ **FER2013 Dataset Processing**: Automated filtering for happy/sad emotions (~11K images per class)
- ✅ **Data Augmentation**: Random flips, rotations, brightness adjustments, and zoom
- ✅ **5-Fold Cross-Validation Support**: Stratified K-fold for robust validation
- ✅ **Custom CNN Architecture**: Efficient design with 3 conv blocks + batch normalization
- ✅ **Smart Training**: Early stopping, LR scheduling (ReduceLROnPlateau), model checkpointing
- ✅ **Comprehensive Metrics**: Accuracy/loss curves, confusion matrix, ROC curve, classification report

### Real-Time Inference
- ✅ **Live Webcam Feed**: Process video at ~6 FPS with configurable frame skipping
- ✅ **Face Detection**: Haar Cascade for fast, reliable face detection
- ✅ **Sentiment Overlay**: Visual display with bounding boxes, labels, and confidence scores
- ✅ **Prediction Logging**: CSV logs with timestamps for analysis
- ✅ **Performance Metrics**: Real-time FPS and inference time display

---

## 💻 System Requirements

### Hardware
- **MacBook** (M1/M2 recommended, Intel supported)
- **Webcam** (built-in or external)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~2GB for dataset + models

### Software
- **Python**: 3.8 - 3.11 (3.9 or 3.10 recommended)
- **macOS**: 11.0 or later (for Metal GPU acceleration)

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CNN-Sentiment-Analysis.git
cd CNN-Sentiment-Analysis
```

### 2. Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Or using conda
conda create -n sentiment python=3.10
conda activate sentiment
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**For M1/M2 Mac GPU Acceleration** (optional but recommended):
```bash
pip install tensorflow-metal
```

### 4. Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
python -c "import cv2; print(f'OpenCV {cv2.__version__} installed')"
```

---

## 📊 Dataset Setup

### Option 1: Manual Download (Recommended)

1. **Create Kaggle Account**: https://www.kaggle.com
2. **Download FER2013**:
   - Visit: https://www.kaggle.com/datasets/msambare/fer2013
   - Click "Download"
3. **Place Dataset**:
   ```bash
   # Extract fer2013.csv to data/raw/
   mv ~/Downloads/fer2013.csv data/raw/
   ```

### Option 2: Kaggle API

1. **Set up Kaggle API**:
   ```bash
   pip install kaggle
   # Get API token from kaggle.com/account
   mkdir ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Download Dataset**:
   ```bash
   kaggle datasets download -d msambare/fer2013
   unzip fer2013.zip -d data/raw/
   ```

### Verify Dataset

```bash
ls -lh data/raw/fer2013.csv
# Should show ~70MB file
```

---

## 🚀 Usage

### Training

#### Basic Training
```bash
python train.py
```

This will:
1. Load and preprocess FER2013 dataset (filter happy/sad, balance classes)
2. Create train/validation/test splits (80/20 split)
3. Build custom CNN architecture
4. Train with data augmentation, early stopping, and LR scheduling
5. Evaluate on test set
6. Generate visualizations (accuracy/loss curves, confusion matrix, etc.)
7. Save best model to `models/sentiment_model.h5`

#### Training Options

```bash
# Force reprocess dataset
python train.py --reprocess

# Train without data augmentation
python train.py --no-augmentation

# Skip visualization generation
python train.py --no-viz

# Print detailed model summary
python train.py --verbose

# Combine options
python train.py --reprocess --verbose
```

#### Expected Training Time
- **MacBook M1/M2**: ~30-60 minutes (with Metal GPU)
- **MacBook Intel**: ~1-3 hours (CPU only)

#### Monitoring Training

Training progress is displayed in real-time:
```
Epoch 1/100
████████████████████ Train Loss: 0.4523 | Train Acc: 0.7845
                     Val Loss: 0.3912   | Val Acc: 0.8123
Epoch 2/100
████████████████████ Train Loss: 0.3456 | Train Acc: 0.8456
                     Val Loss: 0.3123   | Val Acc: 0.8567
...
🎯 TARGET ACCURACY REACHED: 85.67%
```

Logs are saved to:
- **CSV Log**: `logs/training_log.csv`
- **Checkpoints**: `models/checkpoints/best_model.h5`

---

### Real-Time Inference

#### Basic Usage
```bash
python run_realtime.py
```

This will:
1. Load the trained model (auto-detects best checkpoint)
2. Open your webcam
3. Detect faces in real-time
4. Display sentiment predictions with confidence scores
5. Log predictions to `logs/predictions.csv`

#### Inference Options

```bash
# Use specific model
python run_realtime.py --model models/sentiment_model.h5

# Use external webcam (camera index 1)
python run_realtime.py --camera 1

# Combine options
python run_realtime.py --model custom_model.h5 --camera 1
```

#### Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Save current frame to `visualizations/` |

#### Display Information

The live feed shows:
- **Bounding boxes** around detected faces
- **Sentiment label** (HAPPY / SAD)
- **Confidence score** (percentage)
- **FPS** (frames per second)
- **Inference time** (milliseconds)

#### Prediction Logs

Predictions are logged to `logs/predictions.csv`:
```csv
timestamp,frame_number,prediction,confidence,face_x,face_y,face_w,face_h
2024-11-12 14:32:15.123,145,happy,0.9234,120,80,150,150
2024-11-12 14:32:15.456,150,happy,0.9156,118,82,152,152
```

---

## 📁 Project Structure

```
CNN-Sentiment-Analysis/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration (paths, hyperparameters)
├── .gitignore               # Git ignore rules
│
├── train.py                 # Main training script
├── run_realtime.py          # Real-time inference script
│
├── data/                    # Dataset directory
│   ├── raw/                 # Raw FER2013 data
│   └── processed/           # Preprocessed .npy files
│
├── models/                  # Saved models
│   ├── sentiment_model.h5   # Final trained model
│   └── checkpoints/         # Training checkpoints
│       └── best_model.h5
│
├── logs/                    # Training and prediction logs
│   ├── training_log.csv
│   └── predictions.csv
│
├── visualizations/          # Generated plots and figures
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── prediction_examples.png
│
└── src/                     # Source code
    ├── data/
    │   ├── dataset_loader.py    # FER2013 loading & preprocessing
    │   └── augmentation.py      # Data augmentation
    │
    ├── models/
    │   └── cnn_model.py         # CNN architecture definition
    │
    ├── training/
    │   └── trainer.py           # Training loop & callbacks
    │
    ├── inference/
    │   ├── face_detector.py     # Haar Cascade face detection
    │   └── realtime_predictor.py # Real-time inference system
    │
    └── utils/
        └── visualization.py     # Plotting & metrics
```

---

## 🧠 Model Architecture

### CNN Design (Research-Grade)

The architecture is designed for **efficiency** and **interpretability**:

```
Input (48x48x1 grayscale)
    ↓
┌─────────────────────────────┐
│ Conv Block 1                │
│  • Conv2D(32, 3x3) + ReLU   │
│  • BatchNormalization       │
│  • MaxPooling2D(2x2)        │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Conv Block 2                │
│  • Conv2D(64, 3x3) + ReLU   │
│  • BatchNormalization       │
│  • MaxPooling2D(2x2)        │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Conv Block 3                │
│  • Conv2D(128, 3x3) + ReLU  │
│  • BatchNormalization       │
│  • MaxPooling2D(2x2)        │
└─────────────────────────────┘
    ↓
Flatten
    ↓
┌─────────────────────────────┐
│ Dense Layer                 │
│  • Dense(128) + ReLU        │
│  • Dropout(0.5)             │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Output Layer                │
│  • Dense(1) + Sigmoid       │
│  • Binary classification    │
└─────────────────────────────┘
```

**Total Parameters**: ~150K (trainable)

### Design Rationale

1. **Progressively Increasing Filters** (32→64→128): Hierarchical feature learning
2. **3x3 Kernels**: Standard for efficient local pattern detection
3. **Batch Normalization**: Stabilizes training, enables higher learning rates
4. **MaxPooling**: Spatial dimension reduction, translation invariance
5. **Dropout (0.5)**: Prevents overfitting on small dataset
6. **Sigmoid Output**: Binary probability (0=sad, 1=happy)

---

## ⚙️ Training Configuration

All hyperparameters are defined in `config.py` for reproducibility:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Batch Size** | 32 | Balanced for MacBook memory |
| **Initial LR** | 0.001 | Conservative starting point |
| **Optimizer** | Adam | Adaptive learning, good for CNNs |
| **Loss** | Binary Crossentropy | Standard for binary classification |
| **Max Epochs** | 100 | With early stopping |
| **Early Stopping Patience** | 15 | Stop if no improvement for 15 epochs |
| **LR Reduction Patience** | 5 | Reduce LR after 5 epochs plateau |
| **LR Reduction Factor** | 0.5 | Halve learning rate |
| **Target Accuracy** | 85% | Research requirement |

### Data Augmentation

| Augmentation | Range | Purpose |
|--------------|-------|---------|
| **Rotation** | ±15° | Slight head tilt invariance |
| **Horizontal Flip** | 50% | Mirror symmetry |
| **Width/Height Shift** | ±10% | Position invariance |
| **Brightness** | 80-120% | Lighting variations |
| **Zoom** | ±10% | Scale invariance |

---

## 📈 Results & Metrics

### Training Outputs

After training, the following outputs are generated:

#### 1. Visualizations (`visualizations/`)

- **Training History** (`training_history.png`)
  - Accuracy curves (train vs. validation)
  - Loss curves (train vs. validation)
  - Target accuracy threshold line

- **Confusion Matrix** (`confusion_matrix.png`)
  - True Positives, False Positives
  - True Negatives, False Negatives

- **ROC Curve** (`roc_curve.png`)
  - AUC score
  - True Positive Rate vs. False Positive Rate

- **Prediction Examples** (`prediction_examples.png`)
  - 16 sample predictions with ground truth
  - Confidence scores

#### 2. Classification Report (`visualizations/classification_report.txt`)

```
              precision    recall  f1-score   support

         Sad     0.8542    0.8634    0.8588      2200
       Happy     0.8621    0.8529    0.8575      2200

    accuracy                         0.8582      4400
   macro avg     0.8582    0.8582    0.8582      4400
weighted avg     0.8582    0.8582    0.8582      4400
```

#### 3. Training Logs (`logs/training_log.csv`)

Contains epoch-by-epoch metrics for analysis.

### Expected Performance

| Metric | Target | Expected Range |
|--------|--------|----------------|
| **Validation Accuracy** | >85% | 85-90% |
| **Test Accuracy** | >85% | 85-90% |
| **AUC Score** | >0.90 | 0.90-0.95 |
| **Inference Time** | <50ms | 20-40ms |
| **Real-time FPS** | ~6 FPS | 5-10 FPS |

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. **Dataset Not Found**

```
FileNotFoundError: FER2013 dataset not found at data/raw/fer2013.csv
```

**Solution**: Download FER2013 following [Dataset Setup](#dataset-setup)

#### 2. **Webcam Not Opening**

```
RuntimeError: Failed to open camera 0
```

**Solutions**:
- Grant camera permissions: System Preferences → Security & Privacy → Camera
- Try different camera index: `python run_realtime.py --camera 1`
- Check if another app is using the camera

#### 3. **Memory Errors**

```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions**:
- Reduce batch size in `config.py`: `TRAINING_CONFIG['batch_size'] = 16`
- Close other applications
- Use CPU instead of GPU (remove `tensorflow-metal`)

#### 4. **Low Accuracy (<85%)**

**Solutions**:
- Train longer: Increase `epochs` in `config.py`
- Ensure data augmentation is enabled: `python train.py` (without `--no-augmentation`)
- Check dataset balance: Verify ~11K samples per class
- Try different random seed: Change `RANDOM_SEED` in `config.py`

#### 5. **TensorFlow Metal Errors (M1/M2 Mac)**

```
ImportError: cannot import name 'metal' from 'tensorflow'
```

**Solutions**:
- Update TensorFlow: `pip install --upgrade tensorflow tensorflow-metal`
- If persists, remove Metal: `pip uninstall tensorflow-metal` (CPU will still work)

#### 6. **Face Not Detected**

**Solutions**:
- Ensure good lighting
- Face camera directly
- Adjust detection parameters in `config.py`:
  ```python
  INFERENCE_CONFIG['detection_min_neighbors'] = 3  # Less strict
  INFERENCE_CONFIG['detection_min_size'] = (20, 20)  # Smaller faces
  ```

---

## 🔧 Customization

### Modify Hyperparameters

Edit `config.py`:

```python
# Example: Increase batch size for faster training (if you have RAM)
TRAINING_CONFIG['batch_size'] = 64

# Example: More aggressive learning rate
TRAINING_CONFIG['initial_lr'] = 0.005

# Example: Longer training
TRAINING_CONFIG['epochs'] = 150
```

### Modify Model Architecture

Edit `src/models/cnn_model.py`:

```python
# Example: Add a 4th conv block
MODEL_CONFIG['conv_blocks'].append(
    {'filters': 256, 'kernel_size': 3, 'activation': 'relu', 'pool_size': 2}
)
```

### Adjust Real-time Performance

Edit `config.py`:

```python
# Process more frames for smoother predictions (slower)
INFERENCE_CONFIG['process_every_n_frames'] = 3  # ~10 FPS

# Process fewer frames for faster performance
INFERENCE_CONFIG['process_every_n_frames'] = 10  # ~3 FPS
```

---

## 📚 Research Context

This project is designed for **research purposes** with emphasis on:

1. **Reproducibility**: Fixed random seeds, documented hyperparameters
2. **Interpretability**: Simple architecture, clear design choices
3. **Documentation**: Comprehensive code comments and rationale
4. **Metrics**: Complete evaluation suite (confusion matrix, ROC, classification report)
5. **Efficiency**: Optimized for consumer hardware (MacBook)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **FER2013 Dataset**: Pierre-Luc Carrier and Aaron Courville
- **TensorFlow**: Google Brain Team
- **OpenCV**: Intel, Willow Garage

---

## 📞 Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting)
2. Open an issue on GitHub
3. Review code comments in source files

---

## 🚀 Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download FER2013 dataset to data/raw/fer2013.csv

# 3. Train model
python train.py

# 4. Run real-time inference
python run_realtime.py
```

**Enjoy your sentiment analysis system!** 🎉
