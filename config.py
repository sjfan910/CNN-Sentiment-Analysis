"""
Configuration file for Facial Sentiment Analysis CNN
Contains all hyperparameters, paths, and training settings
"""

import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, LOG_DIR, VIZ_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# FER2013 contains 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
# We filter for only happy (3) and sad (4)
TARGET_EMOTIONS = {
    'happy': 3,
    'sad': 4
}
TARGET_SAMPLES_PER_CLASS = 11000  # Target balanced dataset

# Image specifications
IMG_HEIGHT = 48
IMG_WIDTH = 48
IMG_CHANNELS = 1  # Grayscale

# Data split
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
N_FOLDS = 5  # For cross-validation
RANDOM_SEED = 42

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
AUGMENTATION_CONFIG = {
    'rotation_range': 15,  # Slight rotation (degrees)
    'width_shift_range': 0.1,  # Horizontal shift
    'height_shift_range': 0.1,  # Vertical shift
    'horizontal_flip': True,  # Random horizontal flip
    'brightness_range': [0.8, 1.2],  # Brightness adjustment
    'zoom_range': 0.1,  # Slight zoom
    'fill_mode': 'nearest'
}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# Simple yet effective CNN architecture optimized for MacBook
MODEL_CONFIG = {
    'conv_blocks': [
        # Block 1: Initial feature extraction
        {'filters': 32, 'kernel_size': 3, 'activation': 'relu', 'pool_size': 2},
        # Block 2: Deeper features
        {'filters': 64, 'kernel_size': 3, 'activation': 'relu', 'pool_size': 2},
        # Block 3: High-level features
        {'filters': 128, 'kernel_size': 3, 'activation': 'relu', 'pool_size': 2},
    ],
    'dense_layers': [
        {'units': 128, 'activation': 'relu', 'dropout': 0.5},
    ],
    'output_activation': 'sigmoid',  # Binary classification
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Conservative hyperparameters for MacBook training
TRAINING_CONFIG = {
    # Basic settings
    'batch_size': 32,  # Balanced for MacBook memory
    'epochs': 100,
    'initial_lr': 0.001,  # Conservative learning rate

    # Optimizer
    'optimizer': 'adam',  # Adam optimizer with default beta values
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy'],

    # Learning rate scheduling
    'lr_scheduler': 'reduce_on_plateau',
    'lr_patience': 5,  # Reduce LR after 5 epochs without improvement
    'lr_factor': 0.5,  # Reduce LR by half
    'lr_min': 1e-7,  # Minimum learning rate

    # Early stopping
    'early_stopping': True,
    'early_stopping_patience': 15,  # Stop after 15 epochs without improvement
    'early_stopping_monitor': 'val_accuracy',
    'early_stopping_mode': 'max',

    # Model checkpointing
    'save_best_only': True,
    'checkpoint_monitor': 'val_accuracy',
    'checkpoint_mode': 'max',

    # Target performance
    'target_accuracy': 0.85,  # 85% validation accuracy target
}

# ============================================================================
# REAL-TIME INFERENCE CONFIGURATION
# ============================================================================
INFERENCE_CONFIG = {
    # Frame processing
    'process_every_n_frames': 5,  # Process every 5th frame (~6 FPS on typical webcam)
    'display_fps': True,

    # Face detection (Haar Cascade)
    'face_cascade': 'haarcascade_frontalface_default.xml',
    'detection_scale_factor': 1.1,
    'detection_min_neighbors': 5,
    'detection_min_size': (30, 30),

    # Display settings
    'show_confidence': True,
    'show_bounding_box': True,
    'confidence_threshold': 0.5,  # Threshold for sentiment classification

    # Colors (BGR format for OpenCV)
    'color_happy': (0, 255, 0),  # Green
    'color_sad': (0, 0, 255),  # Red
    'color_box': (255, 255, 255),  # White

    # Logging
    'log_predictions': True,
    'log_file': os.path.join(LOG_DIR, 'predictions.csv'),
}

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
# Automatically detect and use available hardware acceleration
DEVICE_CONFIG = {
    'use_mps': True,  # Use Metal Performance Shaders on Mac (if available)
    'mixed_precision': False,  # Mixed precision training (can speed up on supported hardware)
}

# ============================================================================
# DISPLAY MESSAGES
# ============================================================================
print("=" * 80)
print("FACIAL SENTIMENT ANALYSIS - CONFIGURATION LOADED")
print("=" * 80)
print(f"Target Classes: {list(TARGET_EMOTIONS.keys())}")
print(f"Image Size: {IMG_HEIGHT}x{IMG_WIDTH}x{IMG_CHANNELS}")
print(f"Batch Size: {TRAINING_CONFIG['batch_size']}")
print(f"Initial Learning Rate: {TRAINING_CONFIG['initial_lr']}")
print(f"Target Accuracy: {TRAINING_CONFIG['target_accuracy']*100}%")
print(f"Max Epochs: {TRAINING_CONFIG['epochs']}")
print("=" * 80)
