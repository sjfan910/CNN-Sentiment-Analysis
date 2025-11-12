#!/usr/bin/env python3
"""
Main Training Script for Facial Sentiment Analysis CNN
Trains the model on FER2013 dataset and saves the best model
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data.dataset_loader import FER2013Loader
from src.models.cnn_model import create_model
from src.training.trainer import SentimentTrainer
from src.utils.visualization import create_all_visualizations


def setup_gpu():
    """
    Configure GPU/MPS settings for optimal training on Mac
    """
    print("\n" + "="*80)
    print("CONFIGURING HARDWARE ACCELERATION")
    print("="*80)

    # Check for MPS (Metal Performance Shaders) on Mac
    if config.DEVICE_CONFIG['use_mps']:
        try:
            # Try to use Metal on Mac
            physical_devices = tf.config.list_physical_devices()
            print("Available devices:")
            for device in physical_devices:
                print(f"  {device}")

            # On M1/M2 Macs, TensorFlow can use the GPU via Metal
            # This is automatically handled in newer TensorFlow versions
            print("\nNote: On M1/M2 Mac, install tensorflow-metal for GPU acceleration")
            print("  pip install tensorflow-metal")
        except Exception as e:
            print(f"Note: {e}")

    # Set memory growth to avoid OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"\nConfigured {len(gpus)} GPU(s) with memory growth")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

    print("="*80 + "\n")


def main(args):
    """
    Main training pipeline
    """
    print("\n" + "="*80)
    print("FACIAL SENTIMENT ANALYSIS - TRAINING PIPELINE")
    print("="*80)
    print(f"Start time: {tf.timestamp()}")
    print("="*80 + "\n")

    # Configure hardware
    setup_gpu()

    # ========================================================================
    # STEP 1: LOAD AND PREPARE DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*80 + "\n")

    loader = FER2013Loader()

    if args.reprocess:
        print("Force reprocessing data...")
        X_train, X_test, y_train, y_test = loader.prepare_data(force_reprocess=True)
    else:
        X_train, X_test, y_train, y_test = loader.prepare_data()

    # Further split training data into train/validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=config.RANDOM_SEED
    )

    print(f"\nFinal data split:")
    print(f"  Training: {len(y_train_split)} samples")
    print(f"  Validation: {len(y_val_split)} samples")
    print(f"  Test: {len(y_test)} samples")

    # ========================================================================
    # STEP 2: BUILD MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: MODEL ARCHITECTURE")
    print("="*80 + "\n")

    model = create_model(print_summary=args.verbose)

    # ========================================================================
    # STEP 3: TRAIN MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: MODEL TRAINING")
    print("="*80 + "\n")

    trainer = SentimentTrainer(model)

    history = trainer.train(
        X_train_split, y_train_split,
        X_val_split, y_val_split,
        use_augmentation=not args.no_augmentation
    )

    # ========================================================================
    # STEP 4: EVALUATE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: MODEL EVALUATION")
    print("="*80 + "\n")

    # Load best model from checkpoint
    best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.h5')
    if os.path.exists(best_model_path):
        print(f"Loading best model from checkpoint: {best_model_path}")
        from tensorflow import keras
        model = keras.models.load_model(best_model_path)
    else:
        print("Warning: Best model checkpoint not found, using final model")

    results = trainer.evaluate(X_test, y_test)

    # Check if target accuracy reached
    if results['test_accuracy'] >= config.TRAINING_CONFIG['target_accuracy']:
        print(f"\n✓ TARGET ACCURACY ACHIEVED!")
        print(f"  Test Accuracy: {results['test_accuracy']:.2%}")
        print(f"  Target: {config.TRAINING_CONFIG['target_accuracy']:.2%}")
    else:
        print(f"\n⚠ Target accuracy not yet reached")
        print(f"  Test Accuracy: {results['test_accuracy']:.2%}")
        print(f"  Target: {config.TRAINING_CONFIG['target_accuracy']:.2%}")
        print(f"  Shortfall: {(config.TRAINING_CONFIG['target_accuracy'] - results['test_accuracy'])*100:.2f}%")

    # ========================================================================
    # STEP 5: GENERATE VISUALIZATIONS
    # ========================================================================
    if not args.no_viz:
        print("\n" + "="*80)
        print("STEP 5: GENERATING VISUALIZATIONS")
        print("="*80 + "\n")

        viz_paths = create_all_visualizations(model, history, X_test, y_test)

        print("\nGenerated visualizations:")
        for name, path in viz_paths.items():
            print(f"  {name}: {path}")

    # ========================================================================
    # STEP 6: SAVE FINAL MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: SAVING FINAL MODEL")
    print("="*80 + "\n")

    final_model_path = trainer.save_model()

    print(f"\nFinal model saved to: {final_model_path}")

    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review visualizations in: visualizations/")
    print("  2. Check training logs in: logs/")
    print("  3. Run real-time inference:")
    print("     python run_realtime.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Facial Sentiment Analysis CNN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--reprocess', action='store_true',
                       help='Force reprocessing of dataset')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed model summary')

    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
