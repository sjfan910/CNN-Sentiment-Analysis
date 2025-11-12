"""
Training Module for Sentiment CNN
Implements training loop with early stopping, LR scheduling, and checkpointing
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import config
from src.data.augmentation import get_augmentation_generator, get_validation_generator


class SentimentTrainer:
    """
    Trainer class for Sentiment CNN with all training configurations
    """

    def __init__(self, model, training_config=None):
        """
        Initialize trainer

        Args:
            model: Compiled Keras model
            training_config: Dictionary with training parameters
        """
        self.model = model
        self.training_config = training_config if training_config else config.TRAINING_CONFIG
        self.history = None
        self.best_accuracy = 0.0

    def get_callbacks(self, fold=None):
        """
        Create training callbacks for early stopping, LR scheduling, and checkpointing

        Args:
            fold: Fold number for cross-validation (optional)

        Returns:
            List of Keras callbacks
        """
        callbacks = []

        # ====================================================================
        # EARLY STOPPING
        # ====================================================================
        # Stop training if validation accuracy doesn't improve
        if self.training_config['early_stopping']:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=self.training_config['early_stopping_monitor'],
                patience=self.training_config['early_stopping_patience'],
                mode=self.training_config['early_stopping_mode'],
                verbose=1,
                restore_best_weights=True  # Restore best weights when stopped
            )
            callbacks.append(early_stopping)
            print(f"Early Stopping: patience={self.training_config['early_stopping_patience']}, "
                  f"monitor={self.training_config['early_stopping_monitor']}")

        # ====================================================================
        # LEARNING RATE SCHEDULING
        # ====================================================================
        # Reduce learning rate when validation accuracy plateaus
        if self.training_config['lr_scheduler'] == 'reduce_on_plateau':
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor=self.training_config['early_stopping_monitor'],
                factor=self.training_config['lr_factor'],
                patience=self.training_config['lr_patience'],
                verbose=1,
                min_lr=self.training_config['lr_min']
            )
            callbacks.append(lr_scheduler)
            print(f"LR Scheduler: ReduceLROnPlateau, patience={self.training_config['lr_patience']}, "
                  f"factor={self.training_config['lr_factor']}")

        # ====================================================================
        # MODEL CHECKPOINTING
        # ====================================================================
        # Save best model during training
        checkpoint_filename = 'best_model.h5' if fold is None else f'best_model_fold_{fold}.h5'
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, checkpoint_filename)

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=self.training_config['checkpoint_monitor'],
            mode=self.training_config['checkpoint_mode'],
            save_best_only=self.training_config['save_best_only'],
            verbose=1
        )
        callbacks.append(model_checkpoint)
        print(f"Model Checkpoint: {checkpoint_path}")

        # ====================================================================
        # CSV LOGGER
        # ====================================================================
        # Log training metrics to CSV
        log_filename = 'training_log.csv' if fold is None else f'training_log_fold_{fold}.csv'
        log_path = os.path.join(config.LOG_DIR, log_filename)

        csv_logger = keras.callbacks.CSVLogger(log_path, append=False)
        callbacks.append(csv_logger)
        print(f"CSV Logger: {log_path}")

        # ====================================================================
        # CUSTOM CALLBACK FOR PROGRESS TRACKING
        # ====================================================================
        class ProgressCallback(keras.callbacks.Callback):
            """Custom callback to display training progress"""

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Train Loss: {logs.get('loss', 0):.4f} | Train Accuracy: {logs.get('accuracy', 0):.4f}")
                print(f"  Val Loss: {logs.get('val_loss', 0):.4f} | Val Accuracy: {logs.get('val_accuracy', 0):.4f}")

                # Check if target accuracy reached
                val_acc = logs.get('val_accuracy', 0)
                if val_acc >= config.TRAINING_CONFIG['target_accuracy']:
                    print(f"\n🎯 TARGET ACCURACY REACHED: {val_acc:.2%} >= {config.TRAINING_CONFIG['target_accuracy']:.2%}")

        callbacks.append(ProgressCallback())

        return callbacks

    def train(self, X_train, y_train, X_val, y_val, use_augmentation=True, fold=None):
        """
        Train the model

        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            use_augmentation: Whether to use data augmentation
            fold: Fold number for cross-validation (optional)

        Returns:
            Training history
        """
        print("\n" + "="*80)
        print(f"STARTING TRAINING {'(Fold ' + str(fold) + ')' if fold else ''}")
        print("="*80)
        print(f"Training samples: {len(y_train)}")
        print(f"Validation samples: {len(y_val)}")
        print(f"Batch size: {self.training_config['batch_size']}")
        print(f"Epochs: {self.training_config['epochs']}")
        print(f"Data augmentation: {use_augmentation}")
        print("="*80 + "\n")

        # Get callbacks
        callbacks = self.get_callbacks(fold=fold)

        # Training start time
        start_time = time.time()

        # Train with or without augmentation
        if use_augmentation:
            # Create augmentation generators
            train_datagen = get_augmentation_generator()
            val_datagen = get_validation_generator()  # No augmentation for validation

            # Fit generators
            train_datagen.fit(X_train)
            val_datagen.fit(X_val)

            # Train
            history = self.model.fit(
                train_datagen.flow(X_train, y_train, batch_size=self.training_config['batch_size']),
                steps_per_epoch=len(X_train) // self.training_config['batch_size'],
                validation_data=val_datagen.flow(X_val, y_val, batch_size=self.training_config['batch_size']),
                validation_steps=len(X_val) // self.training_config['batch_size'],
                epochs=self.training_config['epochs'],
                callbacks=callbacks,
                verbose=2  # One line per epoch
            )
        else:
            # Train without augmentation
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.training_config['batch_size'],
                validation_data=(X_val, y_val),
                epochs=self.training_config['epochs'],
                callbacks=callbacks,
                verbose=2
            )

        # Training end time
        end_time = time.time()
        training_duration = end_time - start_time

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Training duration: {training_duration/60:.2f} minutes")
        print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print("="*80 + "\n")

        self.history = history
        self.best_accuracy = max(history.history['val_accuracy'])

        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set

        Args:
            X_test: Test images
            y_test: Test labels

        Returns:
            Dictionary with test metrics
        """
        print("\n" + "="*80)
        print("EVALUATING MODEL ON TEST SET")
        print("="*80)

        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)

        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print("="*80 + "\n")

        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }

    def save_model(self, filepath=None):
        """
        Save the trained model

        Args:
            filepath: Path to save model (default: models/sentiment_model.h5)
        """
        if filepath is None:
            filepath = os.path.join(config.MODEL_DIR, 'sentiment_model.h5')

        print(f"\nSaving model to {filepath}...")
        self.model.save(filepath)
        print("Model saved successfully!")

        return filepath


def train_with_cross_validation(model_builder, X, y, n_folds=None):
    """
    Train model using K-fold cross-validation

    Args:
        model_builder: Function that returns a compiled model
        X: All training images
        y: All training labels
        n_folds: Number of folds (default: config value)

    Returns:
        List of training histories for each fold
    """
    from src.data.dataset_loader import get_k_fold_splits

    if n_folds is None:
        n_folds = config.N_FOLDS

    print("\n" + "="*80)
    print(f"TRAINING WITH {n_folds}-FOLD CROSS-VALIDATION")
    print("="*80 + "\n")

    # Get K-fold splits
    skf = get_k_fold_splits(X, y, n_splits=n_folds)

    fold_histories = []
    fold_accuracies = []

    # Train on each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}/{n_folds}")
        print('='*80)

        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Create new model for this fold
        model = model_builder()

        # Create trainer
        trainer = SentimentTrainer(model)

        # Train
        history = trainer.train(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            use_augmentation=True,
            fold=fold
        )

        fold_histories.append(history)
        fold_accuracies.append(max(history.history['val_accuracy']))

    # Print summary
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    for fold, acc in enumerate(fold_accuracies, start=1):
        print(f"Fold {fold}: {acc:.4f} ({acc*100:.2f}%)")
    print("-"*80)
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"Best Fold: {np.argmax(fold_accuracies) + 1} with {max(fold_accuracies):.4f}")
    print("="*80 + "\n")

    return fold_histories


if __name__ == "__main__":
    """
    Test training
    """
    from src.data.dataset_loader import FER2013Loader
    from src.models.cnn_model import create_model

    print("Testing training module...")

    # Load data
    loader = FER2013Loader()
    X_train, X_test, y_train, y_test = loader.prepare_data()

    # Create train/val split from training data
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # Create model
    model = create_model(print_summary=True)

    # Create trainer
    trainer = SentimentTrainer(model)

    # Train for a few epochs (test)
    print("\nRunning quick training test (3 epochs)...")
    config.TRAINING_CONFIG['epochs'] = 3  # Quick test
    history = trainer.train(X_train_split, y_train_split, X_val_split, y_val_split)

    # Evaluate
    results = trainer.evaluate(X_test, y_test)

    print("\nTraining test completed successfully!")
