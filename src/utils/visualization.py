"""
Visualization Tools for Training Metrics
Generates plots for accuracy, loss, confusion matrix, and classification report
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import config


def plot_training_history(history, save_path=None, fold=None):
    """
    Plot training and validation accuracy/loss curves

    Args:
        history: Keras History object or dict with 'loss', 'accuracy', etc.
        save_path: Path to save the plot (default: visualizations/)
        fold: Fold number for cross-validation (optional)

    Returns:
        Path to saved plot
    """
    # Extract history dict
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    epochs = range(1, len(history_dict['accuracy']) + 1)
    ax1.plot(epochs, history_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.axhline(y=config.TRAINING_CONFIG['target_accuracy'], color='g', linestyle='--',
                label=f"Target ({config.TRAINING_CONFIG['target_accuracy']:.0%})", linewidth=1.5)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Highlight best validation accuracy
    best_val_acc = max(history_dict['val_accuracy'])
    best_epoch = history_dict['val_accuracy'].index(best_val_acc) + 1
    ax1.plot(best_epoch, best_val_acc, 'r*', markersize=15, label=f'Best: {best_val_acc:.4f}')

    # Plot loss
    ax2.plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add fold info if provided
    if fold is not None:
        fig.suptitle(f'Training History - Fold {fold}', fontsize=16, fontweight='bold', y=1.02)
    else:
        fig.suptitle('Training History', fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save plot
    if save_path is None:
        filename = 'training_history.png' if fold is None else f'training_history_fold_{fold}.png'
        save_path = os.path.join(config.VIZ_DIR, filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")
    plt.close()

    return save_path


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, fold=None):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels (probabilities or binary)
        class_names: List of class names (default: ['Sad', 'Happy'])
        save_path: Path to save the plot
        fold: Fold number for cross-validation (optional)

    Returns:
        Path to saved plot
    """
    if class_names is None:
        class_names = ['Sad', 'Happy']

    # Convert probabilities to binary predictions if needed
    if y_pred.ndim > 1 or y_pred.max() <= 1.0:
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    else:
        y_pred_binary = y_pred.flatten()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={'size': 14})

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')

    if fold is not None:
        ax.set_title(f'Confusion Matrix - Fold {fold}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save plot
    if save_path is None:
        filename = 'confusion_matrix.png' if fold is None else f'confusion_matrix_fold_{fold}.png'
        save_path = os.path.join(config.VIZ_DIR, filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()

    return save_path


def generate_classification_report(y_true, y_pred, class_names=None, save_path=None, fold=None):
    """
    Generate and save classification report

    Args:
        y_true: True labels
        y_pred: Predicted labels (probabilities or binary)
        class_names: List of class names
        save_path: Path to save the report
        fold: Fold number for cross-validation (optional)

    Returns:
        Classification report string
    """
    if class_names is None:
        class_names = ['Sad', 'Happy']

    # Convert probabilities to binary predictions if needed
    if y_pred.ndim > 1 or y_pred.max() <= 1.0:
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    else:
        y_pred_binary = y_pred.flatten()

    # Generate report
    report = classification_report(y_true, y_pred_binary, target_names=class_names, digits=4)

    print("\n" + "="*80)
    if fold is not None:
        print(f"CLASSIFICATION REPORT - FOLD {fold}")
    else:
        print("CLASSIFICATION REPORT")
    print("="*80)
    print(report)
    print("="*80 + "\n")

    # Save report
    if save_path is None:
        filename = 'classification_report.txt' if fold is None else f'classification_report_fold_{fold}.txt'
        save_path = os.path.join(config.VIZ_DIR, filename)

    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        if fold is not None:
            f.write(f"CLASSIFICATION REPORT - FOLD {fold}\n")
        else:
            f.write("CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(report)
        f.write("\n" + "="*80 + "\n")

    print(f"Classification report saved to: {save_path}")

    return report


def plot_roc_curve(y_true, y_pred_proba, save_path=None, fold=None):
    """
    Plot ROC curve and calculate AUC

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
        fold: Fold number for cross-validation (optional)

    Returns:
        Path to saved plot, AUC score
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba.flatten())
    roc_auc = auc(fpr, tpr)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')

    if fold is not None:
        ax.set_title(f'ROC Curve - Fold {fold}', fontsize=14, fontweight='bold')
    else:
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    if save_path is None:
        filename = 'roc_curve.png' if fold is None else f'roc_curve_fold_{fold}.png'
        save_path = os.path.join(config.VIZ_DIR, filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to: {save_path}")
    plt.close()

    return save_path, roc_auc


def visualize_predictions(model, X_test, y_test, num_samples=16, save_path=None):
    """
    Visualize model predictions on sample images

    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels
        num_samples: Number of samples to visualize
        save_path: Path to save the plot

    Returns:
        Path to saved plot
    """
    # Get random samples
    indices = np.random.choice(len(X_test), size=num_samples, replace=False)
    X_samples = X_test[indices]
    y_samples = y_test[indices]

    # Make predictions
    y_pred = model.predict(X_samples, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()

    # Create figure
    rows = 4
    cols = num_samples // rows
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()

    # Plot each sample
    for i, ax in enumerate(axes):
        # Display image
        ax.imshow(X_samples[i, :, :, 0], cmap='gray')

        # Get true and predicted labels
        true_label = 'Happy' if y_samples[i] == 1 else 'Sad'
        pred_label = 'Happy' if y_pred_binary[i] == 1 else 'Sad'
        confidence = y_pred[i][0] if y_pred_binary[i] == 1 else 1 - y_pred[i][0]

        # Set title (green if correct, red if wrong)
        color = 'green' if y_samples[i] == y_pred_binary[i] else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2%})',
                    fontsize=9, color=color, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Prediction Examples', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save plot
    if save_path is None:
        save_path = os.path.join(config.VIZ_DIR, 'prediction_examples.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction examples saved to: {save_path}")
    plt.close()

    return save_path


def create_all_visualizations(model, history, X_test, y_test, fold=None):
    """
    Create all visualizations at once

    Args:
        model: Trained Keras model
        history: Training history
        X_test: Test images
        y_test: Test labels
        fold: Fold number (optional)

    Returns:
        Dictionary with paths to all saved plots
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")

    paths = {}

    # Training history
    paths['training_history'] = plot_training_history(history, fold=fold)

    # Get predictions
    y_pred = model.predict(X_test, verbose=0)

    # Confusion matrix
    paths['confusion_matrix'] = plot_confusion_matrix(y_test, y_pred, fold=fold)

    # Classification report
    paths['classification_report'] = generate_classification_report(y_test, y_pred, fold=fold)

    # ROC curve
    paths['roc_curve'], auc_score = plot_roc_curve(y_test, y_pred, fold=fold)
    print(f"AUC Score: {auc_score:.4f}")

    # Prediction examples (only for final model, not for each fold)
    if fold is None:
        paths['predictions'] = visualize_predictions(model, X_test, y_test)

    print("\n" + "="*80)
    print("VISUALIZATIONS COMPLETE")
    print("="*80 + "\n")

    return paths


if __name__ == "__main__":
    """
    Test visualization tools
    """
    from src.data.dataset_loader import FER2013Loader
    from src.models.cnn_model import create_model

    print("Testing visualization tools...")

    # Load data
    loader = FER2013Loader()
    X_train, X_test, y_train, y_test = loader.prepare_data()

    # Create and train a simple model (for testing)
    model = create_model(print_summary=False)

    # Create dummy history
    history = {
        'loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'accuracy': [0.75, 0.8, 0.85, 0.87, 0.9],
        'val_loss': [0.45, 0.38, 0.35, 0.33, 0.32],
        'val_accuracy': [0.78, 0.82, 0.86, 0.87, 0.88]
    }

    # Test visualizations
    print("\nTesting training history plot...")
    plot_training_history(history)

    print("\nTesting confusion matrix...")
    y_pred = model.predict(X_test[:100], verbose=0)
    plot_confusion_matrix(y_test[:100], y_pred)

    print("\nTesting classification report...")
    generate_classification_report(y_test[:100], y_pred)

    print("\nTesting ROC curve...")
    plot_roc_curve(y_test[:100], y_pred)

    print("\nVisualization tests completed successfully!")
