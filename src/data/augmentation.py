"""
Data Augmentation Module
Implements augmentation strategies to improve model generalization
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config


def get_augmentation_generator(augmentation_config=None):
    """
    Create ImageDataGenerator with augmentation settings

    Args:
        augmentation_config: Dictionary with augmentation parameters
                            (default: config.AUGMENTATION_CONFIG)

    Returns:
        ImageDataGenerator object configured for augmentation
    """
    if augmentation_config is None:
        augmentation_config = config.AUGMENTATION_CONFIG

    print("\nCreating data augmentation generator...")
    print("Augmentation settings:")
    for key, value in augmentation_config.items():
        print(f"  {key}: {value}")

    # Create generator
    datagen = ImageDataGenerator(
        rotation_range=augmentation_config.get('rotation_range', 15),
        width_shift_range=augmentation_config.get('width_shift_range', 0.1),
        height_shift_range=augmentation_config.get('height_shift_range', 0.1),
        horizontal_flip=augmentation_config.get('horizontal_flip', True),
        brightness_range=augmentation_config.get('brightness_range', [0.8, 1.2]),
        zoom_range=augmentation_config.get('zoom_range', 0.1),
        fill_mode=augmentation_config.get('fill_mode', 'nearest')
    )

    return datagen


def get_validation_generator():
    """
    Create ImageDataGenerator for validation/test (no augmentation)

    Returns:
        ImageDataGenerator object without augmentation
    """
    # No augmentation for validation/test data
    datagen = ImageDataGenerator()
    return datagen


def create_augmented_batches(X, y, batch_size, datagen):
    """
    Create batches with augmentation

    Args:
        X: numpy array of images
        y: numpy array of labels
        batch_size: Batch size
        datagen: ImageDataGenerator object

    Returns:
        Generator yielding (X_batch, y_batch) tuples
    """
    return datagen.flow(X, y, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    """
    Test augmentation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from src.data.dataset_loader import FER2013Loader

    print("Testing data augmentation...")

    # Load sample data
    loader = FER2013Loader()
    X_train, X_test, y_train, y_test = loader.prepare_data()

    # Get augmentation generator
    aug_gen = get_augmentation_generator()

    # Take a single sample
    sample_img = X_train[0:1]  # Shape: (1, 48, 48, 1)
    sample_label = y_train[0:1]

    # Generate augmented versions
    aug_gen.fit(sample_img)
    aug_iterator = aug_gen.flow(sample_img, batch_size=1)

    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    # Show original
    axes[0].imshow(sample_img[0, :, :, 0], cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Show 9 augmented versions
    for i in range(1, 10):
        aug_img = next(aug_iterator)[0]
        axes[i].imshow(aug_img[0, :, :, 0], cmap='gray')
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'{config.VIZ_DIR}/augmentation_examples.png', dpi=150, bbox_inches='tight')
    print(f"\nAugmentation examples saved to {config.VIZ_DIR}/augmentation_examples.png")
    plt.close()
