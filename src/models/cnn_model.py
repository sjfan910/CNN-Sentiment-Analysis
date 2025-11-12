"""
Custom CNN Architecture for Binary Facial Sentiment Classification
Designed for efficiency on MacBook hardware while maintaining high accuracy
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import config


class SentimentCNN:
    """
    Custom CNN for binary sentiment classification (happy vs sad)

    Architecture Design Rationale:
    - Progressively increasing filters (32 -> 64 -> 128) for hierarchical feature learning
    - Small kernel size (3x3) for efficiency while capturing local patterns
    - MaxPooling for spatial dimension reduction and translation invariance
    - Dropout for regularization to prevent overfitting
    - Batch normalization for stable training and faster convergence
    - Simple architecture optimized for MacBook CPU/MPS training
    """

    def __init__(self, input_shape=None, model_config=None):
        """
        Initialize CNN architecture

        Args:
            input_shape: Tuple (height, width, channels)
            model_config: Dictionary with architecture parameters
        """
        if input_shape is None:
            input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)

        if model_config is None:
            model_config = config.MODEL_CONFIG

        self.input_shape = input_shape
        self.model_config = model_config
        self.model = None

    def build_model(self):
        """
        Build the CNN model architecture

        Returns:
            Compiled Keras model
        """
        print("\n" + "="*80)
        print("BUILDING CNN ARCHITECTURE")
        print("="*80)

        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='input_image')
        x = inputs

        # Convolutional blocks
        print("\nConvolutional Blocks:")
        for idx, block_config in enumerate(self.model_config['conv_blocks']):
            filters = block_config['filters']
            kernel_size = block_config['kernel_size']
            activation = block_config['activation']
            pool_size = block_config['pool_size']

            print(f"\nBlock {idx + 1}:")
            print(f"  Conv2D: {filters} filters, {kernel_size}x{kernel_size} kernel, {activation}")

            # Convolutional layer
            x = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation=activation,
                kernel_initializer='he_normal',  # Good for ReLU activation
                name=f'conv_{idx + 1}'
            )(x)

            # Batch normalization (improves training stability)
            x = layers.BatchNormalization(name=f'bn_{idx + 1}')(x)

            # Max pooling
            x = layers.MaxPooling2D(
                pool_size=pool_size,
                name=f'pool_{idx + 1}'
            )(x)
            print(f"  MaxPooling2D: {pool_size}x{pool_size}")
            print(f"  BatchNormalization")

        # Flatten
        x = layers.Flatten(name='flatten')(x)
        print("\nFlatten layer")

        # Dense layers
        print("\nDense Layers:")
        for idx, dense_config in enumerate(self.model_config['dense_layers']):
            units = dense_config['units']
            activation = dense_config['activation']
            dropout = dense_config['dropout']

            x = layers.Dense(
                units=units,
                activation=activation,
                kernel_initializer='he_normal',
                name=f'dense_{idx + 1}'
            )(x)
            print(f"  Dense: {units} units, {activation}")

            # Dropout for regularization
            if dropout > 0:
                x = layers.Dropout(dropout, name=f'dropout_{idx + 1}')(x)
                print(f"  Dropout: {dropout}")

        # Output layer (binary classification)
        outputs = layers.Dense(
            1,
            activation=self.model_config['output_activation'],
            name='output'
        )(x)
        print(f"\nOutput Layer:")
        print(f"  Dense: 1 unit, {self.model_config['output_activation']} activation")

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='SentimentCNN')

        print("\n" + "="*80)
        print("MODEL ARCHITECTURE COMPLETE")
        print("="*80)

        self.model = model
        return model

    def compile_model(self, optimizer=None, loss=None, metrics=None, learning_rate=None):
        """
        Compile the model with optimizer, loss, and metrics

        Args:
            optimizer: Optimizer name or instance
            loss: Loss function
            metrics: List of metrics
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation. Call build_model() first.")

        # Use config defaults if not specified
        if optimizer is None:
            optimizer = config.TRAINING_CONFIG['optimizer']
        if loss is None:
            loss = config.TRAINING_CONFIG['loss']
        if metrics is None:
            metrics = config.TRAINING_CONFIG['metrics']
        if learning_rate is None:
            learning_rate = config.TRAINING_CONFIG['initial_lr']

        # Create optimizer with learning rate
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer

        print("\n" + "="*80)
        print("COMPILING MODEL")
        print("="*80)
        print(f"Optimizer: {optimizer}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Loss: {loss}")
        print(f"Metrics: {metrics}")
        print("="*80)

        # Compile
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )

    def get_model_summary(self):
        """
        Print model summary
        """
        if self.model is None:
            raise ValueError("Model must be built first. Call build_model().")

        return self.model.summary()

    def count_parameters(self):
        """
        Count total and trainable parameters

        Returns:
            Dictionary with parameter counts
        """
        if self.model is None:
            raise ValueError("Model must be built first. Call build_model().")

        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params

        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }


def create_model(print_summary=True):
    """
    Convenience function to create and compile the model

    Args:
        print_summary: Whether to print model summary

    Returns:
        Compiled Keras model
    """
    # Create CNN
    cnn = SentimentCNN()

    # Build architecture
    model = cnn.build_model()

    # Compile
    cnn.compile_model()

    # Print summary
    if print_summary:
        print("\n" + "="*80)
        print("MODEL SUMMARY")
        print("="*80)
        cnn.get_model_summary()

        params = cnn.count_parameters()
        print("\n" + "="*80)
        print("PARAMETER COUNT")
        print("="*80)
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable parameters: {params['trainable']:,}")
        print(f"Non-trainable parameters: {params['non_trainable']:,}")
        print("="*80)

    return model


if __name__ == "__main__":
    """
    Test model creation
    """
    print("Testing CNN model creation...")

    # Create model
    model = create_model(print_summary=True)

    # Test with random input
    import numpy as np

    print("\nTesting forward pass with random input...")
    random_input = np.random.randn(1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS).astype(np.float32)
    output = model.predict(random_input, verbose=0)
    print(f"Input shape: {random_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output value: {output[0][0]:.4f} (should be between 0 and 1 for sigmoid)")

    print("\nModel creation test completed successfully!")
