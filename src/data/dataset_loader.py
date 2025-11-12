"""
FER2013 Dataset Loader and Preprocessor
Handles downloading, filtering, and preprocessing the FER2013 dataset
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
import config


class FER2013Loader:
    """
    Loader for FER2013 dataset with filtering for happy/sad emotions
    """

    def __init__(self):
        self.raw_data_path = os.path.join(config.RAW_DATA_DIR, 'fer2013.csv')
        self.processed_data_path = config.PROCESSED_DATA_DIR
        self.img_size = (config.IMG_HEIGHT, config.IMG_WIDTH)
        self.target_emotions = config.TARGET_EMOTIONS

    def download_instructions(self):
        """
        Provides instructions for downloading FER2013 dataset
        Note: FER2013 requires manual download from Kaggle
        """
        instructions = """
        ========================================================================
        FER2013 DATASET DOWNLOAD INSTRUCTIONS
        ========================================================================

        The FER2013 dataset must be downloaded manually from Kaggle:

        1. Create a Kaggle account at https://www.kaggle.com

        2. Download the dataset from:
           https://www.kaggle.com/datasets/msambare/fer2013
           OR
           https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

        3. Place the 'fer2013.csv' file in:
           {}

        Alternative: Use Kaggle API
        ----------------------------
        If you have Kaggle API configured (~/.kaggle/kaggle.json):

        Run: kaggle datasets download -d msambare/fer2013
        Then: unzip fer2013.zip -d {}

        ========================================================================
        """.format(config.RAW_DATA_DIR, config.RAW_DATA_DIR)

        return instructions

    def load_raw_data(self):
        """
        Load raw FER2013 CSV file
        Returns: pandas DataFrame
        """
        if not os.path.exists(self.raw_data_path):
            print(self.download_instructions())
            raise FileNotFoundError(f"FER2013 dataset not found at {self.raw_data_path}")

        print(f"Loading FER2013 dataset from {self.raw_data_path}...")
        df = pd.read_csv(self.raw_data_path)
        print(f"Loaded {len(df)} samples")
        return df

    def filter_emotions(self, df):
        """
        Filter dataset for only happy and sad emotions

        FER2013 emotion labels:
        0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

        Args:
            df: pandas DataFrame with FER2013 data

        Returns:
            Filtered DataFrame with only happy (3) and sad (4) samples
        """
        print("\nFiltering for happy and sad emotions...")

        # Get emotion codes
        happy_code = self.target_emotions['happy']
        sad_code = self.target_emotions['sad']

        # Filter
        filtered_df = df[df['emotion'].isin([happy_code, sad_code])].copy()

        # Binary labels: happy=1, sad=0
        filtered_df['binary_label'] = (filtered_df['emotion'] == happy_code).astype(int)

        # Print statistics
        print(f"\nFiltered dataset statistics:")
        print(f"Total samples: {len(filtered_df)}")
        print(f"Happy samples: {(filtered_df['binary_label'] == 1).sum()}")
        print(f"Sad samples: {(filtered_df['binary_label'] == 0).sum()}")

        return filtered_df

    def parse_pixels(self, pixel_string):
        """
        Convert pixel string to numpy array

        Args:
            pixel_string: Space-separated pixel values (string)

        Returns:
            numpy array of shape (48, 48)
        """
        pixels = np.array([int(pixel) for pixel in pixel_string.split()])
        return pixels.reshape(self.img_size)

    def preprocess_dataset(self, df):
        """
        Convert pixel strings to numpy arrays and normalize

        Args:
            df: DataFrame with filtered FER2013 data

        Returns:
            X: numpy array of images (N, 48, 48, 1)
            y: numpy array of labels (N,)
        """
        print("\nPreprocessing images...")

        X = []
        y = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            # Parse pixels
            img = self.parse_pixels(row['pixels'])

            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0

            # Add channel dimension
            img = np.expand_dims(img, axis=-1)

            X.append(img)
            y.append(row['binary_label'])

        X = np.array(X)
        y = np.array(y)

        print(f"\nPreprocessed dataset shape: X={X.shape}, y={y.shape}")
        print(f"Data type: {X.dtype}, range: [{X.min():.3f}, {X.max():.3f}]")

        return X, y

    def balance_dataset(self, X, y, target_per_class=None):
        """
        Balance dataset by undersampling majority class

        Args:
            X: numpy array of images
            y: numpy array of labels
            target_per_class: Target number of samples per class (default: config value)

        Returns:
            Balanced X, y
        """
        if target_per_class is None:
            target_per_class = config.TARGET_SAMPLES_PER_CLASS

        print(f"\nBalancing dataset to {target_per_class} samples per class...")

        # Separate classes
        happy_idx = np.where(y == 1)[0]
        sad_idx = np.where(y == 0)[0]

        print(f"Before balancing: Happy={len(happy_idx)}, Sad={len(sad_idx)}")

        # Determine actual samples per class (take minimum of target and available)
        n_samples = min(target_per_class, len(happy_idx), len(sad_idx))

        # Random sampling with fixed seed for reproducibility
        np.random.seed(config.RANDOM_SEED)
        happy_sample = np.random.choice(happy_idx, size=n_samples, replace=False)
        sad_sample = np.random.choice(sad_idx, size=n_samples, replace=False)

        # Combine and shuffle
        balanced_idx = np.concatenate([happy_sample, sad_sample])
        np.random.shuffle(balanced_idx)

        X_balanced = X[balanced_idx]
        y_balanced = y[balanced_idx]

        print(f"After balancing: Total={len(y_balanced)}, Happy={(y_balanced==1).sum()}, Sad={(y_balanced==0).sum()}")

        return X_balanced, y_balanced

    def create_train_test_split(self, X, y, test_size=None, random_state=None):
        """
        Create stratified train/test split

        Args:
            X: numpy array of images
            y: numpy array of labels
            test_size: Fraction for test set (default: config value)
            random_state: Random seed (default: config value)

        Returns:
            X_train, X_test, y_train, y_test
        """
        if test_size is None:
            test_size = 1 - config.TRAIN_TEST_SPLIT

        if random_state is None:
            random_state = config.RANDOM_SEED

        print(f"\nCreating train/test split ({config.TRAIN_TEST_SPLIT:.0%} train, {test_size:.0%} test)...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        print(f"Training set: {len(y_train)} samples (Happy: {(y_train==1).sum()}, Sad: {(y_train==0).sum()})")
        print(f"Test set: {len(y_test)} samples (Happy: {(y_test==1).sum()}, Sad: {(y_test==0).sum()})")

        return X_train, X_test, y_train, y_test

    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """
        Save processed data to disk
        """
        print(f"\nSaving processed data to {self.processed_data_path}...")

        np.save(os.path.join(self.processed_data_path, 'X_train.npy'), X_train)
        np.save(os.path.join(self.processed_data_path, 'X_test.npy'), X_test)
        np.save(os.path.join(self.processed_data_path, 'y_train.npy'), y_train)
        np.save(os.path.join(self.processed_data_path, 'y_test.npy'), y_test)

        print("Data saved successfully!")

    def load_processed_data(self):
        """
        Load preprocessed data from disk

        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\nLoading processed data from {self.processed_data_path}...")

        X_train = np.load(os.path.join(self.processed_data_path, 'X_train.npy'))
        X_test = np.load(os.path.join(self.processed_data_path, 'X_test.npy'))
        y_train = np.load(os.path.join(self.processed_data_path, 'y_train.npy'))
        y_test = np.load(os.path.join(self.processed_data_path, 'y_test.npy'))

        print(f"Loaded training set: X={X_train.shape}, y={y_train.shape}")
        print(f"Loaded test set: X={X_test.shape}, y={y_test.shape}")

        return X_train, X_test, y_train, y_test

    def prepare_data(self, force_reprocess=False):
        """
        Main pipeline: load, filter, preprocess, balance, split, and save data

        Args:
            force_reprocess: If True, reprocess even if processed data exists

        Returns:
            X_train, X_test, y_train, y_test
        """
        # Check if processed data already exists
        processed_files = [
            'X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy'
        ]
        all_exist = all(
            os.path.exists(os.path.join(self.processed_data_path, f))
            for f in processed_files
        )

        if all_exist and not force_reprocess:
            print("Processed data found. Loading from disk...")
            return self.load_processed_data()

        print("Processing raw data...")

        # Load raw data
        df = self.load_raw_data()

        # Filter for happy/sad
        df_filtered = self.filter_emotions(df)

        # Preprocess
        X, y = self.preprocess_dataset(df_filtered)

        # Balance dataset
        X, y = self.balance_dataset(X, y)

        # Train/test split
        X_train, X_test, y_train, y_test = self.create_train_test_split(X, y)

        # Save processed data
        self.save_processed_data(X_train, X_test, y_train, y_test)

        return X_train, X_test, y_train, y_test


def get_k_fold_splits(X, y, n_splits=None, random_state=None):
    """
    Create K-fold cross-validation splits

    Args:
        X: numpy array of images
        y: numpy array of labels
        n_splits: Number of folds (default: config value)
        random_state: Random seed (default: config value)

    Returns:
        StratifiedKFold object
    """
    if n_splits is None:
        n_splits = config.N_FOLDS

    if random_state is None:
        random_state = config.RANDOM_SEED

    print(f"\nCreating {n_splits}-fold cross-validation splits...")

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    return skf


if __name__ == "__main__":
    """
    Test the data loader
    """
    loader = FER2013Loader()

    # Prepare data
    X_train, X_test, y_train, y_test = loader.prepare_data()

    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Image shape: {X_train.shape[1:]}")
    print(f"Class distribution (train): Happy={np.sum(y_train==1)}, Sad={np.sum(y_train==0)}")
    print("="*80)
