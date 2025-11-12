#!/usr/bin/env python3
"""
Real-time Facial Sentiment Analysis
Runs sentiment prediction on live webcam feed
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.inference.realtime_predictor import run_realtime_analysis


def main(args):
    """
    Main function for real-time inference
    """
    print("\n" + "="*80)
    print("REAL-TIME FACIAL SENTIMENT ANALYSIS")
    print("="*80 + "\n")

    # Determine model path
    if args.model:
        model_path = args.model
    else:
        # Try checkpoint first, then final model
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.h5')
        final_path = os.path.join(config.MODEL_DIR, 'sentiment_model.h5')

        if os.path.exists(checkpoint_path):
            model_path = checkpoint_path
            print(f"Using best checkpoint model: {model_path}")
        elif os.path.exists(final_path):
            model_path = final_path
            print(f"Using final model: {model_path}")
        else:
            print("ERROR: No trained model found!")
            print("\nPlease train the model first by running:")
            print("  python train.py")
            print("\nOr specify a model path with --model")
            sys.exit(1)

    # Run real-time analysis
    try:
        run_realtime_analysis(model_path=model_path, camera_index=args.camera)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("  1. Model is trained: python train.py")
        print("  2. Model path is correct: --model /path/to/model.h5")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run real-time facial sentiment analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (auto-detect if not specified)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (0 for built-in webcam)')

    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        sys.exit(0)
