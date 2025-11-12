"""
Real-time Sentiment Analysis System
Captures webcam video, detects faces, and predicts sentiment in real-time
"""

import os
import cv2
import numpy as np
import time
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import config
from src.inference.face_detector import FaceDetector


class RealtimeSentimentPredictor:
    """
    Real-time sentiment prediction system with webcam input
    """

    def __init__(self, model_path, process_every_n_frames=None, log_predictions=None):
        """
        Initialize real-time predictor

        Args:
            model_path: Path to trained model file
            process_every_n_frames: Process every Nth frame (default: config value)
            log_predictions: Whether to log predictions (default: config value)
        """
        # Load configuration
        if process_every_n_frames is None:
            process_every_n_frames = config.INFERENCE_CONFIG['process_every_n_frames']
        if log_predictions is None:
            log_predictions = config.INFERENCE_CONFIG['log_predictions']

        self.process_every_n_frames = process_every_n_frames
        self.log_predictions = log_predictions
        self.log_file = config.INFERENCE_CONFIG['log_file']

        # Load model
        print("="*80)
        print("INITIALIZING REAL-TIME SENTIMENT PREDICTOR")
        print("="*80)
        print(f"Loading model from: {model_path}")

        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully!")

        # Initialize face detector
        self.face_detector = FaceDetector()

        # Initialize logging
        if self.log_predictions:
            self._init_log_file()

        # Performance tracking
        self.frame_count = 0
        self.inference_count = 0
        self.start_time = None
        self.fps_history = []

        print(f"Process every N frames: {self.process_every_n_frames}")
        print(f"Logging enabled: {self.log_predictions}")
        print("="*80 + "\n")

    def _init_log_file(self):
        """
        Initialize prediction log file with headers
        """
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # Write headers
        with open(self.log_file, 'w') as f:
            f.write("timestamp,frame_number,prediction,confidence,face_x,face_y,face_w,face_h\n")

        print(f"Prediction log initialized: {self.log_file}")

    def log_prediction(self, prediction, confidence, face_coords):
        """
        Log prediction to CSV file

        Args:
            prediction: Predicted label ('happy' or 'sad')
            confidence: Prediction confidence (0-1)
            face_coords: Face bounding box (x, y, w, h)
        """
        if not self.log_predictions:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        x, y, w, h = face_coords

        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{self.frame_count},{prediction},{confidence:.4f},{x},{y},{w},{h}\n")

    def predict_sentiment(self, face_image):
        """
        Predict sentiment for a face image

        Args:
            face_image: Preprocessed face image (1, 48, 48, 1)

        Returns:
            Tuple (label, confidence) where label is 'happy' or 'sad'
        """
        # Make prediction
        prediction = self.model.predict(face_image, verbose=0)[0][0]

        # Convert to label and confidence
        if prediction >= config.INFERENCE_CONFIG['confidence_threshold']:
            label = 'happy'
            confidence = prediction
        else:
            label = 'sad'
            confidence = 1 - prediction

        return label, confidence

    def calculate_fps(self, current_time):
        """
        Calculate current FPS

        Args:
            current_time: Current timestamp

        Returns:
            FPS value
        """
        if self.start_time is None:
            self.start_time = current_time
            return 0.0

        elapsed_time = current_time - self.start_time

        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
            return fps

        return 0.0

    def run(self, camera_index=0):
        """
        Run real-time sentiment analysis

        Args:
            camera_index: Camera device index (default: 0 for built-in webcam)
        """
        print("\n" + "="*80)
        print("STARTING REAL-TIME SENTIMENT ANALYSIS")
        print("="*80)
        print("Instructions:")
        print("  - Position your face in front of the camera")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("="*80 + "\n")

        # Open webcam
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}")

        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {width}x{height}")

        # Initialize timing
        self.start_time = time.time()
        last_inference_time = 0

        try:
            while True:
                # Read frame
                ret, frame = cap.read()

                if not ret:
                    print("Warning: Failed to read frame")
                    continue

                self.frame_count += 1
                current_time = time.time()

                # Process frame (every Nth frame)
                if self.frame_count % self.process_every_n_frames == 0:
                    self.inference_count += 1
                    inference_start = time.time()

                    # Detect faces
                    faces = self.face_detector.detect_faces(frame)

                    # Process each detected face
                    for face_coords in faces:
                        # Extract and preprocess face
                        face_image = self.face_detector.extract_face(frame, face_coords)

                        # Predict sentiment
                        label, confidence = self.predict_sentiment(face_image)

                        # Draw bounding box and label
                        self.face_detector.draw_face_box(
                            frame, face_coords, label, confidence
                        )

                        # Log prediction
                        self.log_prediction(label, confidence, face_coords)

                    # Calculate inference time
                    last_inference_time = (time.time() - inference_start) * 1000  # ms

                # Calculate and display FPS
                fps = self.calculate_fps(current_time)

                # Display info overlay
                self._draw_info_overlay(frame, fps, last_inference_time)

                # Show frame
                cv2.imshow('Real-time Sentiment Analysis', frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{config.VIZ_DIR}/capture_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved: {filename}")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()

            # Print summary
            self._print_summary()

    def _draw_info_overlay(self, frame, fps, inference_time):
        """
        Draw information overlay on frame

        Args:
            frame: Video frame
            fps: Current FPS
            inference_time: Last inference time (ms)
        """
        # Info text
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Frames: {self.frame_count}",
            f"Inferences: {self.inference_count}",
            f"Inference Time: {inference_time:.1f}ms"
        ]

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text
        y_offset = 30
        for line in info_lines:
            cv2.putText(frame, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20

        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _print_summary(self):
        """
        Print session summary
        """
        total_time = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        avg_inference_fps = self.inference_count / total_time if total_time > 0 else 0

        print("\n" + "="*80)
        print("SESSION SUMMARY")
        print("="*80)
        print(f"Total Duration: {total_time:.2f} seconds")
        print(f"Total Frames: {self.frame_count}")
        print(f"Total Inferences: {self.inference_count}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average Inference FPS: {avg_inference_fps:.2f}")

        if self.log_predictions:
            print(f"Predictions logged to: {self.log_file}")

        print("="*80 + "\n")


def run_realtime_analysis(model_path=None, camera_index=0):
    """
    Convenience function to run real-time analysis

    Args:
        model_path: Path to trained model (default: models/sentiment_model.h5)
        camera_index: Camera device index
    """
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, 'sentiment_model.h5')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

    # Create predictor
    predictor = RealtimeSentimentPredictor(model_path)

    # Run
    predictor.run(camera_index=camera_index)


if __name__ == "__main__":
    """
    Run real-time sentiment analysis
    """
    import argparse

    parser = argparse.ArgumentParser(description='Real-time Facial Sentiment Analysis')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (default: models/sentiment_model.h5)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')

    args = parser.parse_args()

    try:
        run_realtime_analysis(model_path=args.model, camera_index=args.camera)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train the model first by running:")
        print("  python train.py")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
