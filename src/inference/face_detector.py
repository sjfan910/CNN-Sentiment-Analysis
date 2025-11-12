"""
Face Detection Module for Real-time Inference
Uses Haar Cascade for efficient face detection
"""

import cv2
import numpy as np
import config


class FaceDetector:
    """
    Face detector using Haar Cascade classifier
    Optimized for real-time performance on MacBook
    """

    def __init__(self, cascade_path=None, scale_factor=None, min_neighbors=None, min_size=None):
        """
        Initialize face detector

        Args:
            cascade_path: Path to Haar Cascade XML file
            scale_factor: Image pyramid scale factor
            min_neighbors: Minimum number of neighbor rectangles
            min_size: Minimum face size (width, height)
        """
        # Use config defaults if not specified
        if scale_factor is None:
            scale_factor = config.INFERENCE_CONFIG['detection_scale_factor']
        if min_neighbors is None:
            min_neighbors = config.INFERENCE_CONFIG['detection_min_neighbors']
        if min_size is None:
            min_size = config.INFERENCE_CONFIG['detection_min_size']

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        # Load Haar Cascade
        if cascade_path is None:
            # Try to load from OpenCV data directory
            cascade_path = cv2.data.haarcascades + config.INFERENCE_CONFIG['face_cascade']

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise ValueError(f"Failed to load Haar Cascade from {cascade_path}")

        print("="*80)
        print("FACE DETECTOR INITIALIZED")
        print("="*80)
        print(f"Cascade: {cascade_path}")
        print(f"Scale Factor: {self.scale_factor}")
        print(f"Min Neighbors: {self.min_neighbors}")
        print(f"Min Size: {self.min_size}")
        print("="*80 + "\n")

    def detect_faces(self, frame):
        """
        Detect faces in a frame

        Args:
            frame: Input image (BGR format)

        Returns:
            List of face bounding boxes [(x, y, w, h), ...]
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        return faces

    def extract_face(self, frame, face_coords):
        """
        Extract and preprocess a face region for model input

        Args:
            frame: Input image (BGR format)
            face_coords: Face bounding box (x, y, w, h)

        Returns:
            Preprocessed face image ready for model (48x48x1, normalized)
        """
        x, y, w, h = face_coords

        # Extract face region
        face_region = frame[y:y+h, x:x+w]

        # Convert to grayscale
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Resize to model input size
        face_resized = cv2.resize(face_gray, (config.IMG_WIDTH, config.IMG_HEIGHT))

        # Normalize to [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0

        # Add channel dimension and batch dimension
        face_preprocessed = np.expand_dims(face_normalized, axis=-1)  # (48, 48, 1)
        face_preprocessed = np.expand_dims(face_preprocessed, axis=0)  # (1, 48, 48, 1)

        return face_preprocessed

    def draw_face_box(self, frame, face_coords, label=None, confidence=None, color=None):
        """
        Draw bounding box and label on face

        Args:
            frame: Input image (BGR format)
            face_coords: Face bounding box (x, y, w, h)
            label: Predicted label ('happy' or 'sad')
            confidence: Prediction confidence (0-1)
            color: Box color (BGR tuple)

        Returns:
            Frame with drawn box and label
        """
        x, y, w, h = face_coords

        # Default color
        if color is None:
            if label == 'happy':
                color = config.INFERENCE_CONFIG['color_happy']
            elif label == 'sad':
                color = config.INFERENCE_CONFIG['color_sad']
            else:
                color = config.INFERENCE_CONFIG['color_box']

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Draw label and confidence
        if label is not None:
            # Prepare text
            if confidence is not None and config.INFERENCE_CONFIG['show_confidence']:
                text = f"{label.upper()}: {confidence:.2%}"
            else:
                text = label.upper()

            # Calculate text position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Draw background rectangle for text
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)

            # Draw text
            cv2.putText(frame, text, (x, y - 5), font, font_scale, (255, 255, 255), thickness)

        return frame


if __name__ == "__main__":
    """
    Test face detector
    """
    print("Testing Face Detector...")

    # Initialize detector
    detector = FaceDetector()

    # Test with webcam
    print("\nOpening webcam for testing...")
    print("Press 'q' to quit")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        # Detect faces
        faces = detector.detect_faces(frame)

        # Draw boxes around detected faces
        for face_coords in faces:
            detector.draw_face_box(frame, face_coords, label="Test", confidence=0.95)

        # Display info
        cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Face Detection Test', frame)

        frame_count += 1

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"\nProcessed {frame_count} frames")
    print("Test completed!")

    cap.release()
    cv2.destroyAllWindows()
