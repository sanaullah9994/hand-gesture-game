"""
gesture_recognition.py
-----------------------
Live hand gesture recognition using a pre-trained SVM model.

Loads `models/gesture_model.pkl`, opens the webcam, detects hand
landmarks via MediaPipe, and predicts the gesture class in real time.

Usage:
    python src/gesture_recognition.py
"""

import os
import sys
import cv2
import numpy as np
import joblib
import mediapipe as mp

# Resolve paths relative to project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "models", "gesture_model.pkl")


def load_model(path: str):
    """
    Load the SVM gesture classifier from disk.

    Args:
        path: Absolute path to the .pkl model file.

    Returns:
        Trained sklearn classifier.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'.\n"
            "Run tools/train_model.py first to generate it."
        )
    return joblib.load(path)


def run():
    """Open webcam and predict hand gesture in real time."""
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    print("Gesture recognition running. Press ESC to quit.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture_label = "No hand detected"

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                # Flatten 21 landmarks × 3 coords → 63 features
                lm = [(l.x, l.y, l.z) for l in hand.landmark]
                features = np.array(lm).flatten().reshape(1, -1)
                gesture_label = str(model.predict(features)[0])

            # Display prediction
            cv2.putText(
                frame, f"Gesture: {gesture_label}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, "Press ESC to quit",
                (10, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA,
            )

            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
