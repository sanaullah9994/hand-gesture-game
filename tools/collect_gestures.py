"""
tools/collect_gestures.py
--------------------------
Record hand landmark data from the webcam and append it to
data/gesture_data.csv for model training.

Usage:
    python tools/collect_gestures.py --gesture peace --samples 200

Each frame with a detected hand writes one row of 63 landmark values
(21 landmarks × x/y/z) plus the gesture label to the CSV.

Press ESC to stop recording early.
"""

import argparse
import csv
import os
import sys

import cv2
import mediapipe as mp

# Resolve data directory relative to project root
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "gesture_data.csv")

mp_hands = mp.solutions.hands


def parse_args():
    parser = argparse.ArgumentParser(description="Collect gesture training data.")
    parser.add_argument("--gesture", type=str, required=True,
                        help="Label for the gesture to record (e.g. peace, fist, ok).")
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of samples to collect (default: 200).")
    return parser.parse_args()


def collect(gesture_name: str, target_samples: int):
    """Stream webcam, detect hand, and save landmark rows to CSV."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    collected = 0
    print(f"Recording gesture '{gesture_name}'. Target: {target_samples} samples.")
    print("Press ESC to stop early.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
    ) as hands, open(DATA_PATH, "a", newline="") as csvfile:

        writer = csv.writer(csvfile)

        while collected < target_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lm   = [(l.x, l.y, l.z) for l in results.multi_hand_landmarks[0].landmark]
                flat = [v for point in lm for v in point]   # 63 float values
                writer.writerow(flat + [gesture_name])
                collected += 1

                cv2.putText(frame,
                            f"Recording '{gesture_name}': {collected}/{target_samples}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Collect Gestures — ESC to stop", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {collected} samples saved to: {DATA_PATH}")


if __name__ == "__main__":
    args = parse_args()
    collect(args.gesture, args.samples)
