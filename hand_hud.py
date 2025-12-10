"""
hand_hud.py
-----------
Real-time Hand HUD Visualizer.

Opens the webcam and overlays an animated heads-up display on top of
the detected hand: skeleton lines, radial UI ring, rotation angle,
fingertip gear circles, cube/grid near the wrist, and hand-openness %.

Usage:
    python hand_hud.py
"""

import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

# Local imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from hand_overlay import (
    draw_cube_and_grid,
    draw_fingertip_gears,
    draw_palm_data_text,
    draw_palm_radial_ui,
    draw_rotation_text,
    draw_skeleton,
    landmarks_to_pixel,
)
from utils import Smoother

# Suppress TensorFlow / MediaPipe log noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

mp_hands = mp.solutions.hands

# ──────────────────────────────────────────────
# Smoothers — reduce landmark jitter
# ──────────────────────────────────────────────
palm_smoother     = Smoother(alpha=0.6)
rot_smoother      = Smoother(alpha=0.6)
openness_smoother = Smoother(alpha=0.35)


def compute_palm_rotation(landmarks: list) -> float:
    """
    Estimate 2-D palm rotation in degrees [0, 360].

    Uses the vector from the wrist (landmark 0) to the middle-finger
    MCP joint (landmark 9) to define the palm's pointing direction.

    Args:
        landmarks: List of (x, y, z) normalised landmark tuples.

    Returns:
        Rotation angle in degrees.
    """
    wrist  = np.array(landmarks[0][:2])
    middle = np.array(landmarks[9][:2])
    v = middle - wrist
    angle = float(np.degrees(np.arctan2(v[1], v[0])) % 360)
    return angle


def compute_openness(lm: list, pix: list, palm: tuple, frame_w: int, frame_h: int) -> float:
    """
    Estimate hand openness as a percentage [0, 100].

    Measures mean Euclidean distance of all five fingertips from the
    palm centre, then normalises against empirical closed/open reference
    distances derived from frame dimensions.

    Args:
        lm:      Normalised landmark list.
        pix:     Pixel landmark list.
        palm:    (cx, cy) palm centre pixel.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        Openness percentage [0.0, 100.0].
    """
    tip_indices = [4, 8, 12, 16, 20]
    dists = [
        np.hypot(int(lm[ti][0] * frame_w) - palm[0],
                 int(lm[ti][1] * frame_h) - palm[1])
        for ti in tip_indices
    ]
    mean_dist  = float(np.mean(dists))
    closed_ref = max(12.0, min(40.0, min(frame_w, frame_h) * 0.04))
    open_ref   = max(60.0, min(frame_w, frame_h) * 0.55)
    raw = (mean_dist - closed_ref) / (open_ref - closed_ref) * 100.0
    return float(np.clip(raw, 0.0, 100.0))


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def run():
    """Open webcam and display the hand HUD in real time."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    prev_time = time.time()
    fps = 0.0

    print("Hand HUD running. Press ESC to quit.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # FPS calculation
                now = time.time()
                dt = max(now - prev_time, 1e-6)
                prev_time = now
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                overlay = frame.copy()

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    lm  = [(l.x, l.y, l.z) for l in hand.landmark]
                    pix = landmarks_to_pixel(lm, w, h)

                    # Palm centre = midpoint of wrist (0) and middle MCP (9)
                    palm = tuple(map(int, palm_smoother.update(
                        ((pix[0][0] + pix[9][0]) // 2,
                         (pix[0][1] + pix[9][1]) // 2)
                    )))

                    rot = float(rot_smoother.update(compute_palm_rotation(lm)))
                    openness = float(openness_smoother.update(
                        compute_openness(lm, pix, palm, w, h)
                    ))

                    # Wrist anchor for cube (offset left-below wrist)
                    wrist_anchor = (pix[0][0] - 80, pix[0][1] + 40)

                    # Draw all overlays
                    draw_skeleton(overlay, pix, t=now)
                    draw_palm_radial_ui(overlay, palm, rot, t=now,
                                        width=1.0 + openness / 160.0)
                    draw_rotation_text(overlay, palm, rot)
                    draw_cube_and_grid(overlay, wrist_anchor, t=now)
                    draw_fingertip_gears(overlay, pix, rot, t=now)
                    draw_palm_data_text(overlay, wrist_anchor, openness)

                # HUD text
                cv2.putText(overlay, f"FPS: {int(fps)}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (180, 180, 180), 2, cv2.LINE_AA)
                cv2.putText(overlay, "Press ESC to quit",
                            (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (200, 200, 200), 1, cv2.LINE_AA)

                cv2.imshow("Hand HUD", overlay)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
