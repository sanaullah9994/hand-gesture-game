# Hand Gesture Snake Game

A snake game you control with your hand using your webcam. Built using MediaPipe for hand tracking and Pygame for the game.

Also includes a hand HUD visualizer that shows real-time overlay on your hand with rotation angle, openness %, and skeleton lines.

## What it does

- Move your index finger left/right/up/down in front of the camera to control the snake
- Snake wraps around edges (no wall death)
- Speed increases as your score goes up
- Small camera feed shows in the corner so you can see your hand while playing

The hand HUD mode (hand_hud.py) is a separate visualizer that just shows what MediaPipe detects on your hand — rotation, openness, gear overlays etc.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

Pick option 1 for the HUD or option 2 for the snake game.

## Requirements

- Python 3.9+
- Webcam
- See requirements.txt

## Files

```
main.py              - run this to start
snake_game.py        - the actual game
hand_hud.py          - hand tracking visualizer
src/
  hand_overlay.py    - drawing functions for the HUD
  utils.py           - smoother class to reduce jitter
  gesture_recognition.py - classify gestures using saved model
tools/
  collect_gestures.py  - record your own gesture data
  train_model.py       - train the SVM model on your data
data/
  gesture_data.csv   - training data
```

## Training your own gestures

```bash
# collect data (change gesture name each time)
python tools/collect_gestures.py --gesture fist --samples 200
python tools/collect_gestures.py --gesture peace --samples 200

# train
python tools/train_model.py

# run live recognition
python src/gesture_recognition.py
```

## Notes

- Make sure your hand is visible and well lit
- The deadzone in the middle prevents accidental direction changes
- Model uses SVM with RBF kernel trained on 63 landmark features per frame

