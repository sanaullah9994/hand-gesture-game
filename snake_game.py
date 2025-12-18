"""
snake_game.py
-------------
Hand Gesture Snake Game — controlled via webcam using MediaPipe.

Move your index finger left/right/up/down relative to the camera centre
to steer the snake. The game has wrap-around walls (no boundary death).
Speed increases gradually as your score grows.

Usage:
    python snake_game.py
"""

import math
import random
import sys

import cv2
import mediapipe as mp
import pygame

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DISPLAY_WIDTH  = 800
DISPLAY_HEIGHT = 600
SNAKE_BLOCK    = 20
BASE_SPEED     = 12       # frames per second at score 0
DEADZONE       = 60       # pixel radius around cam centre to ignore
CAM_FEED_SIZE  = (240, 180)  # (width, height) of embedded camera preview

# Colours
BLACK  = (0,   0,   0)
WHITE  = (255, 255, 255)
YELLOW = (255, 255, 102)
RED    = (213, 50,  80)
GREEN  = (0,   255, 0)
BLUE   = (50,  153, 213)
CYAN   = (0,   200, 220)
ORANGE = (255, 140, 0)

# ──────────────────────────────────────────────
# Pygame initialisation
# ──────────────────────────────────────────────
pygame.init()

display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption("Hand Gesture Snake Game 🐍")
clock = pygame.time.Clock()

try:
    font_ui    = pygame.font.SysFont("bahnschrift", 25)
    font_score = pygame.font.SysFont("comicsansms", 35)
except Exception:
    font_ui    = pygame.font.SysFont(None, 24)
    font_score = pygame.font.SysFont(None, 32)

# ──────────────────────────────────────────────
# MediaPipe Hands
# ──────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils


# ──────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────

def draw_snake(snake_list: list):
    """Render the snake body segments."""
    for x, y in snake_list:
        pygame.draw.rect(display, GREEN, [x, y, SNAKE_BLOCK, SNAKE_BLOCK])


def draw_food(fx: float, fy: float):
    """Render the food item."""
    pygame.draw.rect(display, RED, [fx, fy, SNAKE_BLOCK, SNAKE_BLOCK])


def draw_message(msg: str, color: tuple):
    """Render a centred message on screen."""
    surface = font_ui.render(msg, True, color)
    rect = surface.get_rect(center=(DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2))
    display.blit(surface, rect)


def draw_gear_overlay(head_pos: tuple, tick: int):
    """
    Draw an animated gear effect at the snake's head.

    Args:
        head_pos: (x, y) pixel position of the head's top-left corner.
        tick:     Frame counter used to rotate the gear spokes.
    """
    cx = head_pos[0] + SNAKE_BLOCK // 2
    cy = head_pos[1] + SNAKE_BLOCK // 2
    pygame.draw.circle(display, BLUE,   (cx, cy), 28, 2)
    pygame.draw.circle(display, ORANGE, (cx, cy), 18, 2)

    angle = (tick * 0.15) % (2 * math.pi)
    for i in range(8):
        theta = angle + (2 * math.pi * i / 8)
        x1 = cx + int(12 * math.cos(theta))
        y1 = cy + int(12 * math.sin(theta))
        x2 = cx + int(26 * math.cos(theta))
        y2 = cy + int(26 * math.sin(theta))
        pygame.draw.line(display, CYAN, (x1, y1), (x2, y2), 2)


def embed_camera_feed(frame_bgr, pos: tuple = None):
    """
    Blit a resized camera frame onto the Pygame display (top-right corner).

    Args:
        frame_bgr: BGR frame from OpenCV.
        pos:       (x, y) destination; defaults to top-right corner.
    """
    if pos is None:
        pos = (DISPLAY_WIDTH - CAM_FEED_SIZE[0] - 10, 10)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, CAM_FEED_SIZE)
    cam_surface = pygame.image.frombuffer(frame_rgb.tobytes(), CAM_FEED_SIZE, "RGB")
    display.blit(cam_surface, pos)
    pygame.draw.rect(
        display, CYAN,
        [pos[0], pos[1], CAM_FEED_SIZE[0], CAM_FEED_SIZE[1]], 2,
    )


# ──────────────────────────────────────────────
# Gesture detection
# ──────────────────────────────────────────────

def get_gesture_direction(frame_bgr, prev_direction: str, deadzone: int = DEADZONE):
    """
    Infer movement direction from the index-finger tip position.

    Compares the tip pixel against the frame centre; the deadzone
    prevents accidental direction changes when the finger is centred.

    Args:
        frame_bgr:      BGR webcam frame (annotated in place).
        prev_direction: The direction from the last frame.
        deadzone:       Pixel radius around centre to treat as neutral.

    Returns:
        (direction, annotated_frame) — direction is one of
        "LEFT", "RIGHT", "UP", "DOWN", or unchanged prev_direction.
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    direction = prev_direction

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_bgr, hand_lm, mp_hands.HAND_CONNECTIONS)
            tip = hand_lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            ix, iy = int(tip.x * w), int(tip.y * h)
            cx, cy = w // 2, h // 2

            if ix < cx - deadzone and prev_direction != "RIGHT":
                direction = "LEFT"
            elif ix > cx + deadzone and prev_direction != "LEFT":
                direction = "RIGHT"

            if iy < cy - deadzone and prev_direction != "DOWN":
                direction = "UP"
            elif iy > cy + deadzone and prev_direction != "UP":
                direction = "DOWN"

            cv2.circle(frame_bgr, (ix, iy), 10, (0, 255, 0), cv2.FILLED)

    return direction, frame_bgr


# ──────────────────────────────────────────────
# Game state helpers
# ──────────────────────────────────────────────

def random_food_pos() -> tuple:
    """Return a random grid-aligned food position."""
    x = (random.randrange(0, DISPLAY_WIDTH  - SNAKE_BLOCK) // SNAKE_BLOCK) * SNAKE_BLOCK
    y = (random.randrange(0, DISPLAY_HEIGHT - SNAKE_BLOCK) // SNAKE_BLOCK) * SNAKE_BLOCK
    return float(x), float(y)


def apply_direction(x: float, y: float, direction: str):
    """
    Calculate next head position based on direction.

    Returns:
        (new_x, new_y, dx, dy)
    """
    dx, dy = 0, 0
    if direction == "LEFT":
        dx = -SNAKE_BLOCK
    elif direction == "RIGHT":
        dx = SNAKE_BLOCK
    elif direction == "UP":
        dy = -SNAKE_BLOCK
    elif direction == "DOWN":
        dy = SNAKE_BLOCK
    return x + dx, y + dy, dx, dy


def wrap_position(x: float, y: float) -> tuple:
    """Wrap snake head around screen edges (no-wall mode)."""
    x = x % DISPLAY_WIDTH
    y = y % DISPLAY_HEIGHT
    return x, y


# ──────────────────────────────────────────────
# Main game loop
# ──────────────────────────────────────────────

def run_game():
    """Initialise and run the Snake game loop."""
    # Snake state
    x, y = DISPLAY_WIDTH / 2, DISPLAY_HEIGHT / 2
    snake_body: list = []
    snake_length: int = 1
    direction: str = "STOP"

    food_x, food_y = random_food_pos()
    tick: int = 0
    game_over: bool = False

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        pygame.quit()
        sys.exit(1)

    print("Game started! Move your index finger to steer the snake.")
    print("Close the window or press the × button to quit.")

    while not game_over:
        # ── Event handling ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        # ── Gesture input ──
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            direction, frame = get_gesture_direction(frame, direction)

        # ── Move snake ──
        x, y, _, _ = apply_direction(x, y, direction)
        x, y = wrap_position(x, y)

        # ── Build snake body list ──
        head = [x, y]
        snake_body.append(head)
        if len(snake_body) > snake_length:
            del snake_body[0]

        # ── Self-collision check ──
        if head in snake_body[:-1]:
            game_over = True
            break

        # ── Render ──
        display.fill(BLACK)
        draw_food(food_x, food_y)
        draw_snake(snake_body)
        draw_gear_overlay((int(x), int(y)), tick)

        score = snake_length - 1
        score_surface = font_score.render(f"Score: {score}", True, YELLOW)
        display.blit(score_surface, (10, 10))

        if ret:
            embed_camera_feed(frame)

        pygame.display.update()

        # ── Food collision ──
        if abs(x - food_x) < SNAKE_BLOCK and abs(y - food_y) < SNAKE_BLOCK:
            food_x, food_y = random_food_pos()
            snake_length += 1

        # speed goes up every 5 points - gets pretty hard lol
        clock.tick(BASE_SPEED + (snake_length // 5))
        tick += 1

    # ── Game over screen ──
    display.fill(BLACK)
    draw_message(f"Game Over! Score: {snake_length - 1}  — Close window to exit.", RED)
    pygame.display.update()
    pygame.time.delay(3000)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    run_game()
