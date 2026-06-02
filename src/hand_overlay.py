import cv2
import numpy as np


def landmarks_to_pixel(landmarks, width, height):
    return [(int(x * width), int(y * height)) for (x, y, z) in landmarks]


def draw_skeleton(image, pix, t=0):
    # connect joints with lines
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
    ]
    for (i, j) in connections:
        cv2.line(image, pix[i], pix[j], (0, 255, 0), 2, cv2.LINE_AA)
    for (x, y) in pix:
        cv2.circle(image, (x, y), 4, (255, 0, 0), -1)


def draw_palm_radial_ui(image, palm, rot, t=0, width=1.0):
    r = int(40 * width)
    cv2.circle(image, palm, r, (0, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(image, f"Rot:{int(rot)}", (palm[0] + r + 10, palm[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2, cv2.LINE_AA)


def draw_rotation_text(image, palm, rot):
    cv2.putText(image, f"{int(rot)} deg", (palm[0] - 40, palm[1] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_cube_and_grid(image, anchor, t=0):
    x, y = anchor
    cv2.rectangle(image, (x, y), (x + 60, y + 60), (255, 0, 255), 2)
    cv2.line(image, (x, y + 30), (x + 60, y + 30), (255, 0, 255), 1)
    cv2.line(image, (x + 30, y), (x + 30, y + 60), (255, 0, 255), 1)


def draw_fingertip_gears(image, pix, rot, t=0):
    for ti in [4, 8, 12, 16, 20]:
        cv2.circle(image, pix[ti], 12, (0, 255, 255), 2, cv2.LINE_AA)


def draw_palm_data_text(image, anchor, openness):
    cv2.putText(image, f"Open: {int(openness)}%", (anchor[0], anchor[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
