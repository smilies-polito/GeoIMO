import cv2
import numpy as np
from typing import Tuple

# BGR colors
COLOR_STATIC = (255, 200, 0)
COLOR_MOVING = (0, 0, 255)
COLOR_GATED_STATIC = (180, 140, 0)
COLOR_GATED_MOVING = (0, 0, 180)
COLOR_EGO = (0, 255, 0)
COLOR_OBJ_VEL = (0, 255, 255)
COLOR_FOE_VEL = (255, 0, 255)
COLOR_FOE_VEL_GATED = (180, 0, 180)
COLOR_INFO = (255, 200, 0)


def _s(v: float, scale: float) -> int:
    return int(round(float(v) * scale))


def draw_velocity_arrow(img: np.ndarray, center: Tuple[int, int],
                        vx: float, vy: float, color: Tuple[int, int, int],
                        scale_px: float = 0.15, thickness: int = 1,
                        output_scale: float = 1.0) -> None:
    cx, cy = center
    dx = int(round(vx * scale_px * output_scale))
    dy = int(round(vy * scale_px * output_scale))
    if abs(dx) > 2 or abs(dy) > 2:
        cv2.arrowedLine(img, (cx, cy), (cx + dx, cy + dy), color, thickness, tipLength=0.3)


def draw_labeled_bbox(img: np.ndarray, box,
                      label_text: str,
                      color: Tuple[int, int, int],
                      output_scale: float = 1.0,
                      ui_scale: float = 1.0) -> None:
    x1 = _s(box["x"], output_scale)
    y1 = _s(box["y"], output_scale)
    x2 = _s(box["x"] + box["w"], output_scale)
    y2 = _s(box["y"] + box["h"], output_scale)

    th = max(1, int(round(ui_scale)))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.22 * ui_scale
    text_th = 1

    text_size = cv2.getTextSize(label_text, font, font_scale, text_th)[0]
    pad = max(1, int(round(1.5 * ui_scale)))

    y_top = y1 - text_size[1] - 2 * pad
    cv2.rectangle(img, (x1, y_top), (x1 + text_size[0] + 2 * pad, y1), color, -1)
    cv2.putText(img, label_text, (x1 + pad, y1 - pad),
                font, font_scale, (0, 0, 0), text_th, cv2.LINE_AA)
