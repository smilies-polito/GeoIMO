import numpy as np


def _iou(a, b) -> float:
    ax1, ay1 = float(a["x"]), float(a["y"])
    ax2, ay2 = ax1 + float(a["w"]), ay1 + float(a["h"])
    bx1, by1 = float(b["x"]), float(b["y"])
    bx2, by2 = bx1 + float(b["w"]), by1 + float(b["h"])

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = float(a["w"]) * float(a["h"])
    area_b = float(b["w"]) * float(b["h"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes: np.ndarray, iou_thresh: float = 0.3) -> np.ndarray:
    if len(boxes) <= 1:
        return boxes
    order = np.argsort(-boxes["class_confidence"])
    boxes_sorted = boxes[order]
    keep = []
    suppressed = set()
    for i in range(len(boxes_sorted)):
        if i in suppressed:
            continue
        keep.append(i)
        for j in range(i + 1, len(boxes_sorted)):
            if j in suppressed:
                continue
            if _iou(boxes_sorted[i], boxes_sorted[j]) >= iou_thresh:
                suppressed.add(j)
    return boxes_sorted[keep]
