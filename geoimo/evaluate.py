"""Evaluate static/moving classification on matched bbox pairs.

Usage:
    python -m geoimo.evaluate --gt ground_truth.npy --pred predictions.npy \\
        --time-window 5000 --iou-threshold 0.80
"""

import argparse
from collections import defaultdict

import numpy as np


# GT has 4 classes; predictions have 2 (0=moving, 1=static).
# Map GT class_id → pred class_id.
GT_CLASS_REMAP = {
    0: 1,   # person_static  -> static
    1: 1,   # vehicle_static -> static
    2: 0,   # person_moving  -> moving
    3: 0,   # vehicle_moving -> moving
}

VEHICLE_CLASS_IDS = (1, 3)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Prophesee bbox classification on matched pairs."
    )
    p.add_argument("--gt", required=True, help="Path to ground-truth .npy file.")
    p.add_argument("--pred", required=True, help="Path to predictions .npy file.")
    p.add_argument(
        "--time-window",
        type=int,
        default=5000,
        help="Temporal tolerance in microseconds for matching timestamps "
             "(default: 5000 us = 5 ms).",
    )
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.80,
        help="Minimum IoU to accept a spatial match (default: 0.80).",
    )
    p.add_argument(
        "--no-remap",
        action="store_true",
        help="Disable GT class remapping (use when GT and pred already share the same classes).",
    )
    p.add_argument(
        "--class-names",
        type=str,
        default="moving,static",
        help="Comma-separated class names in the evaluation label space "
             "(default: 'moving,static').",
    )
    p.add_argument(
        "--exclude-persons",
        action="store_true",
        help="Exclude person_* GT classes from evaluation and evaluate vehicles only.",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Append evaluation output to this file in addition to stdout.",
    )
    return p.parse_args()


def _iou(box_a, box_b):
    """Compute IoU between two boxes, each with fields x, y, w, h."""
    ax1, ay1 = float(box_a["x"]), float(box_a["y"])
    ax2, ay2 = ax1 + float(box_a["w"]), ay1 + float(box_a["h"])
    bx1, by1 = float(box_b["x"]), float(box_b["y"])
    bx2, by2 = bx1 + float(box_b["w"]), by1 + float(box_b["h"])

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = float(box_a["w"]) * float(box_a["h"])
    area_b = float(box_b["w"]) * float(box_b["h"])
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def _group_by_time(gt, pred, time_window):
    """Assign each pred box to its nearest GT timestamp if within time_window.

    Returns:
        groups: list of (gt_slice, pred_slice) per GT timestamp
        unmatched_pred: predictions that could not be assigned to any GT timestamp
    """
    all_gt_ts = np.unique(gt["t"])
    if len(all_gt_ts) == 0:
        return [], pred

    pred_ts = pred["t"].astype(np.int64)
    gt_ts_arr = all_gt_ts.astype(np.int64)

    nearest_idx = np.searchsorted(gt_ts_arr, pred_ts, side="left")
    nearest_idx = np.clip(nearest_idx, 0, len(gt_ts_arr) - 1)

    left_idx = np.clip(nearest_idx - 1, 0, len(gt_ts_arr) - 1)
    dist_right = np.abs(pred_ts - gt_ts_arr[nearest_idx])
    dist_left = np.abs(pred_ts - gt_ts_arr[left_idx])

    use_left = dist_left < dist_right
    nearest_idx[use_left] = left_idx[use_left]

    nearest_dist = np.abs(pred_ts - gt_ts_arr[nearest_idx])
    within_window = nearest_dist <= time_window

    groups = []
    unmatched_mask = ~within_window

    for i, ts in enumerate(all_gt_ts):
        gt_slice = gt[gt["t"] == ts]
        pred_mask = within_window & (nearest_idx == i)
        pred_slice = pred[pred_mask]
        groups.append((gt_slice, pred_slice))

    unmatched_pred = pred[unmatched_mask]
    return groups, unmatched_pred


def _match_boxes(gt_boxes, pred_boxes, iou_threshold):
    """Match boxes greedily by descending IoU.

    Evaluation is classification on matched pairs only.
    Unmatched GT/pred boxes are counted as skipped, not FN/FP.

    Returns:
        per_class: dict[class_id] -> {"tp", "fp", "fn"}
        stats: matching statistics
        matched_label_pairs: list of (gt_cls, pred_cls)
    """
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    matched_label_pairs = []

    stats = {
        "skipped_gt": 0,
        "skipped_pred": 0,
        "matched_pairs": 0,
        "class_matches": 0,
        "class_mismatches": 0,
    }

    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return per_class, stats, matched_label_pairs

    if len(gt_boxes) == 0:
        stats["skipped_pred"] += len(pred_boxes)
        return per_class, stats, matched_label_pairs

    if len(pred_boxes) == 0:
        stats["skipped_gt"] += len(gt_boxes)
        return per_class, stats, matched_label_pairs

    pairs = []
    for pi, p in enumerate(pred_boxes):
        for gi, g in enumerate(gt_boxes):
            iou = _iou(g, p)
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))

    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_pred = set()

    for iou_val, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue

        matched_gt.add(gi)
        matched_pred.add(pi)

        gt_cls = int(gt_boxes[gi]["class_id"])
        pred_cls = int(pred_boxes[pi]["class_id"])

        matched_label_pairs.append((gt_cls, pred_cls))
        stats["matched_pairs"] += 1

        if gt_cls == pred_cls:
            per_class[gt_cls]["tp"] += 1
            stats["class_matches"] += 1
        else:
            per_class[gt_cls]["fn"] += 1
            per_class[pred_cls]["fp"] += 1
            stats["class_mismatches"] += 1

    stats["skipped_pred"] += (len(pred_boxes) - len(matched_pred))
    stats["skipped_gt"] += (len(gt_boxes) - len(matched_gt))

    return per_class, stats, matched_label_pairs


def evaluate(
    gt_path,
    pred_path,
    time_window,
    iou_threshold,
    class_names,
    no_remap=False,
    exclude_persons=False,
):
    gt = np.load(gt_path)
    pred = np.load(pred_path)

    gt.sort(order="t")
    pred.sort(order="t")

    if exclude_persons:
        gt = gt[np.isin(gt["class_id"], VEHICLE_CLASS_IDS)]

    if not no_remap:
        gt = gt.copy()
        for old_id, new_id in GT_CLASS_REMAP.items():
            gt["class_id"][gt["class_id"] == old_id] = new_id

    groups, unmatched_pred = _group_by_time(gt, pred, time_window)

    totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    skipped_gt = 0
    skipped_pred = 0
    matched_pairs = 0
    class_matches = 0
    class_mismatches = 0
    matched_label_pairs_all = []

    for gt_boxes, pred_boxes in groups:
        per_class, stats, matched_label_pairs = _match_boxes(
            gt_boxes, pred_boxes, iou_threshold
        )

        for cls, counts in per_class.items():
            totals[cls]["tp"] += counts["tp"]
            totals[cls]["fp"] += counts["fp"]
            totals[cls]["fn"] += counts["fn"]

        skipped_gt += stats["skipped_gt"]
        skipped_pred += stats["skipped_pred"]
        matched_pairs += stats["matched_pairs"]
        class_matches += stats["class_matches"]
        class_mismatches += stats["class_mismatches"]
        matched_label_pairs_all.extend(matched_label_pairs)

    skipped_pred += len(unmatched_pred)

    name_map = {}
    if class_names:
        for i, name in enumerate(class_names.split(",")):
            name_map[i] = name.strip()

    all_classes = sorted(set(totals.keys()) | set(int(c) for c in np.unique(gt["class_id"])))

    print(f"\n{'=' * 70}")
    print(f"Evaluation  |  IoU threshold: {iou_threshold}  |  Time window: {time_window} µs")
    print(f"Mode: {'vehicles only (persons excluded)' if exclude_persons else 'all classes (persons + vehicles)'}")
    print(f"GT boxes: {len(gt)}  |  Pred boxes: {len(pred)}")
    print(f"{'=' * 70}")
    print(f"{'Class':<25s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s}")
    print(f"{'-' * 70}")

    overall_tp, overall_fp, overall_fn = 0, 0, 0
    for cls in all_classes:
        tp = totals[cls]["tp"]
        fp = totals[cls]["fp"]
        fn = totals[cls]["fn"]

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        label = name_map.get(cls, str(cls))

        print(f"{label:<25s} {tp:>6d} {fp:>6d} {fn:>6d} {precision:>8.4f} {recall:>8.4f} {f1:>8.4f}")

    print(f"{'-' * 70}")

    precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"{'OVERALL':<25s} {overall_tp:>6d} {overall_fp:>6d} {overall_fn:>6d} "
          f"{precision:>8.4f} {recall:>8.4f} {f1:>8.4f}")

    class_acc = class_matches / matched_pairs if matched_pairs > 0 else 0.0

    print(f"\nMatched pairs (time+geometry): {matched_pairs}")
    print(f"Class agreement on matched pairs: {class_matches}/{matched_pairs} ({class_acc:.4f})")
    print(f"Class mismatches on matched pairs: {class_mismatches}")

    print("\nMatched-pairs filtering:")
    print(f"  Considered common GT boxes:   {matched_pairs}")
    print(f"  Considered common pred boxes: {matched_pairs}")
    print(f"  Skipped non-common GT boxes:  {skipped_gt}")
    print(f"  Skipped non-common pred boxes:{skipped_pred}")

    # Optional: binary confusion matrix for matched pairs only
    if set(name_map.keys()) >= {0, 1} or len(all_classes) <= 2:
        moving_id = 0
        static_id = 1

        tp_moving = sum(1 for g, p in matched_label_pairs_all if g == moving_id and p == moving_id)
        fn_moving = sum(1 for g, p in matched_label_pairs_all if g == moving_id and p == static_id)
        fp_moving = sum(1 for g, p in matched_label_pairs_all if g == static_id and p == moving_id)
        tn_moving = sum(1 for g, p in matched_label_pairs_all if g == static_id and p == static_id)

        total_matched = len(matched_label_pairs_all)
        matched_accuracy = (tp_moving + tn_moving) / total_matched if total_matched > 0 else 0.0

        print("\nConfusion matrix on matched pairs only:")
        print("                 Pred moving   Pred static")
        print(f"GT moving        {tp_moving:>11d} {fn_moving:>13d}")
        print(f"GT static        {fp_moving:>11d} {tn_moving:>13d}")
        print(f"\nMatched-pairs accuracy: {matched_accuracy:.4f}")

    print(f"{'=' * 70}\n")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


if __name__ == "__main__":
    import sys as _sys

    args = parse_args()

    def _run():
        evaluate(
            args.gt,
            args.pred,
            args.time_window,
            args.iou_threshold,
            args.class_names,
            no_remap=args.no_remap,
            exclude_persons=args.exclude_persons,
        )

    if args.log_file:
        with open(args.log_file, "a") as _lf:
            _orig = _sys.stdout
            _sys.stdout = _Tee(_orig, _lf)
            try:
                _run()
            finally:
                _sys.stdout = _orig
    else:
        _run()