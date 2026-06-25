#!/usr/bin/env bash
# GeoIMO — end-to-end quickstart wrapper
#
# Defaults process ~12 seconds of MVSEC outdoor_day2 (skipping the first 10 s)
# out of the box — short enough to finish quickly, long enough to be meaningful:
#
#   bash quickstart.sh
#
# Override any variable to switch dataset or paths, e.g.:
#
#   DATASET=prophesee \
#   EVENT_PATH=data/moorea_2019-02-18_000_td_61500000_121500000_td.dat \
#   BBOX_PATH=data/annotations/moorea_2019-02-18_000_td_61500000_121500000_motion_labels.npy \
#   DELTA_T=16666 \
#   bash quickstart.sh
#
# Recommended delta_t values:
#   MVSEC outdoor_day2   ->  91432 µs  (~10.9 fps, best-performing config; 4x GT interval)
#   Prophesee sequences  ->  16666 µs  (~60 fps, matches GT annotation rate)

set -euo pipefail

# ---------- container ----------
SIF="${SIF:-geoimo.sif}"

# ---------- dataset ----------
DATASET="${DATASET:-mvsec}"
EVENT_PATH="${EVENT_PATH:-data/outdoor_day2_events.dat}"
BBOX_PATH="${BBOX_PATH:-data/annotations/outdoor_day2_motion_labels.npy}"
DELTA_T="${DELTA_T:-91432}"
# Skip the first N microseconds of the sequence (default: 10 s).
SKIP_US="${SKIP_US:-10000000}"
# Number of frames to process (default: 128 ≈ 12 s at delta_t=91432 µs).
MAX_FRAMES="${MAX_FRAMES:-128}"

# ---------- output ----------
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-output/geoimo_out_${RUN_ID}.avi}"
LABEL_OUTPUT_PATH="${LABEL_OUTPUT_PATH:-output/geoimo_pred_${RUN_ID}.npy}"
LOG_FILE="${LOG_FILE:-output/geoimo_run_${RUN_ID}.log}"
mkdir -p "$(dirname "$OUTPUT_PATH")"

# ---------- algorithm flags ----------
ENABLE_YAW="${ENABLE_YAW:-TRUE}"
BBOX_METHOD="${BBOX_METHOD:-radial}"
BBOX_TRACKER_MODE="${BBOX_TRACKER_MODE:-polar}"
RESIDUAL_METHOD="${RESIDUAL_METHOD:-relative}"
THRESHOLD="${THRESHOLD:-0.6}"
FOE_SCORING="${FOE_SCORING:-global}"
FOE_N_CELLS="${FOE_N_CELLS:-1}"

YAW_ARG=""
if [ "$ENABLE_YAW" = "TRUE" ]; then
    YAW_ARG="--enable_yaw"
fi

echo "======================================================"
echo " GeoIMO — classification + evaluation"
echo "======================================================"
echo "  Dataset:   $DATASET"
echo "  Events:    $EVENT_PATH"
echo "  Boxes:     $BBOX_PATH"
echo "  delta_t:   $DELTA_T µs"
echo "  Output:    $OUTPUT_PATH"
echo "  Container: $SIF"
echo "  Log:       $LOG_FILE"
echo "======================================================"

apptainer exec "$SIF" python -m geoimo.classify \
    --dataset         "$DATASET" \
    --events          "$EVENT_PATH" \
    --boxes           "$BBOX_PATH" \
    -o                "$OUTPUT_PATH" \
    --delta_t         "$DELTA_T" \
    -n                "$MAX_FRAMES" \
    -s                "$SKIP_US" \
    --bbox_method     "$BBOX_METHOD" \
    --bbox_tracker_mode "$BBOX_TRACKER_MODE" \
    --residual_method "$RESIDUAL_METHOD" \
    --threshold       "$THRESHOLD" \
    --foe_scoring     "$FOE_SCORING" \
    --foe_n_cells     "$FOE_N_CELLS" \
    --label_output_npy "$LABEL_OUTPUT_PATH" \
    --log_file         "$LOG_FILE" \
    $YAW_ARG

echo ""
echo "Classification done. Running evaluation..."
echo ""

TIME_WINDOW=$((DELTA_T / 5))

apptainer exec "$SIF" python -m geoimo.evaluate \
    --gt              "$BBOX_PATH" \
    --pred            "$LABEL_OUTPUT_PATH" \
    --time-window     "$TIME_WINDOW" \
    --class-names     "moving,static" \
    --iou-threshold   0.8 \
    --log-file        "$LOG_FILE"

echo ""
echo "Done. Output video: $OUTPUT_PATH"
echo "Log written to:    $LOG_FILE"
