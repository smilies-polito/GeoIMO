import time
import logging
import numpy as np
import cv2
from typing import Optional

from third_party.prophesee.io.psee_loader import PSEELoader
from third_party.prophesee.visualize import vis_utils as vis

from geoimo.logging_utils import log_every
from geoimo.intrinsics import compute_intrinsics, resolve_dataset_mode
from geoimo.contrast import FOE_SCORING_MODES
from geoimo.residuals import (
    RESIDUAL_METHODS, RESIDUAL_DEFAULTS, residual_relative_dynamic,
)
from geoimo.geometry import nms_boxes
from geoimo.ego_motion import (
    optimize_ego_motion_4param, compute_ego_velocity,
    grid_search_bg_foe_and_s,
)
from geoimo.bbox_motion import estimate_bbox_total_velocity, estimate_bbox_foe_s
from geoimo.tracking import EgoParamKalman, BboxTrackerManager
from geoimo.visualization import (
    COLOR_STATIC, COLOR_MOVING, COLOR_GATED_STATIC, COLOR_GATED_MOVING,
    COLOR_EGO, COLOR_OBJ_VEL, COLOR_FOE_VEL, COLOR_FOE_VEL_GATED, COLOR_INFO,
    _s, draw_velocity_arrow, draw_labeled_bbox,
)


def process_video(event_file: str, box_file: str, output_file: str,
                  dataset: str = "auto",
                  delta_t: int = 50000,
                  skip: int = 0,
                  max_frames: int = -1,
                  s_range_max: float = 5.0,
                  n_s_grid: int = 21,
                  ema_alpha: float = 0.4,
                  foe_ema_alpha: float = 0.2,
                  warm_start_s: float = 4.0,
                  residual_threshold: float = -1.0,
                  residual_method: str = "relative_dynamic",
                  relative_floor_base: float = 5.0,
                  relative_floor_boost: float = 7.0,
                  relative_floor_speed_transition: float = 12.0,
                  bbox_method: str = "both",
                  bbox_ema_alpha: float = 0.5,
                  bbox_match_dist: float = 40.0,
                  bbox_pad: int = 12,
                  warm_start_foe: float = 60.0,
                  foe_max_jump: float = 80.0,
                  s_min_for_foe: float = 0.3,
                  foe_turn_yaw_enter: float = 0.08,
                  foe_turn_yaw_exit: float = 0.05,
                  foe_turn_yaw_ref: float = 0.20,
                  foe_turn_bound_scale: float = 1.0,
                  foe_stop_s_thresh: float = 0.12,
                  foe_stop_hold_frames: int = 4,
                  foe_stop_revert_gain: float = 4.0,
                  foe_yaw_coupling: float = 1.0,
                  enable_yaw: bool = False,
                  omega_y_max: float = 0.5,
                  yaw_ema_alpha: float = 0.3,
                  ego_kalman_q_s: float = 0.05,
                  ego_kalman_q_foe: float = 4.0,
                  ego_kalman_q_oy: float = 0.02,
                  ego_kalman_base_r_s: float = 1.5,
                  ego_kalman_base_r_foe: float = 500.0,
                  ego_kalman_base_r_oy: float = 2.0,
                  ego_kalman_gate: float = 3.0,
                  ego_kalman_q_foe_y: Optional[float] = None,
                  ego_kalman_base_r_foe_y: Optional[float] = None,
                  foe_init_x: float = -1.0,
                  foe_init_y: float = -1.0,
                  yaw_reg_lambda: float = 10.0,
                  foe_reg_lambda_x: float = -1.0,
                  foe_reg_lambda_y: float = -1.0,
                  foe_y_reg_scale: float = 6.0,
                  s_min: float = 0.0,
                  bbox_reinit_mode: str = "direction_only",
                  bbox_gated_reset_n: int = 3,
                  kalman_q: float = 100.0,
                  kalman_base_r: float = 400.0,
                  kalman_gate: float = 4.0,
                  kalman_max_misses: int = 5,
                  bbox_tracker_mode: str = "cartesian",
                  polar_q_theta: float = 0.02,
                  polar_q_speed: float = 120.0,
                  polar_base_r_theta: float = 0.12,
                  polar_base_r_speed: float = 400.0,
                  polar_gate_theta: float = 3.0,
                  polar_gate_speed: float = 4.0,
                  polar_min_speed_theta: float = 8.0,
                  foe_scoring: str = "global",
                  foe_n_cells: int = 4,
                  max_bg_events: int = 200_000,
                  max_bbox_events: int = 30_000,
                  label_output_npy: str = "",
                  logger: Optional[logging.Logger] = None,
                  log_every_n_frames: int = 1) -> None:

    logger = logger or logging.getLogger("geoimo")
    foe_n_cells = max(1, int(foe_n_cells))
    if bbox_tracker_mode not in ("cartesian", "polar"):
        raise ValueError("bbox_tracker_mode must be 'cartesian' or 'polar'.")

    logger.info("=" * 70)
    if enable_yaw:
        logger.info("GeoIMO Classification (4-param bg: FOE + s + yaw)")
        logger.info("  Background: coarse 3D grid (foe_x, foe_y, s) → L-BFGS-B 4D refinement (+ω_y)")
    else:
        logger.info("GeoIMO Classification (3-param bg: FOE + s)")
        logger.info("  Background: coarse-to-fine 3D search over (foe_x, foe_y, s)")
    logger.info("  Bbox: direct total velocity + local FOE+s comparison")
    logger.info(f"  FOE scoring mode: {foe_scoring}  (n_cells={foe_n_cells})")
    logger.info("=" * 70)

    event_loader = PSEELoader(event_file)
    box_loader = PSEELoader(box_file)

    height, width = event_loader.get_size()

    mode = resolve_dataset_mode(dataset, width, height)
    intr = compute_intrinsics(mode, width, height)

    output_scale = 3.0 if mode == "mvsec" else 1.0

    cx, cy = width / 2.0, height / 2.0

    if mode == "prophesee":
        default_foe_x, default_foe_y = (640.0, 280.0)
    elif mode == "mvsec":
        default_foe_x, default_foe_y = (173.0, 110.0)
    else:
        default_foe_x, default_foe_y = (cx, cy)

    if foe_init_x == -1:
        foe_init_x = default_foe_x
    if foe_init_y == -1:
        foe_init_y = default_foe_y
    foe_x, foe_y = foe_init_x, foe_init_y

    if ego_kalman_q_foe_y is None:
        ego_kalman_q_foe_y = ego_kalman_q_foe
    if ego_kalman_base_r_foe_y is None:
        ego_kalman_base_r_foe_y = ego_kalman_base_r_foe

    logger.info(f"Video: {width}x{height}")
    logger.info(f"Dataset mode: {mode}  (reported: {intr.dataset_name})")
    logger.info(f"Camera intrinsics: fx={intr.fx:.1f}, fy={intr.fy:.1f}, f_avg={intr.f_avg:.1f} px")
    logger.info(f"FOE initial: ({foe_x:.0f}, {foe_y:.0f})  neutral=({foe_init_x:.0f},{foe_init_y:.0f})")
    logger.info(f"Output scale: {output_scale}x (nearest)")
    logger.info(f"Duration: {event_loader.duration_s:.2f}s")

    if max_frames == -1:
        max_frames = int(np.ceil(event_loader.duration_s * 1e6 / delta_t))
    logger.info(f"Max frames to process: {max_frames} (delta_t={delta_t}µs)")

    residual_fn = RESIDUAL_METHODS[residual_method]
    if residual_method == "relative_dynamic":
        def _relative_dynamic_residual(vx_total, vy_total, ego_vx, ego_vy, f_avg):
            return residual_relative_dynamic(
                vx_total, vy_total, ego_vx, ego_vy, f_avg,
                floor_base=relative_floor_base,
                floor_boost=relative_floor_boost,
                speed_transition=relative_floor_speed_transition,
            )
        residual_fn = _relative_dynamic_residual

    if residual_threshold < 0:
        residual_threshold = RESIDUAL_DEFAULTS[residual_method]
    logger.info(f"Residual method: {residual_method}  |  Threshold: {residual_threshold:.3f}")
    if residual_method == "relative_dynamic":
        logger.info(
            f"Relative dynamic floor: base={relative_floor_base:.2f}  "
            f"boost={relative_floor_boost:.2f}  "
            f"transition={relative_floor_speed_transition:.2f}px/s"
        )

    logger.info(f"Bbox velocity method: {bbox_method}")
    logger.info(f"Bbox tracker mode: {bbox_tracker_mode}")
    if bbox_tracker_mode == "polar":
        logger.info(
            f"Polar tracker: q_theta={polar_q_theta:.4f}  q_speed={polar_q_speed:.1f}  "
            f"R_theta={polar_base_r_theta:.3f}  R_speed={polar_base_r_speed:.1f}  "
            f"gate_theta={polar_gate_theta:.1f}σ  gate_speed={polar_gate_speed:.1f}σ  "
            f"min_speed_theta={polar_min_speed_theta:.1f}px/s"
        )
    else:
        logger.info(
            f"Kalman tracker: Q={kalman_q:.0f}  base_R={kalman_base_r:.0f}  "
            f"gate={kalman_gate:.1f}σ  max_misses={kalman_max_misses}"
        )
    if enable_yaw:
        logger.info(
            f"Yaw estimation: ENABLED  (omega_y_max={omega_y_max:.2f} rad/s, "
            f"fine stage: L-BFGS-B)"
        )
    else:
        logger.info("Yaw estimation: DISABLED (3-param fine grid search)")

    res_scale = np.sqrt((width * height) / (346.0 * 260.0))
    scaled_vel_range = 80.0 * res_scale
    scaled_coarse_step = 20.0 * res_scale
    scaled_fine_step = 5.0 * res_scale
    scaled_foe_range = 200.0 * res_scale
    scaled_foe_coarse = 50.0 * res_scale
    scaled_foe_fine = 12.0 * res_scale

    bg_foe_coarse_x = np.maximum(15.0, float(width) / 12.0)
    bg_foe_coarse_y = np.maximum(15.0, float(height) / 12.0)
    bg_foe_fine_x = bg_foe_coarse_x / 4.0
    bg_foe_fine_y = bg_foe_coarse_y / 4.0

    warm_foe_x = warm_start_foe * res_scale
    warm_foe_y = warm_start_foe * res_scale * 0.3

    logger.info(
        f"Resolution scale: {res_scale:.2f}x (vel search: +/-{scaled_vel_range:.0f} px/s, "
        f"bbox FOE: +/-{scaled_foe_range:.0f} px)"
    )

    base_foe_x_min = -float(width) * 0.2
    base_foe_x_max = float(width) * 1.2
    base_foe_y_min = foe_init_y - float(height) * 0.15
    base_foe_y_max = foe_init_y + float(height) * 0.15
    foe_x_min = base_foe_x_min
    foe_x_max = base_foe_x_max
    foe_y_min = base_foe_y_min
    foe_y_max = base_foe_y_max

    if skip > 0:
        event_loader.seek_time(skip)
        box_loader.seek_time(skip)
        logger.info(f"Skipped to: {skip / 1e6:.2f}s")

    s_range_full = np.linspace(s_min, s_range_max, n_s_grid)

    out_w = int(round(width * output_scale))
    out_h = int(round(height * output_scale))
    if out_w % 2 == 1:
        out_w += 1
    if out_h % 2 == 1:
        out_h += 1

    ui_scale = max(1.0, min(out_w, out_h) / 260.0)
    logger.info(f"Output: {out_w}x{out_h}  (output_scale={output_scale}x, ui_scale={ui_scale:.2f}x)")

    res_ratio = (width * height) / (346.0 * 260.0)
    min_bbox_events = int(round(100 * res_ratio))
    min_bbox_area = int(round(150 * res_ratio))
    logger.info(
        f"Bbox filters: min_events={min_bbox_events}, min_area={min_bbox_area} "
        f"(res_ratio={res_ratio:.2f}x vs MVSEC)"
    )

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(1e6 / delta_t)
    writer = cv2.VideoWriter(output_file, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        logger.error(f"Failed to open VideoWriter for: {output_file}")
        raise RuntimeError("cv2.VideoWriter could not be opened (check codec/path/permissions).")

    dt_sec = delta_t * 1e-6
    dt_ref_sec = 0.066666

    kf_s = EgoParamKalman(
        val_init=0.0, P_val=25.0, P_rate=50.0,
        q_val=0.5, q_rate=ego_kalman_q_s,
        base_R=ego_kalman_base_r_s, dt_ref=dt_ref_sec,
        gate=ego_kalman_gate,
        val_min=s_min, val_max=s_range_max,
        mean_revert_theta=0.5, mean_revert_target=0.0,
    )
    kf_foe_x = EgoParamKalman(
        val_init=foe_init_x, P_val=float(width * width), P_rate=400.0,
        q_val=100.0, q_rate=ego_kalman_q_foe,
        base_R=ego_kalman_base_r_foe, dt_ref=dt_ref_sec,
        gate=ego_kalman_gate,
        val_min=foe_x_min, val_max=foe_x_max,
    )
    kf_foe_y = EgoParamKalman(
        val_init=foe_init_y, P_val=float(height * height), P_rate=100.0,
        q_val=10.0, q_rate=ego_kalman_q_foe_y,
        base_R=ego_kalman_base_r_foe_y, dt_ref=dt_ref_sec,
        gate=ego_kalman_gate,
        val_min=foe_y_min, val_max=foe_y_max,
        mean_revert_theta=1.0, mean_revert_target=foe_init_y,
    )
    kf_omega_y = EgoParamKalman(
        val_init=0.0, P_val=1.0, P_rate=0.5,
        q_val=0.05, q_rate=ego_kalman_q_oy,
        base_R=ego_kalman_base_r_oy, dt_ref=dt_ref_sec,
        gate=ego_kalman_gate,
        val_min=-omega_y_max, val_max=omega_y_max,
        mean_revert_theta=2.0, mean_revert_target=0.0,
    )
    ego_sharpness_ema = 0.0
    foe_diverge_count = 0
    FOE_DIVERGE_MAX = 3

    _foe_reg_auto = yaw_reg_lambda / max(1.0, intr.f_avg * intr.f_avg)
    if foe_reg_lambda_x < 0:
        foe_reg_lambda_x = _foe_reg_auto
    if foe_reg_lambda_y < 0:
        foe_reg_lambda_y = _foe_reg_auto * max(1.0, foe_y_reg_scale)

    logger.info(f"delta_t={delta_t}µs  →  dt_sec={dt_sec:.6f}s")
    logger.info(
        f"Event caps: max_bg_events={max_bg_events}  max_bbox_events={max_bbox_events}"
    )

    prev_foe = False
    turn_state = False
    stop_state = False
    stop_counter = 0

    n_static_trans = 0
    n_moving_trans = 0
    n_static_radial = 0
    n_moving_radial = 0
    n_gated_radial = 0
    frame_count = 0

    trk_mgr = BboxTrackerManager(
        match_dist=bbox_match_dist,
        max_misses=kalman_max_misses,
        P_init=2500.0,
        q=kalman_q,
        base_R=kalman_base_r,
        dt_ref=dt_ref_sec,
        sharpness_ref=1.0,
        gate=kalman_gate,
        tracker_mode=bbox_tracker_mode,
        polar_q_theta=polar_q_theta,
        polar_q_speed=polar_q_speed,
        polar_base_r_theta=polar_base_r_theta,
        polar_base_r_speed=polar_base_r_speed,
        polar_gate_theta=polar_gate_theta,
        polar_gate_speed=polar_gate_speed,
        polar_min_speed_theta=polar_min_speed_theta,
    )
    bbox_sharpness_ema = 0.0

    label_output_npy = (label_output_npy or "").strip()
    LABEL_OUT_DTYPE = np.dtype([
        ('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'),
        ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4'),
    ])
    label_out_rows: list = []

    prev_bbox_states = []

    t_start = time.time()

    for i in range(max_frames):
        if event_loader.done:
            logger.info("Event loader reports done. Stopping.")
            break

        if log_every_n_frames > 0:
            log_every(logger, i + 1, log_every_n_frames,
                      f"Frame {i + 1:04d}/{max_frames:04d}")

        events = event_loader.load_delta_t(delta_t)
        boxes_raw = box_loader.load_delta_t(delta_t)
        if events.size == 0:
            continue

        yaw_abs_prev = abs(kf_omega_y.value) if enable_yaw else 0.0
        prev_turn_state = turn_state
        if enable_yaw:
            if turn_state:
                turn_state = (yaw_abs_prev >= foe_turn_yaw_exit)
            else:
                turn_state = (yaw_abs_prev >= foe_turn_yaw_enter)
        else:
            turn_state = False

        turn_ratio = (min(2.0, yaw_abs_prev / max(foe_turn_yaw_ref, 1e-6))
                      if enable_yaw else 0.0)
        x_extra = float(width) * 0.5 * foe_turn_bound_scale * turn_ratio
        y_extra = float(height) * 0.15 * foe_turn_bound_scale * turn_ratio

        foe_x_min = base_foe_x_min - x_extra
        foe_x_max = base_foe_x_max + x_extra
        foe_y_min = base_foe_y_min - y_extra
        foe_y_max = base_foe_y_max + y_extra

        kf_foe_x.val_min = foe_x_min
        kf_foe_x.val_max = foe_x_max
        kf_foe_y.val_min = foe_y_min
        kf_foe_y.val_max = foe_y_max

        if prev_turn_state != turn_state:
            logger.info(
                f"Frame {i + 1}: turn_state={'ON' if turn_state else 'OFF'}  "
                f"|yaw|={yaw_abs_prev:.3f} rad/s"
            )

        boxes = nms_boxes(boxes_raw, iou_thresh=0.3)
        t_ref = (events["t"].min() + events["t"].max()) / 2 * 1e-6

        img_small = vis.make_binary_histo(events, width=width, height=height)
        img = cv2.resize(img_small, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        bg_mask = np.ones(len(events), dtype=bool)
        for box in boxes:
            bx1, by1 = int(box["x"]), int(box["y"])
            bx2, by2 = int(box["x"] + box["w"]), int(box["y"] + box["h"])
            bbox_cx_tmp = (bx1 + bx2) / 2.0
            bbox_cy_tmp = (by1 + by2) / 2.0

            was_static = trk_mgr.is_static(bbox_cx_tmp, bbox_cy_tmp)
            if not was_static and prev_bbox_states:
                for st in prev_bbox_states:
                    if np.hypot(bbox_cx_tmp - st[0], bbox_cy_tmp - st[1]) < bbox_match_dist:
                        was_static = st[4]
                        break

            if not was_static:
                box_mask = ((events["x"] >= bx1) & (events["x"] < bx2) &
                            (events["y"] >= by1) & (events["y"] < by2))
                bg_mask &= ~box_mask

        bg_events = events[bg_mask]
        if max_bg_events > 0 and len(bg_events) > max_bg_events:
            _idx = np.random.choice(len(bg_events), max_bg_events, replace=False)
            _idx.sort()
            bg_events = bg_events[_idx]

        min_bg_events = 500
        bg_sharp = 0.0
        if len(bg_events) < min_bg_events:
            raw_foe_x, raw_foe_y, ego_s = foe_x, foe_y, 0.0
            raw_omega_y = 0.0
        else:
            if prev_foe:
                foe_x_coarse = np.arange(foe_x - warm_foe_x,
                                          foe_x + warm_foe_x + bg_foe_coarse_x,
                                          bg_foe_coarse_x)
                foe_y_coarse = np.arange(foe_y - warm_foe_y,
                                          foe_y + warm_foe_y + bg_foe_coarse_y,
                                          bg_foe_coarse_y)
                s_coarse = np.linspace(
                    max(s_min, kf_s.value - warm_start_s),
                    min(s_range_max, kf_s.value + warm_start_s),
                    n_s_grid
                )
            else:
                foe_x_coarse = np.arange(-float(width) * 0.25,
                                          float(width) * 1.25 + bg_foe_coarse_x,
                                          bg_foe_coarse_x)
                foe_y_coarse = np.arange(foe_init_y - float(height) * 0.15,
                                          foe_init_y + float(height) * 0.15 + bg_foe_coarse_y,
                                          bg_foe_coarse_y)
                s_coarse = s_range_full

            c_fx, c_fy, c_s, c_sharp = grid_search_bg_foe_and_s(
                bg_events, t_ref, height, width,
                foe_x_vals=foe_x_coarse, foe_y_vals=foe_y_coarse,
                s_vals=s_coarse,
                foe_scoring=foe_scoring,
                n_cells=foe_n_cells,
            )

            if enable_yaw:
                raw_foe_x, raw_foe_y, ego_s, raw_omega_y, bg_sharp, _ = \
                    optimize_ego_motion_4param(
                        bg_events, t_ref, height, width, cx, cy, intr.f_avg,
                        init_foe_x=c_fx, init_foe_y=c_fy, init_s=c_s,
                        init_omega_y=kf_omega_y.value,
                        s_max=s_range_max, omega_y_max=omega_y_max,
                        foe_x_bounds=(foe_x_min, foe_x_max),
                        foe_y_bounds=(foe_y_min, foe_y_max),
                        yaw_reg_lambda=yaw_reg_lambda,
                        foe_reg_lambda_x=foe_reg_lambda_x,
                        foe_reg_lambda_y=foe_reg_lambda_y,
                        foe_ref_x=foe_init_x,
                        foe_ref_y=foe_init_y,
                    )
            else:
                foe_x_fine = np.arange(c_fx - bg_foe_coarse_x,
                                        c_fx + bg_foe_coarse_x + bg_foe_fine_x,
                                        bg_foe_fine_x)
                foe_y_fine = np.arange(c_fy - bg_foe_coarse_y,
                                        c_fy + bg_foe_coarse_y + bg_foe_fine_y,
                                        bg_foe_fine_y)
                s_half = (s_coarse[1] - s_coarse[0]) if len(s_coarse) > 1 else 1.0
                s_fine = np.linspace(max(s_min, c_s - s_half),
                                     min(s_range_max, c_s + s_half), n_s_grid)

                raw_foe_x, raw_foe_y, ego_s, bg_sharp = grid_search_bg_foe_and_s(
                    bg_events, t_ref, height, width,
                    foe_x_vals=foe_x_fine, foe_y_vals=foe_y_fine,
                    s_vals=s_fine,
                    foe_scoring=foe_scoring,
                    n_cells=foe_n_cells,
                )
                raw_omega_y = 0.0

        if bg_sharp > 0:
            if ego_sharpness_ema <= 0:
                ego_sharpness_ema = bg_sharp
            else:
                ego_sharpness_ema += 0.05 * (bg_sharp - ego_sharpness_ema)
            for kf in (kf_s, kf_foe_x, kf_foe_y, kf_omega_y):
                kf.sharpness_ref = ego_sharpness_ema

        if enable_yaw and kf_omega_y.initialized:
            kf_omega_y.predict(dt_sec)
        if kf_s.initialized:
            kf_s.predict(dt_sec)
        foe_x_ctrl = 0.0
        if enable_yaw and foe_yaw_coupling != 0.0 and kf_omega_y.initialized:
            foe_x_ctrl = foe_yaw_coupling * intr.f_avg * kf_omega_y.value
        if kf_foe_x.initialized:
            kf_foe_x.predict(dt_sec, control_vel=foe_x_ctrl)
        if kf_foe_y.initialized:
            kf_foe_y.predict(dt_sec)

        if bg_sharp > 0:
            kf_s.update(ego_s, bg_sharp, dt_sec)

            yaw_for_foe_gate = (max(abs(raw_omega_y), abs(kf_omega_y.value))
                                if enable_yaw else 0.0)
            foe_measure_reliable = (
                (abs(ego_s) >= s_min_for_foe) or
                (enable_yaw and yaw_for_foe_gate >= foe_turn_yaw_enter)
            )
            if foe_measure_reliable:
                kf_foe_x.update(raw_foe_x, bg_sharp, dt_sec)
                kf_foe_y.update(raw_foe_y, bg_sharp, dt_sec)

            if enable_yaw:
                kf_omega_y.update(raw_omega_y, bg_sharp, dt_sec)

        smooth_s = kf_s.value
        foe_x = kf_foe_x.value
        foe_y = kf_foe_y.value
        smooth_omega_y = kf_omega_y.value if enable_yaw else 0.0

        prev_stop_state = stop_state
        not_turning_now = (not enable_yaw) or (abs(smooth_omega_y) < foe_turn_yaw_exit)
        if abs(smooth_s) < foe_stop_s_thresh and not_turning_now:
            stop_counter += 1
        else:
            stop_counter = 0
        stop_state = (stop_counter >= max(1, int(foe_stop_hold_frames)))

        if stop_state:
            alpha = 1.0 - np.exp(-max(0.0, foe_stop_revert_gain) * dt_sec)
            rate_decay = np.exp(-max(0.0, foe_stop_revert_gain) * dt_sec)
            kf_foe_x.val += alpha * (foe_init_x - kf_foe_x.val)
            kf_foe_y.val += alpha * (foe_init_y - kf_foe_y.val)
            kf_foe_x.rate *= rate_decay
            kf_foe_y.rate *= rate_decay
            foe_x = kf_foe_x.value
            foe_y = kf_foe_y.value

        if prev_stop_state != stop_state:
            logger.info(
                f"Frame {i + 1}: stop_revert={'ON' if stop_state else 'OFF'}  "
                f"s={smooth_s:.3f}  yaw={smooth_omega_y:.3f}"
            )

        foe_sane_x_lo = foe_x_min - float(width) * 0.1
        foe_sane_x_hi = foe_x_max + float(width) * 0.1
        foe_sane_y_lo = foe_y_min - float(height) * 0.1
        foe_sane_y_hi = foe_y_max + float(height) * 0.1
        foe_oob = (foe_x < foe_sane_x_lo or foe_x > foe_sane_x_hi or
                   foe_y < foe_sane_y_lo or foe_y > foe_sane_y_hi)
        if foe_oob:
            foe_diverge_count += 1
        else:
            foe_diverge_count = 0

        if foe_diverge_count >= FOE_DIVERGE_MAX:
            logger.warning(
                f"Frame {frame_count}: FOE diverged ({foe_x:.0f},{foe_y:.0f}) "
                f"for {foe_diverge_count} consecutive frames → resetting to "
                f"neutral ({foe_init_x:.0f},{foe_init_y:.0f})"
            )
            kf_foe_x.reset(logger=logger, reason="dynamic sane-zone divergence")
            kf_foe_y.reset(logger=logger, reason="dynamic sane-zone divergence")
            foe_x = kf_foe_x.value
            foe_y = kf_foe_y.value
            foe_diverge_count = 0

        prev_foe = True

        trk_mgr.predict_all(dt_sec)
        cur_trk_entries = []
        cur_bbox_states = []

        for box in boxes:
            bx1, by1 = int(box["x"]), int(box["y"])
            bx2, by2 = int(box["x"] + box["w"]), int(box["y"] + box["h"])
            bbox_cx = (bx1 + bx2) / 2.0
            bbox_cy = (by1 + by2) / 2.0

            bbox_mask = ((events["x"] >= bx1) & (events["x"] < bx2) &
                         (events["y"] >= by1) & (events["y"] < by2))
            bbox_events = events[bbox_mask]
            if max_bbox_events > 0 and len(bbox_events) > max_bbox_events:
                _idx = np.random.choice(len(bbox_events), max_bbox_events, replace=False)
                _idx.sort()
                bbox_events = bbox_events[_idx]

            skip_computation = False
            if len(bbox_events) < min_bbox_events:
                skip_computation = True
            elif float(box["h"]) * float(box["w"]) < min_bbox_area:
                skip_computation = True
            else:
                px1 = max(0, bx1 - bbox_pad)
                py1 = max(0, by1 - bbox_pad)
                px2 = min(width, bx2 + bbox_pad)
                py2 = min(height, by2 + bbox_pad)
                pw = int(px2 - px1)
                ph = int(py2 - py1)
                if pw <= 4 or ph <= 4:
                    skip_computation = True

            if skip_computation:
                is_static = True
                n_static_trans += 1
                n_static_radial += 1
                color = COLOR_STATIC
                label_text = "S (skip)"
                draw_labeled_bbox(img, box, label_text, color,
                                  output_scale=output_scale, ui_scale=ui_scale)
                if label_output_npy:
                    row = np.empty(1, dtype=LABEL_OUT_DTYPE)
                    row['t'] = int(box['t'])
                    row['x'] = float(box['x'])
                    row['y'] = float(box['y'])
                    row['w'] = float(box['w'])
                    row['h'] = float(box['h'])
                    row['class_id'] = 1
                    row['class_confidence'] = float(box['class_confidence'])
                    row['track_id'] = int(box['track_id'])
                    label_out_rows.append(row)
                continue

            matched_t = None
            if prev_bbox_states:
                best_d = bbox_match_dist
                for st in prev_bbox_states:
                    d = float(np.hypot(bbox_cx - st[0], bbox_cy - st[1]))
                    if d < best_d:
                        best_d = d
                        matched_t = st

            ego_vx_bbox, ego_vy_bbox = compute_ego_velocity(
                bbox_cx, bbox_cy, foe_x, foe_y, smooth_s, smooth_omega_y,
                cx, cy, intr.f_avg
            )

            vx_total_t = vy_total_t = 0.0
            res_val_t = 0.0
            is_static_t = True
            did_translational = False

            if bbox_method in ("translational", "both"):
                did_translational = True
                raw_vx, raw_vy, _ = estimate_bbox_total_velocity(
                    bbox_events, t_ref,
                    patch_x1=px1, patch_y1=py1,
                    patch_w=pw, patch_h=ph,
                    search_range=scaled_vel_range,
                    coarse_step=scaled_coarse_step,
                    fine_step=scaled_fine_step
                )
                if matched_t is not None:
                    vx_total_t = bbox_ema_alpha * raw_vx + (1 - bbox_ema_alpha) * matched_t[2]
                    vy_total_t = bbox_ema_alpha * raw_vy + (1 - bbox_ema_alpha) * matched_t[3]
                else:
                    vx_total_t, vy_total_t = raw_vx, raw_vy

                res_val_t = residual_fn(vx_total_t, vy_total_t,
                                        ego_vx_bbox, ego_vy_bbox, intr.f_avg)
                is_static_t = (res_val_t <= residual_threshold)

            foe_vx_r = foe_vy_r = 0.0
            res_val_r = 0.0
            is_static_r = True
            did_radial = False
            trk = None

            if bbox_method in ("radial", "both"):
                did_radial = True
                local_foe_x, local_foe_y, local_s, local_sharp = estimate_bbox_foe_s(
                    bbox_events, t_ref,
                    patch_x1=px1, patch_y1=py1,
                    patch_w=pw, patch_h=ph,
                    bbox_cx=bbox_cx, bbox_cy=bbox_cy,
                    s_max=s_range_max,
                    n_s_coarse=n_s_grid,
                    foe_range=scaled_foe_range,
                    foe_coarse_step=scaled_foe_coarse,
                    foe_fine_step=scaled_foe_fine
                )
                raw_foe_vx = local_s * (bbox_cx - local_foe_x)
                raw_foe_vy = local_s * (bbox_cy - local_foe_y)

                if local_sharp > 0:
                    if bbox_sharpness_ema <= 0:
                        bbox_sharpness_ema = local_sharp
                    else:
                        bbox_sharpness_ema += 0.05 * (local_sharp - bbox_sharpness_ema)
                    trk_mgr.sharpness_ref = bbox_sharpness_ema
                    for _tc, _ty, trk_obj, _ex in trk_mgr.trackers:
                        trk_obj.sharpness_ref = bbox_sharpness_ema

                trk = trk_mgr.get_tracker(bbox_cx, bbox_cy)
                if trk is None:
                    if bbox_reinit_mode == "zero":
                        vx_i, vy_i = 0.0, 0.0
                    elif bbox_reinit_mode == "direction_only":
                        ego_mag = float(np.hypot(ego_vx_bbox, ego_vy_bbox))
                        if ego_mag > 1e-3:
                            vx_i = ego_vx_bbox / ego_mag * 1.0
                            vy_i = ego_vy_bbox / ego_mag * 1.0
                        else:
                            vx_i, vy_i = 0.0, 0.0
                    else:
                        vx_i, vy_i = ego_vx_bbox, ego_vy_bbox
                    trk = trk_mgr.create_tracker(bbox_cx, bbox_cy,
                                                  vx_init=vx_i, vy_init=vy_i)
                    trk.sharpness_ref = trk_mgr.sharpness_ref
                    trk.update(raw_foe_vx, raw_foe_vy, max(local_sharp, 1e-6), dt_sec)
                else:
                    trk.update(raw_foe_vx, raw_foe_vy, max(local_sharp, 1e-6), dt_sec)
                    if trk.misses >= bbox_gated_reset_n and bbox_gated_reset_n > 0:
                        trk.soft_reset(raw_foe_vx, raw_foe_vy)

                foe_vx_r, foe_vy_r = trk.state

                res_val_r = residual_fn(foe_vx_r, foe_vy_r,
                                        ego_vx_bbox, ego_vy_bbox, intr.f_avg)
                is_static_r = (res_val_r <= residual_threshold)

                if trk.gated:
                    n_gated_radial += 1

            if bbox_method == "translational":
                is_static = is_static_t
            elif bbox_method == "radial":
                is_static = is_static_r
            else:
                is_static = is_static_t

            if did_radial and trk is not None:
                cur_trk_entries.append((
                    bbox_cx, bbox_cy, trk,
                    {"is_static": is_static, "is_static_r": is_static_r},
                ))

            cur_bbox_states.append((
                bbox_cx, bbox_cy,
                float(vx_total_t), float(vy_total_t),
                is_static
            ))

            if did_translational:
                if is_static_t:
                    n_static_trans += 1
                else:
                    n_moving_trans += 1
            if did_radial:
                if is_static_r:
                    n_static_radial += 1
                else:
                    n_moving_radial += 1

            bbox_gated = (did_radial and trk is not None and trk.gated)

            if label_output_npy:
                row = np.empty(1, dtype=LABEL_OUT_DTYPE)
                row['t'] = int(box['t'])
                row['x'] = float(box['x'])
                row['y'] = float(box['y'])
                row['w'] = float(box['w'])
                row['h'] = float(box['h'])
                row['class_id'] = 1 if is_static else 0
                row['class_confidence'] = float(box['class_confidence'])
                row['track_id'] = int(box['track_id'])
                label_out_rows.append(row)

            if bbox_method == "translational":
                tag = "S" if is_static_t else "M"
                label_text = f"{tag} ({res_val_t:.1f})"
                color = COLOR_STATIC if is_static_t else COLOR_MOVING
            elif bbox_method == "radial":
                tag = "S" if is_static_r else "M"
                color = COLOR_STATIC if is_static_r else COLOR_MOVING
                label_text = f"{tag} ({res_val_r:.1f})"
            else:
                tag_t = "S" if is_static_t else "M"
                tag_r = "S" if is_static_r else "M"
                color = COLOR_STATIC if is_static_r else COLOR_MOVING
                label_text = f"{tag_t}/{tag_r} ({res_val_t:.1f}/{res_val_r:.1f})"

            draw_labeled_bbox(img, box, label_text, color,
                              output_scale=output_scale, ui_scale=ui_scale)

            bc = (_s(bbox_cx, output_scale), _s(bbox_cy, output_scale))
            th_arrow = max(1, int(round(ui_scale)))

            draw_velocity_arrow(img, bc, ego_vx_bbox, ego_vy_bbox, COLOR_EGO,
                                scale_px=0.15, thickness=th_arrow,
                                output_scale=output_scale)
            if did_translational:
                draw_velocity_arrow(img, bc, vx_total_t, vy_total_t, COLOR_OBJ_VEL,
                                    scale_px=0.15, thickness=th_arrow,
                                    output_scale=output_scale)
            if did_radial:
                draw_velocity_arrow(img, bc, foe_vx_r, foe_vy_r, COLOR_FOE_VEL,
                                    scale_px=0.15, thickness=th_arrow,
                                    output_scale=output_scale)

        trk_mgr.update_positions(cur_trk_entries)
        prev_bbox_states = cur_bbox_states

        # Global overlays
        foe_xi, foe_yi = _s(foe_x, output_scale), _s(foe_y, output_scale)
        cross_len = max(10, int(round(10 * ui_scale)))
        cross_th = max(1, int(round(ui_scale)))

        cv2.line(img, (foe_xi - cross_len, foe_yi), (foe_xi + cross_len, foe_yi),
                 COLOR_EGO, cross_th)
        cv2.line(img, (foe_xi, foe_yi - cross_len), (foe_xi, foe_yi + cross_len),
                 COLOR_EGO, cross_th)

        font = cv2.FONT_HERSHEY_SIMPLEX
        info_th = 2
        fs_big = 0.32 * ui_scale
        fs_mid = 0.28 * ui_scale
        fs_small = 0.25 * ui_scale

        cv2.putText(img, "FOE (est)",
                    (foe_xi + _s(12, ui_scale), foe_yi - _s(4, ui_scale)),
                    font, 0.22 * ui_scale, COLOR_EGO, 1, cv2.LINE_AA)

        cv2.putText(img, f"s_bg={smooth_s:.2f} 1/s",
                    (_s(10, ui_scale), _s(20, ui_scale)),
                    font, fs_big, COLOR_INFO, info_th, cv2.LINE_AA)
        cv2.putText(img, f"FOE=({foe_x:.0f},{foe_y:.0f})",
                    (_s(10, ui_scale), _s(38, ui_scale)),
                    font, fs_mid, COLOR_INFO, info_th, cv2.LINE_AA)
        cv2.putText(img, f"sharp={bg_sharp:.1f}",
                    (_s(10, ui_scale), _s(53, ui_scale)),
                    font, fs_mid, COLOR_INFO, info_th, cv2.LINE_AA)
        if enable_yaw:
            cv2.putText(
                img,
                f"yaw={smooth_omega_y:.3f} rad/s ({np.degrees(smooth_omega_y):.1f} deg/s)",
                (_s(10, ui_scale), _s(68, ui_scale)),
                font, fs_mid, COLOR_INFO, info_th, cv2.LINE_AA)
            _thresh_y = _s(83, ui_scale)
        else:
            _thresh_y = _s(68, ui_scale)
        cv2.putText(img, f"thresh={residual_threshold:.2f} ({residual_method})",
                    (_s(10, ui_scale), _thresh_y),
                    font, fs_mid, COLOR_INFO, info_th, cv2.LINE_AA)

        legend_lines = [("EGO arrow (expected)", COLOR_EGO)]
        if bbox_method in ("translational", "both"):
            legend_lines.append(("OBJ arrow (measured)", COLOR_OBJ_VEL))
        if bbox_method in ("radial", "both"):
            legend_lines.append(("FOE+s arrow (radial)", COLOR_FOE_VEL))

        h_legend = out_h - _s(10, ui_scale)
        for li, (ltxt, lcol) in enumerate(reversed(legend_lines)):
            cv2.putText(img, ltxt,
                        (_s(10, ui_scale), h_legend - _s(15 * li, ui_scale)),
                        font, fs_small, lcol, info_th, cv2.LINE_AA)
        y_sm = min(out_h - _s(2, ui_scale), h_legend + _s(15, ui_scale))
        cv2.putText(img, "STATIC", (_s(10, ui_scale), y_sm),
                    font, fs_small, COLOR_STATIC, info_th, cv2.LINE_AA)
        cv2.putText(img, "MOVING", (_s(80, ui_scale), y_sm),
                    font, fs_small, COLOR_MOVING, info_th, cv2.LINE_AA)

        writer.write(img)
        frame_count += 1

    writer.release()

    if label_output_npy:
        if len(label_out_rows) == 0:
            logger.warning(f"Label dump requested but no detections: {label_output_npy}")
        else:
            label_out = np.concatenate(label_out_rows, axis=0)
            np.save(label_output_npy, label_out)
            logger.info(f"Saved predicted labels: {len(label_out)} rows -> {label_output_npy}")

    elapsed = time.time() - t_start
    if bbox_method == "radial":
        n_total = n_static_radial + n_moving_radial
    else:
        n_total = n_static_trans + n_moving_trans
    logger.info(f"Saved {frame_count} frames to {output_file}")
    logger.info(
        f"Runtime: {elapsed // 3600:.0f}h {(elapsed % 3600) // 60:.0f}m "
        f"{(elapsed % 3600) % 60:.1f}s  ({(elapsed / max(1, frame_count)):.3f}s/frame)"
    )
    logger.info(f"Total detections: {n_total}")
    if bbox_method in ("translational", "both"):
        logger.info(f"  Translational (T): STATIC={n_static_trans}  MOVING={n_moving_trans}")
    if bbox_method in ("radial", "both"):
        logger.info(
            f"  Radial  (FOE+s R): STATIC={n_static_radial}  MOVING={n_moving_radial}  "
            f"GATED={n_gated_radial}"
        )
    logger.info(
        f"Ego Kalman gated: s={kf_s.n_gated}  foe_x={kf_foe_x.n_gated}  "
        f"foe_y={kf_foe_y.n_gated}  omega_y={kf_omega_y.n_gated}"
    )
