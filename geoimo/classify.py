"""Entry point: python -m geoimo.classify [options]"""
import argparse

from geoimo.logging_utils import setup_logging
from geoimo.contrast import FOE_SCORING_MODES
from geoimo.residuals import RESIDUAL_METHODS, RESIDUAL_DEFAULTS
from geoimo.pipeline import process_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GeoIMO: classify bounding-box objects as STATIC or MOVING "
                    "using geometry-driven ego-motion estimation on event streams."
    )

    # --- I/O ---
    parser.add_argument("--events", required=True,
                        help="Path to event stream file (.dat or .npy).")
    parser.add_argument("--boxes", required=True,
                        help="Path to bounding-box file (.npy).")
    parser.add_argument("-o", "--output", required=True,
                        help="Output video path (.avi).")
    parser.add_argument("--label_output_npy", type=str, default="",
                        help="Optional: save predicted labels as .npy.")

    # --- Dataset / timing ---
    parser.add_argument("--dataset", type=str, default="auto",
                        choices=["auto", "mvsec", "prophesee"],
                        help="Dataset mode (default: auto-detect by frame size).")
    parser.add_argument("--delta_t", type=int, default=22858,
                        help="Time window per frame in µs (default: 22858 ≈ 43.7 fps, "
                             "matching MVSEC outdoor_day2 GT annotation rate).")
    parser.add_argument("-s", "--skip", type=int, default=0,
                        help="Skip first N microseconds (default: 0).")
    parser.add_argument("-n", "--max_frames", type=int, default=-1,
                        help="Max frames to process (-1 = all).")

    # --- Ego-motion grid search ---
    parser.add_argument("--s_max", type=float, default=5.0,
                        help="Max expansion rate s (default: 5.0).")
    parser.add_argument("--n_s", type=int, default=21,
                        help="Number of s grid points (default: 21).")
    parser.add_argument("--s_min", type=float, default=0.0,
                        help="Min allowed s (0 = no reverse driving).")

    # --- Residual classification ---
    parser.add_argument("--residual_method", type=str, default="relative_dynamic",
                        choices=list(RESIDUAL_METHODS.keys()),
                        help="Residual method (default: relative_dynamic).")
    parser.add_argument("--threshold", type=float, default=-1.0,
                        help="Residual threshold (-1 = use method default).")
    parser.add_argument("--relative_floor_base", type=float, default=5.0)
    parser.add_argument("--relative_floor_boost", type=float, default=7.0)
    parser.add_argument("--relative_floor_speed_transition", type=float, default=12.0)

    # --- Bbox velocity method ---
    parser.add_argument("--bbox_method", type=str, default="both",
                        choices=["translational", "radial", "both"],
                        help="Bbox velocity estimation method (default: both).")
    parser.add_argument("--bbox_ema", type=float, default=0.5)
    parser.add_argument("--bbox_match_dist", type=float, default=40.0)
    parser.add_argument("--bbox_pad", type=int, default=12)
    parser.add_argument("--bbox_reinit_mode", type=str, default="full",
                        choices=["full", "direction_only", "zero"])
    parser.add_argument("--bbox_gated_reset_n", type=int, default=2)

    # --- FOE scoring ---
    parser.add_argument("--foe_scoring", type=str, default="global",
                        choices=list(FOE_SCORING_MODES),
                        help="FOE scoring mode (default: global).")
    parser.add_argument("--foe_n_cells", type=int, default=4)
    parser.add_argument("--warm_foe", type=float, default=60.0)
    parser.add_argument("--foe_max_jump", type=float, default=80.0)
    parser.add_argument("--s_min_foe", type=float, default=0.15)
    parser.add_argument("--foe_init_x", type=float, default=-1.0)
    parser.add_argument("--foe_init_y", type=float, default=-1.0)
    parser.add_argument("--foe_stop_s_thresh", type=float, default=0.12)
    parser.add_argument("--foe_stop_hold_frames", type=int, default=4)
    parser.add_argument("--foe_stop_revert_gain", type=float, default=4.0)

    # --- Yaw estimation ---
    parser.add_argument("--enable_yaw", action="store_true",
                        help="Enable yaw (ω_y) estimation via L-BFGS-B.")
    parser.add_argument("--omega_y_max", type=float, default=0.5)
    parser.add_argument("--yaw_ema_alpha", type=float, default=0.3)
    parser.add_argument("--yaw_reg_lambda", type=float, default=10.0)
    parser.add_argument("--foe_reg_lambda_x", type=float, default=-1.0)
    parser.add_argument("--foe_reg_lambda_y", type=float, default=-1.0)
    parser.add_argument("--foe_y_reg_scale", type=float, default=6.0)
    parser.add_argument("--foe_yaw_coupling", type=float, default=1.0)
    parser.add_argument("--foe_turn_yaw_enter", type=float, default=0.08)
    parser.add_argument("--foe_turn_yaw_exit", type=float, default=0.05)
    parser.add_argument("--foe_turn_yaw_ref", type=float, default=0.20)
    parser.add_argument("--foe_turn_bound_scale", type=float, default=1.0)

    # --- Ego-motion Kalman filter ---
    parser.add_argument("--ego_kalman_q_s", type=float, default=0.05)
    parser.add_argument("--ego_kalman_q_foe", type=float, default=4.0)
    parser.add_argument("--ego_kalman_q_oy", type=float, default=0.02)
    parser.add_argument("--ego_kalman_base_r_s", type=float, default=1.5)
    parser.add_argument("--ego_kalman_base_r_foe", type=float, default=500.0)
    parser.add_argument("--ego_kalman_base_r_oy", type=float, default=2.0)
    parser.add_argument("--ego_kalman_gate", type=float, default=3.0)
    parser.add_argument("--ego_kalman_q_foe_y", type=float, default=None)
    parser.add_argument("--ego_kalman_base_r_foe_y", type=float, default=None)

    # --- Bbox Kalman tracker ---
    parser.add_argument("--bbox_tracker_mode", type=str, default="polar",
                        choices=["cartesian", "polar"])
    parser.add_argument("--kalman_q", type=float, default=100.0)
    parser.add_argument("--kalman_base_r", type=float, default=400.0)
    parser.add_argument("--kalman_gate", type=float, default=4.0)
    parser.add_argument("--kalman_max_misses", type=int, default=10)
    parser.add_argument("--polar_q_theta", type=float, default=0.02)
    parser.add_argument("--polar_q_speed", type=float, default=200.0)
    parser.add_argument("--polar_base_r_theta", type=float, default=0.12)
    parser.add_argument("--polar_base_r_speed", type=float, default=400.0)
    parser.add_argument("--polar_gate_theta", type=float, default=3.0)
    parser.add_argument("--polar_gate_speed", type=float, default=4.0)
    parser.add_argument("--polar_min_speed_theta", type=float, default=8.0)

    # --- Event caps ---
    parser.add_argument("--max_bg_events", type=int, default=200_000)
    parser.add_argument("--max_bbox_events", type=int, default=30_000)

    # --- EMA (legacy, kept for compatibility) ---
    parser.add_argument("--ema_alpha", type=float, default=0.4)
    parser.add_argument("--foe_ema_alpha", type=float, default=0.2)
    parser.add_argument("--warm_s", type=float, default=4.0)

    # --- Logging ---
    parser.add_argument("--log_file", type=str, default="")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log_every", type=int, default=1)

    args = parser.parse_args()

    logger = setup_logging(log_file=(args.log_file or None), level=args.log_level)
    logger.info("GeoIMO classification starting.")
    logger.info(f"Args: {vars(args)}")

    process_video(
        args.events, args.boxes, args.output,
        dataset=args.dataset,
        delta_t=args.delta_t,
        skip=args.skip,
        max_frames=args.max_frames,
        s_range_max=args.s_max,
        n_s_grid=args.n_s,
        ema_alpha=args.ema_alpha,
        foe_ema_alpha=args.foe_ema_alpha,
        warm_start_s=args.warm_s,
        residual_threshold=args.threshold,
        residual_method=args.residual_method,
        relative_floor_base=args.relative_floor_base,
        relative_floor_boost=args.relative_floor_boost,
        relative_floor_speed_transition=args.relative_floor_speed_transition,
        bbox_method=args.bbox_method,
        bbox_ema_alpha=args.bbox_ema,
        bbox_match_dist=args.bbox_match_dist,
        bbox_pad=args.bbox_pad,
        warm_start_foe=args.warm_foe,
        foe_max_jump=args.foe_max_jump,
        s_min_for_foe=args.s_min_foe,
        foe_turn_yaw_enter=args.foe_turn_yaw_enter,
        foe_turn_yaw_exit=args.foe_turn_yaw_exit,
        foe_turn_yaw_ref=args.foe_turn_yaw_ref,
        foe_turn_bound_scale=args.foe_turn_bound_scale,
        foe_stop_s_thresh=args.foe_stop_s_thresh,
        foe_stop_hold_frames=args.foe_stop_hold_frames,
        foe_stop_revert_gain=args.foe_stop_revert_gain,
        foe_yaw_coupling=args.foe_yaw_coupling,
        enable_yaw=args.enable_yaw,
        omega_y_max=args.omega_y_max,
        yaw_ema_alpha=args.yaw_ema_alpha,
        ego_kalman_q_s=args.ego_kalman_q_s,
        ego_kalman_q_foe=args.ego_kalman_q_foe,
        ego_kalman_q_oy=args.ego_kalman_q_oy,
        ego_kalman_base_r_s=args.ego_kalman_base_r_s,
        ego_kalman_base_r_foe=args.ego_kalman_base_r_foe,
        ego_kalman_base_r_oy=args.ego_kalman_base_r_oy,
        ego_kalman_gate=args.ego_kalman_gate,
        ego_kalman_q_foe_y=args.ego_kalman_q_foe_y,
        ego_kalman_base_r_foe_y=args.ego_kalman_base_r_foe_y,
        foe_init_x=args.foe_init_x,
        foe_init_y=args.foe_init_y,
        yaw_reg_lambda=args.yaw_reg_lambda,
        foe_reg_lambda_x=args.foe_reg_lambda_x,
        foe_reg_lambda_y=args.foe_reg_lambda_y,
        foe_y_reg_scale=args.foe_y_reg_scale,
        s_min=args.s_min,
        bbox_reinit_mode=args.bbox_reinit_mode,
        bbox_gated_reset_n=args.bbox_gated_reset_n,
        kalman_q=args.kalman_q,
        kalman_base_r=args.kalman_base_r,
        kalman_gate=args.kalman_gate,
        kalman_max_misses=args.kalman_max_misses,
        bbox_tracker_mode=args.bbox_tracker_mode,
        polar_q_theta=args.polar_q_theta,
        polar_q_speed=args.polar_q_speed,
        polar_base_r_theta=args.polar_base_r_theta,
        polar_base_r_speed=args.polar_base_r_speed,
        polar_gate_theta=args.polar_gate_theta,
        polar_gate_speed=args.polar_gate_speed,
        polar_min_speed_theta=args.polar_min_speed_theta,
        foe_scoring=args.foe_scoring,
        foe_n_cells=args.foe_n_cells,
        max_bg_events=args.max_bg_events,
        max_bbox_events=args.max_bbox_events,
        label_output_npy=args.label_output_npy,
        logger=logger,
        log_every_n_frames=args.log_every,
    )


if __name__ == "__main__":
    main()
