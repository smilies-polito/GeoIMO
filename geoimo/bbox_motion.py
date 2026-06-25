import numpy as np
from typing import Tuple

from geoimo.contrast import polarity_sharpness


def estimate_bbox_total_velocity(events: np.ndarray, t_ref: float,
                                  patch_x1: int, patch_y1: int,
                                  patch_w: int, patch_h: int,
                                  time_scale: float = 1e-6,
                                  search_range: float = 150.0,
                                  coarse_step: float = 30.0,
                                  fine_step: float = 8.0) -> Tuple[float, float, float]:
    if len(events) < 30:
        return 0.0, 0.0, 0.0

    t = events["t"].astype(np.float64) * time_scale
    x = events["x"].astype(np.float64) - float(patch_x1)
    y = events["y"].astype(np.float64) - float(patch_y1)
    p = events["p"]
    dt = t - t_ref

    n_pix = patch_w * patch_h
    counts = np.zeros(n_pix, dtype=np.int32)

    vx_coarse = np.arange(-search_range, search_range + coarse_step, coarse_step)
    vy_coarse = np.arange(-search_range, search_range + coarse_step, coarse_step)

    best_vx_coarse, best_vy_coarse = 0.0, 0.0
    best_sharp_coarse = -1.0

    for vy in vy_coarse:
        y_warped = np.floor(y - vy * dt).astype(np.int32)
        y_valid = (y_warped >= 0) & (y_warped < patch_h)
        for vx in vx_coarse:
            x_warped = np.floor(x - vx * dt).astype(np.int32)
            valid = (x_warped >= 0) & (x_warped < patch_w) & y_valid
            if not valid.any():
                continue
            indices = y_warped[valid] * patch_w + x_warped[valid]
            sharp = polarity_sharpness(indices, p[valid], n_pix, counts)
            if sharp > best_sharp_coarse:
                best_sharp_coarse = sharp
                best_vx_coarse = float(vx)
                best_vy_coarse = float(vy)

    vx_fine = np.arange(best_vx_coarse - coarse_step,
                        best_vx_coarse + coarse_step + fine_step, fine_step)
    vy_fine = np.arange(best_vy_coarse - coarse_step,
                        best_vy_coarse + coarse_step + fine_step, fine_step)

    best_vx, best_vy = best_vx_coarse, best_vy_coarse
    best_sharp = best_sharp_coarse

    for vy in vy_fine:
        y_warped = np.floor(y - vy * dt).astype(np.int32)
        y_valid = (y_warped >= 0) & (y_warped < patch_h)
        for vx in vx_fine:
            x_warped = np.floor(x - vx * dt).astype(np.int32)
            valid = (x_warped >= 0) & (x_warped < patch_w) & y_valid
            if not valid.any():
                continue
            indices = y_warped[valid] * patch_w + x_warped[valid]
            sharp = polarity_sharpness(indices, p[valid], n_pix, counts)
            if sharp > best_sharp:
                best_sharp = sharp
                best_vx = float(vx)
                best_vy = float(vy)

    return best_vx, best_vy, best_sharp


def estimate_bbox_foe_s(events: np.ndarray, t_ref: float,
                         patch_x1: int, patch_y1: int,
                         patch_w: int, patch_h: int,
                         bbox_cx: float, bbox_cy: float,
                         time_scale: float = 1e-6,
                         s_max: float = 5.0, n_s_coarse: int = 11,
                         foe_range: float = 200.0,
                         foe_coarse_step: float = 50.0,
                         foe_fine_step: float = 12.0) -> Tuple[float, float, float, float]:
    if len(events) < 30:
        return bbox_cx, bbox_cy, 0.0, 0.0

    t = events["t"].astype(np.float64) * time_scale
    x = events["x"].astype(np.float64)
    y = events["y"].astype(np.float64)
    p = events["p"]
    dt = t - t_ref

    n_pix = patch_w * patch_h
    counts = np.zeros(n_pix, dtype=np.int32)

    foe_xs_c = np.arange(bbox_cx - foe_range, bbox_cx + foe_range + foe_coarse_step,
                          foe_coarse_step)
    foe_ys_c = np.arange(bbox_cy - foe_range, bbox_cy + foe_range + foe_coarse_step,
                          foe_coarse_step)
    s_vals_c = np.linspace(-s_max, s_max, n_s_coarse)

    best_fx, best_fy, best_sc = bbox_cx, bbox_cy, 0.0
    best_sharp_c = -1.0

    for fy in foe_ys_c:
        dy = y - fy
        for fx in foe_xs_c:
            dx = x - fx
            for s_val in s_vals_c:
                s_dt = s_val * dt
                xw = np.floor(x - s_dt * dx - float(patch_x1)).astype(np.int32)
                yw = np.floor(y - s_dt * dy - float(patch_y1)).astype(np.int32)
                valid = (xw >= 0) & (xw < patch_w) & (yw >= 0) & (yw < patch_h)
                if not valid.any():
                    continue
                indices = yw[valid] * patch_w + xw[valid]
                sharp = polarity_sharpness(indices, p[valid], n_pix, counts)
                if sharp > best_sharp_c:
                    best_sharp_c = sharp
                    best_fx, best_fy, best_sc = float(fx), float(fy), float(s_val)

    foe_xs_f = np.arange(best_fx - foe_coarse_step,
                          best_fx + foe_coarse_step + foe_fine_step, foe_fine_step)
    foe_ys_f = np.arange(best_fy - foe_coarse_step,
                          best_fy + foe_coarse_step + foe_fine_step, foe_fine_step)
    s_half = (s_vals_c[1] - s_vals_c[0]) if len(s_vals_c) > 1 else 1.0
    s_vals_f = np.linspace(max(-s_max, best_sc - s_half),
                           min(s_max, best_sc + s_half), n_s_coarse)

    best_fx_f, best_fy_f, best_sf = best_fx, best_fy, best_sc
    best_sharp = best_sharp_c

    for fy in foe_ys_f:
        dy = y - fy
        for fx in foe_xs_f:
            dx = x - fx
            for s_val in s_vals_f:
                s_dt = s_val * dt
                xw = np.floor(x - s_dt * dx - float(patch_x1)).astype(np.int32)
                yw = np.floor(y - s_dt * dy - float(patch_y1)).astype(np.int32)
                valid = (xw >= 0) & (xw < patch_w) & (yw >= 0) & (yw < patch_h)
                if not valid.any():
                    continue
                indices = yw[valid] * patch_w + xw[valid]
                sharp = polarity_sharpness(indices, p[valid], n_pix, counts)
                if sharp > best_sharp:
                    best_sharp = sharp
                    best_fx_f, best_fy_f, best_sf = float(fx), float(fy), float(s_val)

    return best_fx_f, best_fy_f, best_sf, best_sharp
