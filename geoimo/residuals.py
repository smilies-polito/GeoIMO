import numpy as np


def residual_absolute(vx_total: float, vy_total: float,
                      ego_vx: float, ego_vy: float, f_avg: float) -> float:
    vx_res = vx_total - ego_vx
    vy_res = vy_total - ego_vy
    residual_px = float(np.hypot(vx_res, vy_res))
    return float(np.degrees(residual_px / f_avg))


def residual_relative(vx_total: float, vy_total: float,
                      ego_vx: float, ego_vy: float, f_avg: float) -> float:
    vx_res = vx_total - ego_vx
    vy_res = vy_total - ego_vy
    res_mag = float(np.hypot(vx_res, vy_res))
    total_mag = float(np.hypot(vx_total, vy_total))
    ego_mag = float(np.hypot(ego_vx, ego_vy))
    denom = max(total_mag, ego_mag, 5.0)
    return res_mag / denom


def residual_relative_dynamic(vx_total: float, vy_total: float,
                               ego_vx: float, ego_vy: float, f_avg: float,
                               floor_base: float = 5.0,
                               floor_boost: float = 7.0,
                               speed_transition: float = 12.0) -> float:
    vx_res = vx_total - ego_vx
    vy_res = vy_total - ego_vy
    res_mag = float(np.hypot(vx_res, vy_res))
    total_mag = float(np.hypot(vx_total, vy_total))
    ego_mag = float(np.hypot(ego_vx, ego_vy))
    speed_ref = max(total_mag, ego_mag)

    floor_base = max(0.0, float(floor_base))
    floor_boost = max(0.0, float(floor_boost))
    speed_transition = max(1e-6, float(speed_transition))
    floor_eff = floor_base + floor_boost * np.exp(-((speed_ref / speed_transition) ** 2))

    denom = max(total_mag, ego_mag, float(floor_eff))
    return res_mag / denom


def residual_cosine(vx_total: float, vy_total: float,
                    ego_vx: float, ego_vy: float, f_avg: float) -> float:
    total_mag = float(np.hypot(vx_total, vy_total))
    ego_mag = float(np.hypot(ego_vx, ego_vy))
    if total_mag < 1e-3 or ego_mag < 1e-3:
        return 0.0
    cos_sim = (vx_total * ego_vx + vy_total * ego_vy) / (total_mag * ego_mag)
    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_sim)))


def residual_magnitude_ratio(vx_total: float, vy_total: float,
                              ego_vx: float, ego_vy: float, f_avg: float) -> float:
    total_mag = float(np.hypot(vx_total, vy_total))
    ego_mag = float(np.hypot(ego_vx, ego_vy))
    denom = max(ego_mag, 5.0)
    return abs(total_mag - ego_mag) / denom


RESIDUAL_METHODS = {
    "absolute": residual_absolute,
    "relative": residual_relative,
    "relative_dynamic": residual_relative_dynamic,
    "cosine": residual_cosine,
    "magnitude_ratio": residual_magnitude_ratio,
}

RESIDUAL_DEFAULTS = {
    "absolute": 15.0,
    "relative": 0.3,
    "relative_dynamic": 0.3,
    "cosine": 30.0,
    "magnitude_ratio": 0.5,
}
