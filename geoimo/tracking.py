import numpy as np
from typing import Optional, Tuple


def _wrap_angle_rad(theta: float) -> float:
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


class EgoParamKalman:
    """2-D Kalman filter (value, rate) for a single ego-motion parameter."""

    def __init__(self, val_init: float = 0.0,
                 P_val: float = 1000.0,
                 P_rate: float = 100.0,
                 q_val: float = 1.0,
                 q_rate: float = 0.25,
                 base_R: float = 50.0,
                 dt_ref: float = 0.066666,
                 sharpness_ref: float = 1.0,
                 gate: float = 4.0,
                 val_min: float = -1e9,
                 val_max: float = 1e9,
                 mean_revert_theta: float = 0.0,
                 mean_revert_target: Optional[float] = None):
        self.val_init = val_init
        self.val = val_init
        self.rate = 0.0
        self._P_val_init = P_val
        self._P_rate_init = P_rate
        self.P_00 = P_val
        self.P_01 = 0.0
        self.P_11 = P_rate
        self.q_val = q_val
        self.q_rate = q_rate
        self.base_R = base_R
        self.dt_ref = dt_ref
        self.sharpness_ref = sharpness_ref
        self.gate = gate
        self.val_min = val_min
        self.val_max = val_max
        self.mean_revert_theta = mean_revert_theta
        self.mean_revert_target = val_init if mean_revert_target is None else mean_revert_target
        self.initialized = False
        self.gated = False
        self.n_gated = 0
        self.n_resets = 0

    def reset(self, logger=None, reason: str = "") -> None:
        self.n_resets += 1
        if logger:
            logger.warning(
                f"EgoKalman RESET #{self.n_resets}: val={self.val:.2f} → {self.val_init:.2f}  ({reason})"
            )
        self.val = self.val_init
        self.rate = 0.0
        self.P_00 = self._P_val_init
        self.P_01 = 0.0
        self.P_11 = self._P_rate_init
        self.initialized = False
        self.gated = False

    def predict(self, dt_sec: float, control_vel: float = 0.0) -> None:
        dt = dt_sec
        self.val += (self.rate + control_vel) * dt

        if self.mean_revert_theta > 0:
            alpha = 1.0 - np.exp(-self.mean_revert_theta * dt)
            self.val += alpha * (self.mean_revert_target - self.val)
            self.rate *= np.exp(-self.mean_revert_theta * dt)

        p00 = self.P_00 + 2.0 * self.P_01 * dt + self.P_11 * dt * dt
        p01 = self.P_01 + self.P_11 * dt
        p11 = self.P_11

        dt2 = dt * dt
        dt3 = dt2 * dt

        self.P_00 = p00 + self.q_rate * dt3 / 3.0 + self.q_val * dt
        self.P_01 = p01 + self.q_rate * dt2 / 2.0
        self.P_11 = p11 + self.q_rate * dt
        self.gated = False

    def update(self, z: float, sharpness: float, dt_sec: float) -> None:
        dt_scale = self.dt_ref / max(dt_sec, 1e-9)
        R = self.base_R * dt_scale * (self.sharpness_ref / max(sharpness, 1e-6))

        if not self.initialized:
            self.val = float(np.clip(z, self.val_min, self.val_max))
            self.rate = 0.0
            self.initialized = True
            return

        innov = z - self.val
        S = self.P_00 + R

        if innov * innov / S > self.gate * self.gate:
            self.gated = True
            self.n_gated += 1
            return

        K0 = self.P_00 / S
        K1 = self.P_01 / S

        self.val += K0 * innov
        self.rate += K1 * innov

        old_P01 = self.P_01
        self.P_00 = (1.0 - K0) * self.P_00
        self.P_01 = (1.0 - K0) * old_P01
        self.P_11 = self.P_11 - K1 * old_P01

        self.val = float(np.clip(self.val, self.val_min, self.val_max))

    @property
    def value(self) -> float:
        return self.val


class BboxKalmanTracker:
    """Cartesian (vx, vy) Kalman tracker for a bounding box."""

    def __init__(self, vx_init: float, vy_init: float,
                 P_init: float = 2500.0,
                 q: float = 100.0,
                 base_R: float = 400.0,
                 dt_ref: float = 0.066666,
                 sharpness_ref: float = 1.0,
                 gate: float = 4.0):
        self.vx = vx_init
        self.vy = vy_init
        self.P_vx = P_init
        self.P_vy = P_init
        self.q = q
        self.base_R = base_R
        self.dt_ref = dt_ref
        self.sharpness_ref = sharpness_ref
        self.gate = gate
        self.age = 0
        self.hits = 0
        self.misses = 0
        self.last_sharpness = 0.0
        self.gated = False

    def predict(self, dt_sec: float) -> None:
        self.P_vx += self.q * dt_sec
        self.P_vy += self.q * dt_sec
        self.age += 1
        self.gated = False

    def update(self, z_vx: float, z_vy: float,
               sharpness: float, dt_sec: float) -> None:
        self.last_sharpness = sharpness
        R = self._measurement_noise(sharpness, dt_sec)

        innov_vx = z_vx - self.vx
        innov_vy = z_vy - self.vy

        S_vx = self.P_vx + R
        S_vy = self.P_vy + R

        maha_sq = (innov_vx * innov_vx) / S_vx + (innov_vy * innov_vy) / S_vy
        if maha_sq > self.gate * self.gate:
            self.misses += 1
            self.gated = True
            return

        K_vx = self.P_vx / S_vx
        K_vy = self.P_vy / S_vy

        self.vx += K_vx * innov_vx
        self.vy += K_vy * innov_vy

        self.P_vx *= (1.0 - K_vx)
        self.P_vy *= (1.0 - K_vy)

        self.hits += 1
        self.misses = 0

    def _measurement_noise(self, sharpness: float, dt_sec: float) -> float:
        s = max(sharpness, 1e-6)
        dt_scale = self.dt_ref / max(dt_sec, 1e-9)
        return self.base_R * dt_scale * (self.sharpness_ref / s)

    def soft_reset(self, z_vx: float, z_vy: float, P_init: float = 2500.0) -> None:
        self.vx = z_vx
        self.vy = z_vy
        self.P_vx = P_init
        self.P_vy = P_init
        self.misses = 0
        self.gated = False

    @property
    def state(self) -> Tuple[float, float]:
        return (self.vx, self.vy)


class BboxPolarKalmanTracker:
    """Polar-domain (speed, direction) Kalman tracker for a bounding box."""

    def __init__(self, vx_init: float, vy_init: float,
                 P_speed_init: float = 2500.0,
                 P_theta_init: float = 0.5,
                 q_speed: float = 100.0,
                 q_theta: float = 0.02,
                 base_R_speed: float = 400.0,
                 base_R_theta: float = 0.12,
                 dt_ref: float = 0.066666,
                 sharpness_ref: float = 1.0,
                 gate_speed: float = 4.0,
                 gate_theta: float = 3.0,
                 min_speed_for_theta_update: float = 8.0):
        self.speed = float(np.hypot(vx_init, vy_init))
        self.theta = (float(np.arctan2(vy_init, vx_init))
                      if self.speed > 1e-6 else 0.0)

        self.P_speed = P_speed_init
        self.P_theta = P_theta_init

        self._P_speed_init = P_speed_init
        self._P_theta_init = P_theta_init

        self.q_speed = q_speed
        self.q_theta = q_theta
        self.base_R_speed = base_R_speed
        self.base_R_theta = base_R_theta
        self.dt_ref = dt_ref
        self.sharpness_ref = sharpness_ref
        self.gate_speed = gate_speed
        self.gate_theta = gate_theta
        self.min_speed_for_theta_update = min_speed_for_theta_update

        self.age = 0
        self.hits = 0
        self.misses = 0
        self.last_sharpness = 0.0
        self.gated = False
        self.initialized = False

    def predict(self, dt_sec: float) -> None:
        self.P_speed += self.q_speed * dt_sec
        self.P_theta += self.q_theta * dt_sec
        self.age += 1
        self.gated = False

    def update(self, z_vx: float, z_vy: float,
               sharpness: float, dt_sec: float) -> None:
        self.last_sharpness = sharpness
        z_speed = float(np.hypot(z_vx, z_vy))
        z_theta = (float(np.arctan2(z_vy, z_vx))
                   if z_speed > 1e-9 else self.theta)

        R_speed, R_theta = self._measurement_noises(sharpness, dt_sec)

        if not self.initialized:
            self.speed = z_speed
            if z_speed >= self.min_speed_for_theta_update:
                self.theta = z_theta
            self.initialized = True
            self.hits += 1
            self.misses = 0
            self.gated = False
            return

        updated_any = False
        speed_gated = False
        theta_gated = False

        innov_speed = z_speed - self.speed
        S_speed = self.P_speed + R_speed
        if (innov_speed * innov_speed) / max(S_speed, 1e-12) > self.gate_speed * self.gate_speed:
            speed_gated = True
        else:
            K_speed = self.P_speed / S_speed
            self.speed += K_speed * innov_speed
            self.P_speed *= (1.0 - K_speed)
            updated_any = True

        if z_speed >= self.min_speed_for_theta_update:
            innov_theta = _wrap_angle_rad(z_theta - self.theta)
            S_theta = self.P_theta + R_theta
            if (innov_theta * innov_theta) / max(S_theta, 1e-12) > self.gate_theta * self.gate_theta:
                theta_gated = True
            else:
                K_theta = self.P_theta / S_theta
                self.theta = _wrap_angle_rad(self.theta + K_theta * innov_theta)
                self.P_theta *= (1.0 - K_theta)
                updated_any = True

        if updated_any:
            self.hits += 1
            self.misses = 0
            self.gated = bool(speed_gated or theta_gated)
        else:
            self.misses += 1
            self.gated = True

    def _measurement_noises(self, sharpness: float, dt_sec: float) -> Tuple[float, float]:
        s = max(sharpness, 1e-6)
        dt_scale = self.dt_ref / max(dt_sec, 1e-9)
        scale = dt_scale * (self.sharpness_ref / s)
        return self.base_R_speed * scale, self.base_R_theta * scale

    def soft_reset(self, z_vx: float, z_vy: float, P_init: float = 2500.0) -> None:
        z_speed = float(np.hypot(z_vx, z_vy))
        self.speed = z_speed
        if z_speed >= 1e-9:
            self.theta = float(np.arctan2(z_vy, z_vx))
        self.P_speed = P_init
        self.P_theta = self._P_theta_init
        self.misses = 0
        self.gated = False

    @property
    def state(self) -> Tuple[float, float]:
        vx = self.speed * np.cos(self.theta)
        vy = self.speed * np.sin(self.theta)
        return float(vx), float(vy)


class BboxTrackerManager:
    """Pool of per-bbox Kalman trackers with greedy nearest-centre matching."""

    def __init__(self, match_dist: float = 40.0,
                 max_misses: int = 5,
                 P_init: float = 2500.0,
                 q: float = 100.0,
                 base_R: float = 400.0,
                 dt_ref: float = 0.066666,
                 sharpness_ref: float = 1.0,
                 gate: float = 4.0,
                 tracker_mode: str = "cartesian",
                 polar_q_theta: float = 0.02,
                 polar_q_speed: float = 100.0,
                 polar_base_r_theta: float = 0.12,
                 polar_base_r_speed: float = 400.0,
                 polar_gate_theta: float = 3.0,
                 polar_gate_speed: float = 4.0,
                 polar_min_speed_theta: float = 8.0):
        self.match_dist = match_dist
        self.max_misses = max_misses
        self.P_init = P_init
        self.q = q
        self.base_R = base_R
        self.dt_ref = dt_ref
        self.sharpness_ref = sharpness_ref
        self.gate = gate
        self.tracker_mode = tracker_mode
        self.polar_q_theta = polar_q_theta
        self.polar_q_speed = polar_q_speed
        self.polar_base_r_theta = polar_base_r_theta
        self.polar_base_r_speed = polar_base_r_speed
        self.polar_gate_theta = polar_gate_theta
        self.polar_gate_speed = polar_gate_speed
        self.polar_min_speed_theta = polar_min_speed_theta
        self.trackers: list = []

    def predict_all(self, dt_sec: float) -> None:
        for _cx, _cy, trk, _extra in self.trackers:
            trk.predict(dt_sec)

    def get_tracker(self, bbox_cx: float, bbox_cy: float) -> Optional[object]:
        best_trk = None
        best_d = self.match_dist
        for t_cx, t_cy, trk, _extra in self.trackers:
            d = float(np.hypot(bbox_cx - t_cx, bbox_cy - t_cy))
            if d < best_d:
                best_d = d
                best_trk = trk
        return best_trk

    def create_tracker(self, bbox_cx: float, bbox_cy: float,
                       vx_init: float, vy_init: float,
                       extra: Optional[dict] = None) -> object:
        if self.tracker_mode == "polar":
            trk = BboxPolarKalmanTracker(
                vx_init=vx_init, vy_init=vy_init,
                P_speed_init=self.P_init, P_theta_init=0.5,
                q_speed=self.polar_q_speed, q_theta=self.polar_q_theta,
                base_R_speed=self.polar_base_r_speed,
                base_R_theta=self.polar_base_r_theta,
                dt_ref=self.dt_ref, sharpness_ref=self.sharpness_ref,
                gate_speed=self.polar_gate_speed, gate_theta=self.polar_gate_theta,
                min_speed_for_theta_update=self.polar_min_speed_theta,
            )
        else:
            trk = BboxKalmanTracker(
                vx_init, vy_init,
                P_init=self.P_init, q=self.q,
                base_R=self.base_R, dt_ref=self.dt_ref,
                sharpness_ref=self.sharpness_ref, gate=self.gate,
            )
        self.trackers.append((bbox_cx, bbox_cy, trk, extra or {}))
        return trk

    def update_positions(self, new_entries: list) -> None:
        self.trackers = [
            (cx, cy, trk, extra)
            for cx, cy, trk, extra in new_entries
            if trk.misses <= self.max_misses
        ]

    def is_static(self, bbox_cx: float, bbox_cy: float) -> bool:
        for t_cx, t_cy, _trk, extra in self.trackers:
            if float(np.hypot(bbox_cx - t_cx, bbox_cy - t_cy)) < self.match_dist:
                return extra.get("is_static", False)
        return False
