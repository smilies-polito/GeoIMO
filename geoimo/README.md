# GeoIMO — Algorithm Reference

This document explains the methods implemented in the `geoimo/` package, the choices
available at each stage, and the flags that control them.

---

## Pipeline overview

Each time window (`delta_t` µs) the pipeline does four things:

```
events in window
      │
      ├─► Ego-motion estimation  (ego_motion.py)
      │         FOE + scale s (+ optional yaw ωy)
      │         → Kalman-smoothed (foe_x, foe_y, s, ωy)
      │
      ├─► Per-bbox velocity      (bbox_motion.py)
      │         contrast-maximisation inside each bounding box
      │         → Kalman-smoothed (vx, vy) per track
      │
      ├─► Residual computation   (residuals.py)
      │         predicted ego velocity at bbox centre vs measured bbox velocity
      │         → scalar residual per bbox
      │
      └─► Classification         (pipeline.py)
                residual > threshold  →  MOVING
                residual ≤ threshold  →  STATIC
```

---

## 1. Ego-motion estimation

### 1.1 Focus of Expansion model

Under pure forward translation the apparent pixel velocity at a point `(x, y)` is:

```
vx = s · (x − foe_x)
vy = s · (y − foe_y)
```

where `(foe_x, foe_y)` is the Focus of Expansion (vanishing point of the
translational flow field) and `s` is a scale proportional to the forward speed
divided by depth.

### 1.2 Grid-search contrast maximisation

`grid_search_bg_foe_and_s()` finds `(foe_x, foe_y, s)` by maximising the
**polarity-split image of warped events** (`contrast.polarity_sharpness`).
Events are warped backward in time:

```
x_warped = x − s · dt · (x − foe_x)
y_warped = y − s · dt · (y − foe_y)
```

A sharp (high-contrast) image means the FOE and scale accurately describe the
motion. The search is over a discrete grid of `(foe_x, foe_y, s)` values.

**FOE scoring modes** (`--foe_scoring`, default `global`):

| Mode | Description |
|---|---|
| `global` | Sum of polarity-split squared-sum contrast over the whole frame |
| `cell_uniform` | Average contrast across spatial cells (equal weight) |
| `cell_sharpness` | Contrast-weighted average across cells |
| `cell_median` | Median cell contrast (robust to outliers) |
| `cell_trimmed` | Trimmed mean (drop top and bottom 20% of cells) |
| `cell_voting` | Each cell votes independently; results combined by weighted median |

Cell-based modes divide the frame into an `n × n` grid (`--foe_n_cells`, default 4).
They can be more robust when large independently-moving objects occupy part of the frame,
since they prevent a single region from dominating the score.

### 1.3 Yaw compensation (`--enable_yaw`)

When the camera rotates about the vertical axis (yaw rate `ωy`), the apparent
velocity acquires a rotational component:

```
vx += −(f + x̄²/f) · ωy
vy += −(x̄ · ȳ / f) · ωy
```

where `x̄ = x − cx`, `ȳ = y − cy`, and `f` is the focal length.

With `--enable_yaw`, `optimize_ego_motion_4param()` runs L-BFGS-B over
`(foe_x, foe_y, s, ωy)` jointly. Without it, `ωy = 0` is assumed (faster,
sufficient for straight-line driving).

Yaw estimation uses regularisation terms to prevent drift:
- `--yaw_reg_lambda` penalises large `ωy` (default 10.0).
- `--foe_reg_lambda_x/y` anchor the FOE to its Kalman-predicted position.

### 1.4 Ego-motion Kalman filter

The raw per-frame `(foe_x, foe_y, s, ωy)` estimates are noisy. Each parameter
is independently smoothed by an `EgoParamKalman` filter (tracking.py), a
constant-velocity 2-state Kalman filter with adaptive measurement noise:

```
R ∝ (base_R) × (dt_ref / dt) × (sharpness_ref / sharpness)
```

Measurement noise scales inversely with contrast (sharpness): high-contrast
frames get more weight; low-event frames get less.

Key tuning flags for ego-motion Kalman:

| Flag | Default | Effect |
|---|---|---|
| `--ego_kalman_q_foe` | 4.0 | Process noise for FOE position |
| `--ego_kalman_q_s` | 0.05 | Process noise for scale s |
| `--ego_kalman_q_oy` | 0.02 | Process noise for yaw ωy |
| `--ego_kalman_base_r_foe` | 500.0 | Base measurement noise for FOE |
| `--ego_kalman_base_r_s` | 1.5 | Base measurement noise for s |
| `--ego_kalman_gate` | 3.0 | Mahalanobis gate (reject outlier frames) |

---

## 2. Bbox velocity estimation

For each bounding box in the current frame, events inside the box (with
`--bbox_pad` pixel margin) are used to estimate the object's velocity.
Two methods are available.

### 2.1 Translational (`--bbox_method translational`)

`estimate_bbox_total_velocity()` finds the 2D velocity `(vx, vy)` that best
warps events to a sharp image inside the bbox patch. This is an unconstrained
search over `(vx, vy)` using a coarse-to-fine grid.

This method captures all motion, regardless of its cause. It works well when
the object's apparent motion has a significant translational component.

### 2.2 Radial (`--bbox_method radial`)

`estimate_bbox_foe_s()` fits a local FOE + scale model inside the bbox. This
constrains the motion to radially expand/contract from a local vanishing point,
which is more appropriate for objects whose main motion relative to the camera
is translational.

### 2.3 Combined (`--bbox_method both`, default)

Both translational and radial estimates are computed. The one with the higher
contrast sharpness (better fit) is selected. This is the most robust option
and is recommended for most cases.

### 2.4 Bbox Kalman tracker

Raw per-frame velocity estimates are noisy. Each tracked object has its own
Kalman filter maintained by `BboxTrackerManager`. Trackers are matched to
detections frame-to-frame by nearest centre distance (`--bbox_match_dist`,
default 40 px).

Two tracker modes are available (`--bbox_tracker_mode`):

**`cartesian` (BboxKalmanTracker)**  
Tracks `(vx, vy)` as independent 1D filters. Simple and fast. Can be
appropriate when motion direction is variable and independent.

**`polar` (BboxPolarKalmanTracker, default)**  
Tracks `(speed, θ)` in polar coordinates. Speed and direction are filtered
separately. Direction updates are suppressed when speed is below
`--polar_min_speed_theta` (default 8 px/frame), preventing noisy direction
estimates from corrupting the state. Better at maintaining consistent direction
while allowing speed to fluctuate naturally.

Key polar tracker flags:

| Flag | Default | Effect |
|---|---|---|
| `--polar_q_speed` | 200.0 | Speed process noise |
| `--polar_q_theta` | 0.02 | Direction process noise (rad²/s) |
| `--polar_base_r_speed` | 400.0 | Speed measurement noise |
| `--polar_base_r_theta` | 0.12 | Direction measurement noise (rad²) |
| `--polar_gate_speed` | 4.0 | Speed Mahalanobis gate |
| `--polar_gate_theta` | 3.0 | Direction Mahalanobis gate |

---

## 3. Residual methods

After ego-motion and bbox-velocity estimation, the pipeline computes how much
the object's measured velocity `(vx_obj, vy_obj)` differs from the predicted
ego-motion velocity `(vx_ego, vy_ego)` at the bbox centre.

Five residual methods are available (`--residual_method`):

### `relative_dynamic` (default)

```
residual = |v_obj − v_ego| / max(|v_obj|, |v_ego|, floor_eff)
```

The denominator floor adapts to the local speed regime. At low ego-speed the
floor is higher (`floor_base + floor_boost`), reducing false positives from
slow-moving cameras. At high speed the floor approaches `floor_base`,
allowing more sensitive detection.

```
floor_eff = floor_base + floor_boost · exp(−(speed / speed_transition)²)
```

Tuning flags: `--relative_floor_base` (default 5.0), `--relative_floor_boost`
(default 7.0), `--relative_floor_speed_transition` (default 12.0).
Default threshold: **0.3**.

### `relative`

```
residual = |v_obj − v_ego| / max(|v_obj|, |v_ego|, 5.0)
```

Simplified version of `relative_dynamic` with a fixed floor of 5.0.
Default threshold: **0.3**.

### `absolute`

```
residual = arctan(|v_obj − v_ego| / f) · (180/π)   [degrees]
```

Converts the residual pixel velocity to an angular error using the mean focal
length. Scale-invariant in angular space, but sensitive to choice of threshold
across different focal lengths. Default threshold: **15.0°**.

### `cosine`

```
residual = arccos(v_obj · v_ego / (|v_obj| |v_ego|))   [degrees]
```

Measures the angle between the two velocity vectors, ignoring magnitude.
Useful when speed magnitude is unreliable but direction is well estimated.
Returns 0 when either vector is near-zero. Default threshold: **30.0°**.

### `magnitude_ratio`

```
residual = ||v_obj| − |v_ego|| / max(|v_ego|, 5.0)
```

Compares only the speed magnitudes, ignoring direction. Can catch objects
moving at a different speed than ego-motion but in the same direction.
Default threshold: **0.5**.

The threshold can be overridden with `--threshold`. Set to `-1` (default) to
use the per-method default above.

---

## 4. Key flags summary

| Flag | Default | Description |
|---|---|---|
| `--delta_t` | 22858 | Frame duration in µs |
| `--skip` / `-s` | 0 | Skip first N µs of the stream |
| `--max_frames` / `-n` | -1 | Frames to process (−1 = all) |
| `--dataset` | auto | `mvsec`, `prophesee`, or auto-detect |
| `--enable_yaw` | off | Enable yaw estimation via L-BFGS-B |
| `--bbox_method` | `both` | `translational`, `radial`, or `both` |
| `--bbox_tracker_mode` | `polar` | `cartesian` or `polar` |
| `--residual_method` | `relative_dynamic` | Residual criterion (see §3) |
| `--threshold` | -1 | Residual threshold (−1 = method default) |
| `--foe_scoring` | `global` | FOE scoring mode (see §1.2) |
| `--foe_n_cells` | 4 | Grid size for cell-based FOE scoring |
| `--s_max` | 5.0 | Max expansion rate in grid search |
| `--n_s` | 21 | Number of s grid points |
| `--max_bg_events` | 200000 | Cap on background events per frame |
| `--max_bbox_events` | 30000 | Cap on events per bbox per frame |
