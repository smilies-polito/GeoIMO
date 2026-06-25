import numpy as np
from typing import Tuple

from geoimo.contrast import polarity_sharpness, FOE_SCORING_MODES, _cell_score


def _neg_contrast_4param(params, x, y, dt, p, height, width, cx, cy, f_avg,
                         yaw_reg_lambda=0.0,
                         foe_reg_lambda_x=0.0,
                         foe_reg_lambda_y=0.0,
                         foe_ref_x=0.0, foe_ref_y=0.0):
    foe_x, foe_y, s, omega_y = params
    x_bar = x - cx
    y_bar = y - cy

    rot_vx = -(f_avg + x_bar * x_bar / f_avg) * omega_y
    rot_vy = -(x_bar * y_bar / f_avg) * omega_y

    vx = s * (x - foe_x) + rot_vx
    vy = s * (y - foe_y) + rot_vy

    xw = x - vx * dt
    yw = y - vy * dt

    x0 = np.floor(xw).astype(np.int32)
    y0 = np.floor(yw).astype(np.int32)
    wx = xw - x0.astype(np.float64)
    wy = yw - y0.astype(np.float64)

    n_pix = height * width
    total = 0.0

    for mask in ((p == 1), (p == 0)):
        n_m = int(mask.sum())
        if n_m < 5:
            continue

        x0m, y0m = x0[mask], y0[mask]
        wxm, wym = wx[mask], wy[mask]
        img = np.zeros(n_pix, dtype=np.float64)

        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            xi = x0m + dx
            yi = y0m + dy
            w = (wxm if dx else (1.0 - wxm)) * (wym if dy else (1.0 - wym))
            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            if valid.any():
                flat = yi[valid] * width + xi[valid]
                img += np.bincount(flat, weights=w[valid], minlength=n_pix)

        total += float((img * img).sum()) / max(1, n_m)

    if yaw_reg_lambda > 0:
        total -= yaw_reg_lambda * omega_y * omega_y
    if foe_reg_lambda_x > 0:
        total -= foe_reg_lambda_x * (foe_x - foe_ref_x) ** 2
    if foe_reg_lambda_y > 0:
        total -= foe_reg_lambda_y * (foe_y - foe_ref_y) ** 2

    return -total


def optimize_ego_motion_4param(events, t_ref, height, width,
                                cx, cy, f_avg,
                                init_foe_x, init_foe_y, init_s,
                                init_omega_y=0.0,
                                s_max=5.0, omega_y_max=0.5,
                                foe_x_bounds=None, foe_y_bounds=None,
                                time_scale=1e-6,
                                max_iter=60,
                                yaw_reg_lambda: float = 0.0,
                                foe_reg_lambda_x: float = 0.0,
                                foe_reg_lambda_y: float = 0.0,
                                foe_ref_x: float = 0.0,
                                foe_ref_y: float = 0.0):
    from scipy.optimize import minimize as sp_minimize, approx_fprime

    if len(events) < 50:
        return init_foe_x, init_foe_y, init_s, 0.0, 0.0, 0.0

    t = events["t"].astype(np.float64) * time_scale
    x = events["x"].astype(np.float64)
    y = events["y"].astype(np.float64)
    p = events["p"]
    dt = t - t_ref

    if foe_x_bounds is None:
        foe_x_bounds = (-float(width) * 0.5, float(width) * 1.5)
    if foe_y_bounds is None:
        foe_y_bounds = (-float(height) * 0.5, float(height) * 1.5)

    x0 = np.array([init_foe_x, init_foe_y, init_s, init_omega_y])
    bounds = [foe_x_bounds, foe_y_bounds, (-s_max, s_max),
              (-omega_y_max, omega_y_max)]
    args = (x, y, dt, p, height, width, cx, cy, f_avg,
            yaw_reg_lambda, foe_reg_lambda_x, foe_reg_lambda_y,
            foe_ref_x, foe_ref_y)

    eps = np.array([0.5, 0.5, 0.02, 0.005])

    def jac_fn(params, *_args):
        return approx_fprime(params, _neg_contrast_4param, eps, *args)

    result = sp_minimize(
        _neg_contrast_4param, x0, args=args, jac=jac_fn,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-10},
    )

    foe_x_opt, foe_y_opt, s_opt, oy_opt = result.x
    raw_args_no_reg = (x, y, dt, p, height, width, cx, cy, f_avg,
                       0.0, 0.0, 0.0, foe_ref_x, foe_ref_y)
    raw_contrast = float(-_neg_contrast_4param(result.x, *raw_args_no_reg))
    return (float(foe_x_opt), float(foe_y_opt), float(s_opt),
            float(oy_opt), raw_contrast, float(-result.fun))


def compute_ego_velocity(px, py, foe_x, foe_y, s, omega_y,
                         cx, cy, f_avg):
    x_bar = float(px) - cx
    y_bar = float(py) - cy
    vx = s * (float(px) - foe_x) - (f_avg + x_bar * x_bar / f_avg) * omega_y
    vy = s * (float(py) - foe_y) - (x_bar * y_bar / f_avg) * omega_y
    return float(vx), float(vy)


def grid_search_s_fixed_foe(events: np.ndarray, t_ref: float,
                             height: int, width: int,
                             foe_x: float, foe_y: float,
                             s_range: np.ndarray,
                             time_scale: float = 1e-6) -> Tuple[float, float]:
    if len(events) < 30:
        return 0.0, 0.0

    t = events["t"].astype(np.float64) * time_scale
    x = events["x"].astype(np.float64)
    y = events["y"].astype(np.float64)
    p = events["p"]
    dt = t - t_ref

    pos_mask = (p == 1)
    neg_mask = (p == 0)
    x_pos, y_pos, dt_pos = x[pos_mask], y[pos_mask], dt[pos_mask]
    x_neg, y_neg, dt_neg = x[neg_mask], y[neg_mask], dt[neg_mask]

    dx_pos = x_pos - foe_x
    dy_pos = y_pos - foe_y
    dx_neg = x_neg - foe_x
    dy_neg = y_neg - foe_y

    n_pix = height * width
    counts = np.zeros(n_pix, dtype=np.int32)

    best_s = 0.0
    best_sharp = -1.0

    s_dt_pos_all = np.outer(s_range, dt_pos)
    s_dt_neg_all = np.outer(s_range, dt_neg)

    xw_pos_all = np.floor(x_pos - s_dt_pos_all * dx_pos).astype(np.int32)
    yw_pos_all = np.floor(y_pos - s_dt_pos_all * dy_pos).astype(np.int32)
    xw_neg_all = np.floor(x_neg - s_dt_neg_all * dx_neg).astype(np.int32)
    yw_neg_all = np.floor(y_neg - s_dt_neg_all * dy_neg).astype(np.int32)

    valid_pos_all = ((xw_pos_all >= 0) & (xw_pos_all < width) &
                     (yw_pos_all >= 0) & (yw_pos_all < height))
    valid_neg_all = ((xw_neg_all >= 0) & (xw_neg_all < width) &
                     (yw_neg_all >= 0) & (yw_neg_all < height))

    for k in range(len(s_range)):
        v_p = valid_pos_all[k]
        v_n = valid_neg_all[k]
        idx_parts, pol_parts = [], []
        if v_p.any():
            idx_parts.append(yw_pos_all[k, v_p] * width + xw_pos_all[k, v_p])
            pol_parts.append(np.ones(int(v_p.sum()), dtype=np.int32))
        if v_n.any():
            idx_parts.append(yw_neg_all[k, v_n] * width + xw_neg_all[k, v_n])
            pol_parts.append(np.zeros(int(v_n.sum()), dtype=np.int32))
        if idx_parts:
            all_idx = np.concatenate(idx_parts)
            all_pol = np.concatenate(pol_parts)
            total = polarity_sharpness(all_idx, all_pol, n_pix, counts)
        else:
            total = 0.0

        if total > best_sharp:
            best_sharp = total
            best_s = float(s_range[k])

    return best_s, best_sharp


def grid_search_bg_foe_and_s(events: np.ndarray, t_ref: float,
                              height: int, width: int,
                              foe_x_vals: np.ndarray, foe_y_vals: np.ndarray,
                              s_vals: np.ndarray,
                              time_scale: float = 1e-6,
                              foe_scoring: str = "global",
                              n_cells: int = 4) -> Tuple[float, float, float, float]:
    if len(events) < 30:
        return float(width) / 2, float(height) / 2, 0.0, 0.0

    t = events["t"].astype(np.float64) * time_scale
    x = events["x"].astype(np.float64)
    y = events["y"].astype(np.float64)
    p = events["p"]
    dt = t - t_ref

    pos_mask = (p == 1)
    neg_mask = (p == 0)
    x_pos, y_pos, dt_pos = x[pos_mask], y[pos_mask], dt[pos_mask]
    x_neg, y_neg, dt_neg = x[neg_mask], y[neg_mask], dt[neg_mask]

    n_pix = height * width
    counts = np.zeros(n_pix, dtype=np.int32)

    use_cells = foe_scoring.startswith("cell_")
    use_voting = (foe_scoring == "cell_voting")
    cell_h = max(1, height // n_cells) if use_cells else height
    cell_w = max(1, width // n_cells) if use_cells else width
    n_cells_y = (height + cell_h - 1) // cell_h if use_cells else 1
    n_cells_x = (width + cell_w - 1) // cell_w if use_cells else 1
    total_cells = n_cells_y * n_cells_x

    if use_voting:
        cell_best_fx = np.full(total_cells, float(width) / 2, dtype=np.float64)
        cell_best_fy = np.full(total_cells, float(height) / 2, dtype=np.float64)
        cell_best_s = np.zeros(total_cells, dtype=np.float64)
        cell_best_sharp = np.full(total_cells, -1.0, dtype=np.float64)

    best_fx = float(width) / 2
    best_fy = float(height) / 2
    best_s = 0.0
    best_sharp = -1.0

    s_dt_pos_all = np.outer(s_vals, dt_pos)
    s_dt_neg_all = np.outer(s_vals, dt_neg)

    for fy_val in foe_y_vals:
        dy_pos = y_pos - fy_val
        dy_neg = y_neg - fy_val
        for fx_val in foe_x_vals:
            dx_pos = x_pos - fx_val
            dx_neg = x_neg - fx_val

            xw_pos = np.floor(x_pos - s_dt_pos_all * dx_pos).astype(np.int32)
            yw_pos = np.floor(y_pos - s_dt_pos_all * dy_pos).astype(np.int32)
            xw_neg = np.floor(x_neg - s_dt_neg_all * dx_neg).astype(np.int32)
            yw_neg = np.floor(y_neg - s_dt_neg_all * dy_neg).astype(np.int32)

            vp_all = ((xw_pos >= 0) & (xw_pos < width) &
                      (yw_pos >= 0) & (yw_pos < height))
            vn_all = ((xw_neg >= 0) & (xw_neg < width) &
                      (yw_neg >= 0) & (yw_neg < height))

            for k in range(len(s_vals)):
                if use_cells:
                    cell_contrasts = np.zeros(total_cells, dtype=np.float64)
                    cell_counts = np.zeros(total_cells, dtype=np.int64)

                    for vld, xw_arr, yw_arr in (
                        (vp_all[k], xw_pos[k], yw_pos[k]),
                        (vn_all[k], xw_neg[k], yw_neg[k]),
                    ):
                        n_v = int(vld.sum())
                        if n_v < 3:
                            continue
                        xw_v = xw_arr[vld]
                        yw_v = yw_arr[vld]
                        flat_idx = yw_v * width + xw_v
                        h = np.bincount(flat_idx, minlength=n_pix)
                        h2d = h.reshape(height, width)
                        for cy_i in range(n_cells_y):
                            r0 = cy_i * cell_h
                            r1 = min(r0 + cell_h, height)
                            for cx_i in range(n_cells_x):
                                ci = cy_i * n_cells_x + cx_i
                                c0 = cx_i * cell_w
                                c1 = min(c0 + cell_w, width)
                                cell_patch = h2d[r0:r1, c0:c1]
                                n_cell = int(cell_patch.sum())
                                if n_cell < 2:
                                    continue
                                cell_counts[ci] += n_cell
                                cell_contrasts[ci] += float(
                                    (cell_patch * cell_patch).sum()
                                ) / n_cell

                    if use_voting:
                        for ci in range(total_cells):
                            if cell_counts[ci] > 0 and cell_contrasts[ci] > cell_best_sharp[ci]:
                                cell_best_sharp[ci] = cell_contrasts[ci]
                                cell_best_fx[ci] = float(fx_val)
                                cell_best_fy[ci] = float(fy_val)
                                cell_best_s[ci] = float(s_vals[k])
                    else:
                        active = (cell_counts > 0)
                        if active.any():
                            total = _cell_score(cell_contrasts[active], foe_scoring)
                        else:
                            total = 0.0
                        if total > best_sharp:
                            best_sharp = total
                            best_fx = float(fx_val)
                            best_fy = float(fy_val)
                            best_s = float(s_vals[k])
                else:
                    v_p = vp_all[k]
                    n_vp = int(v_p.sum())
                    if n_vp > 0:
                        flat_p = yw_pos[k, v_p] * width + xw_pos[k, v_p]
                        hp = np.bincount(flat_p, minlength=n_pix)
                        sp = float((hp * hp).sum()) / n_vp
                    else:
                        sp = 0.0

                    v_n = vn_all[k]
                    n_vn = int(v_n.sum())
                    if n_vn > 0:
                        flat_n = yw_neg[k, v_n] * width + xw_neg[k, v_n]
                        hn = np.bincount(flat_n, minlength=n_pix)
                        sn = float((hn * hn).sum()) / n_vn
                    else:
                        sn = 0.0

                    total = sp + sn
                    if total > best_sharp:
                        best_sharp = total
                        best_fx = float(fx_val)
                        best_fy = float(fy_val)
                        best_s = float(s_vals[k])

    if use_voting:
        active = (cell_best_sharp > 0)
        n_active = int(active.sum())
        if n_active > 0:
            a_fx = cell_best_fx[active]
            a_fy = cell_best_fy[active]
            a_s = cell_best_s[active]
            a_w = cell_best_sharp[active]

            def _weighted_median(vals, weights):
                order = np.argsort(vals)
                v_sorted = vals[order]
                w_sorted = weights[order]
                cum = np.cumsum(w_sorted)
                half = cum[-1] * 0.5
                idx = np.searchsorted(cum, half)
                return float(v_sorted[min(idx, len(v_sorted) - 1)])

            best_fx = _weighted_median(a_fx, a_w)
            best_fy = _weighted_median(a_fy, a_w)
            best_s = _weighted_median(a_s, a_w)
            best_sharp = float(a_w.sum() / n_active)

    return best_fx, best_fy, best_s, best_sharp
