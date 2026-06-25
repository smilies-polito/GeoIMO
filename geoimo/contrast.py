import numpy as np
from typing import Optional


def polarity_sharpness(indices: np.ndarray, polarities: np.ndarray,
                       n_pix: int,
                       counts_buf: Optional[np.ndarray] = None) -> float:
    total = 0.0
    for pol_val in (1, 0):
        mask = (polarities == pol_val)
        n_m = int(mask.sum())
        if n_m < 3:
            continue
        h = np.bincount(indices[mask], minlength=n_pix)
        total += float((h * h).sum()) / n_m
    return total


FOE_SCORING_MODES = (
    "global", "cell_uniform", "cell_sharpness",
    "cell_median", "cell_trimmed", "cell_voting",
)


def _cell_score(cell_contrasts: np.ndarray, mode: str) -> float:
    n = len(cell_contrasts)
    if n == 0:
        return 0.0

    if mode == "cell_uniform":
        return float(cell_contrasts.mean())

    elif mode == "cell_sharpness":
        weights = np.sqrt(np.maximum(cell_contrasts, 0.0))
        w_sum = weights.sum()
        if w_sum < 1e-12:
            return 0.0
        return float((weights * cell_contrasts).sum() / w_sum)

    elif mode == "cell_median":
        return float(np.median(cell_contrasts))

    elif mode == "cell_trimmed":
        if n < 5:
            return float(cell_contrasts.mean())
        k = max(1, int(round(0.2 * n)))
        sorted_c = np.sort(cell_contrasts)
        return float(sorted_c[k:-k].mean())

    else:
        return float(cell_contrasts.sum())
