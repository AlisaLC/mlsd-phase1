"""The project's only quality metric."""

import numpy as np
import numpy.typing as npt

# Predictions are clipped to [EPS, 1 - EPS] so that a 0 or 1 gives a large but finite loss.
EPS = 1e-15


def nce(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    """Normalized cross-entropy: the log loss of p over the log loss of always predicting mean(y)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError(f"y and p must be non-empty 1-D arrays of the same length, got {y.shape} and {p.shape}")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be 0 or 1")
    if np.isnan(p).any():
        raise ValueError("predictions must not be NaN")
    p = np.clip(p, EPS, 1 - EPS)
    ctr = y.mean()
    if ctr <= 0 or ctr >= 1:
        raise ValueError("NCE is undefined when every label is the same")
    log_loss = -np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))
    base_log_loss = -(ctr * np.log(ctr) + (1 - ctr) * np.log1p(-ctr))
    return float(log_loss / base_log_loss)
