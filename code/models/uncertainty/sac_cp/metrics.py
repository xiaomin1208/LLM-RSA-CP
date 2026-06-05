import numpy as np


def interval_metrics(y, low, high, alpha):
    y = np.asarray(y, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    y, low, high = np.broadcast_arrays(y, low, high)
    valid = np.isfinite(y) & np.isfinite(low) & np.isfinite(high)
    covered = (y >= low) & (y <= high)
    width = high - low
    winkler = width + (2.0 / alpha) * (
        (low - y) * (y < low) + (y - high) * (y > high)
    )
    if not np.any(valid):
        return {
            "coverage": float("nan"),
            "dcov_abs_pct": float("nan"),
            "dcov_signed_pct": float("nan"),
            "width": float("nan"),
            "winkler": float("nan"),
        }
    coverage = float(np.mean(covered[valid]))
    return {
        "coverage": coverage,
        "dcov_signed_pct": float((coverage - (1.0 - alpha)) * 100.0),
        "dcov_abs_pct": float(abs((coverage - (1.0 - alpha)) * 100.0)),
        "width": float(np.mean(width[valid])),
        "winkler": float(np.mean(winkler[valid])),
    }


def higher_quantile(values, q):
    values = np.asarray(values, dtype=np.float64)
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0:
        return 0.0
    idx = int(np.ceil(q * values.size)) - 1
    idx = min(max(idx, 0), values.size - 1)
    return float(values[idx])


def split_cp_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 0.0
    level = np.ceil((scores.size + 1) * (1.0 - alpha)) / scores.size
    return higher_quantile(scores, min(level, 1.0))


def weighted_quantile(values, q, weights=None):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    values = values[valid]
    if values.size == 0:
        return 0.0
    if weights is None:
        return higher_quantile(values, q)
    weights = np.asarray(weights, dtype=np.float64)[valid]
    total = np.sum(weights)
    if total <= 0 or not np.isfinite(total):
        return higher_quantile(values, q)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order] / total
    cdf = np.cumsum(weights)
    return float(values[min(np.searchsorted(cdf, q, side="left"), values.size - 1)])
