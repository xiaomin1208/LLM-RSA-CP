import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_rescp_table6 import TARGETS


DATASET_MAP = {
    "solar": "Solar",
    "beijing": "Beijing",
    "exchange": "Exchange",
    "elec": "ACEA",
}

MODEL_MAP = {
    "rnn": "RNN",
    "transformer": "Transf",
    "arima": "ARIMA",
}


def weighted_quantile(values, quantile, weights=None):
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        return np.quantile(values, quantile, method="higher")
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    weights = weights / np.sum(weights)
    cdf = np.cumsum(weights)
    return values[min(np.searchsorted(cdf, quantile, side="left"), len(values) - 1)]


def interval_metrics(residuals, low, high, alpha):
    covered = (residuals >= low) & (residuals <= high)
    width = high - low
    winkler = width + (2.0 / alpha) * ((low - residuals) * (residuals < low) + (residuals - high) * (residuals > high))
    return {
        "coverage": float(np.nanmean(covered)),
        "dcov_signed_pct": float((np.nanmean(covered) - (1.0 - alpha)) * 100.0),
        "dcov_abs_pct": float(abs((np.nanmean(covered) - (1.0 - alpha)) * 100.0)),
        "width": float(np.nanmean(width)),
        "winkler": float(np.nanmean(winkler)),
    }


def split_residuals(residual_path):
    residual_path = Path(residual_path)
    residuals = pd.read_hdf(residual_path, key="target")
    mask = pd.read_hdf(residual_path, key="target_mask").astype(bool)
    indices = np.load(residual_path.parent / "indices.npz")
    valid_target_indices = indices["valid_target_indices"]
    pos_by_abs = {int(v): i for i, v in enumerate(valid_target_indices)}
    calib_pos = [pos_by_abs[int(i)] for i in indices["calib_indices"] if int(i) in pos_by_abs]
    test_pos = [pos_by_abs[int(i)] for i in indices["test_indices"] if int(i) in pos_by_abs]
    cal = residuals.iloc[calib_pos].to_numpy(dtype=np.float64)
    test = residuals.iloc[test_pos].to_numpy(dtype=np.float64)
    test_mask = mask.iloc[test_pos].to_numpy(dtype=bool)
    return cal, test, test_mask


def scan_one(residual_path, alpha, modes):
    cal, test, test_mask = split_residuals(residual_path)
    rows = []
    # Flattened CP is the quickest fair baseline: it uses calibration residuals only.
    cal_flat = cal[np.isfinite(cal)]
    test_eval = np.where(test_mask, test, np.nan)
    if "global" in modes:
        low = weighted_quantile(cal_flat, alpha / 2.0)
        high = weighted_quantile(cal_flat, 1.0 - alpha / 2.0)
        rows.append({"mode": "global", **interval_metrics(test_eval, low, high, alpha)})
    if "per_node" in modes:
        lows = []
        highs = []
        for j in range(cal.shape[1]):
            vals = cal[:, j]
            vals = vals[np.isfinite(vals)]
            lows.append(weighted_quantile(vals, alpha / 2.0))
            highs.append(weighted_quantile(vals, 1.0 - alpha / 2.0))
        low = np.asarray(lows)[None, :]
        high = np.asarray(highs)[None, :]
        rows.append({"mode": "per_node", **interval_metrics(test_eval, low, high, alpha)})
    if "recent_per_node" in modes:
        for window in (250, 500, 1000, 2000, 4000):
            lows = []
            highs = []
            for j in range(cal.shape[1]):
                vals = cal[-min(window, cal.shape[0]):, j]
                vals = vals[np.isfinite(vals)]
                lows.append(weighted_quantile(vals, alpha / 2.0))
                highs.append(weighted_quantile(vals, 1.0 - alpha / 2.0))
            low = np.asarray(lows)[None, :]
            high = np.asarray(highs)[None, :]
            rows.append({"mode": f"recent_per_node_{window}", **interval_metrics(test_eval, low, high, alpha)})
    if "exp_per_node" in modes:
        for decay in (0.995, 0.9975, 0.999, 0.9995):
            n = cal.shape[0]
            weights = decay ** np.arange(n - 1, -1, -1)
            lows = []
            highs = []
            for j in range(cal.shape[1]):
                vals = cal[:, j]
                valid = np.isfinite(vals)
                lows.append(weighted_quantile(vals[valid], alpha / 2.0, weights[valid]))
                highs.append(weighted_quantile(vals[valid], 1.0 - alpha / 2.0, weights[valid]))
            low = np.asarray(lows)[None, :]
            high = np.asarray(highs)[None, :]
            rows.append({"mode": f"exp_per_node_{decay}", **interval_metrics(test_eval, low, high, alpha)})
    return rows


def infer_keys(path):
    parts = [p.lower() for p in Path(path).parts]
    dataset = next((DATASET_MAP[p] for p in parts if p in DATASET_MAP), None)
    model = next((MODEL_MAP[p] for p in parts if p in MODEL_MAP), None)
    return dataset, model


def main():
    parser = argparse.ArgumentParser(description="Quick scan CP intervals on official ResCP residuals.h5 files.")
    parser.add_argument("--root", default="tmp_rescp_official/extract/reservoir-conformal-prediction-dev-main/reservoir_conformal_prediction/logs/base")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--modes", default="global,per_node,recent_per_node,exp_per_node")
    args = parser.parse_args()
    root = Path(args.root)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    out = []
    for residual_path in sorted(root.glob("*/*/*/*/residuals.h5")):
        dataset, model = infer_keys(residual_path)
        if dataset is None or model is None:
            continue
        target = TARGETS.get((dataset, model))
        rows = scan_one(residual_path, args.alpha, modes)
        best = min(rows, key=lambda r: (r["winkler"], r["width"], r["dcov_abs_pct"]))
        rec = {
            "dataset": dataset,
            "model": model,
            "path": str(residual_path),
            "target": target,
            "best": best,
            "all_modes": rows,
        }
        if target:
            rec["pass_all"] = (
                best["dcov_abs_pct"] < target["dcov_abs"]
                and best["width"] < target["width"]
                and best["winkler"] < target["winkler"]
            )
        out.append(rec)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
