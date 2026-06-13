import argparse
import copy
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_rescp_table6 import get_targets  # noqa: E402
from models.uncertainty.sac_cp.metrics import higher_quantile, interval_metrics, split_cp_quantile  # noqa: E402
from run_sac_cp_timeseries_experiments import CELLS, find_residual_paths, split_residuals_and_features  # noqa: E402


SMOKE_CELLS = [("Exchange", "RNN"), ("ACEA", "RNN"), ("Solar", "RNN")]


def cell_label(cell):
    return f"{cell[0]}/{cell[1]}"


def method_record(name, q, metrics):
    return {
        "name": name,
        "threshold": float(q),
        "coverage": metrics["coverage"],
        "delta_cov": metrics["dcov_signed_pct"],
        "delta_cov_abs": metrics["dcov_abs_pct"],
        "width": metrics["width"],
        "winkler": metrics["winkler"],
    }


def lag(arr, steps):
    out = np.zeros_like(arr, dtype=np.float64)
    if steps < arr.shape[0]:
        out[steps:] = arr[:-steps]
    return out


def rolling_mean_past(arr, window):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        prefix = np.zeros((1,), dtype=np.float64)
    else:
        prefix = np.zeros((1, arr.shape[1]), dtype=np.float64)
    c = np.cumsum(np.concatenate([prefix, arr], axis=0), axis=0)
    out = np.zeros_like(arr, dtype=np.float64)
    for i in range(arr.shape[0]):
        start = max(0, i - window)
        denom = max(i - start, 1)
        out[i] = (c[i] - c[start]) / denom
    return out


def rolling_std_past(arr, window):
    mean = rolling_mean_past(arr, window)
    sq = rolling_mean_past(np.asarray(arr, dtype=np.float64) ** 2, window)
    return np.sqrt(np.maximum(sq - mean * mean, 0.0))


DEFAULT_FEATURE_TEMPLATE = {
    "name": "base_prequential",
    "row_lags": [1, 2, 4],
    "row_windows": [8, 24, 48, 168],
    "periods": [24.0, 168.0, "global"],
    "node_lags": [1, 2, 4, 24],
    "node_windows": [8, 24, 48],
    "include_row_iqr": True,
    "include_node_id": True,
}


def sanitize_feature_template(raw):
    allowed = {
        "name", "row_lags", "row_windows", "periods", "node_lags", "node_windows",
        "include_row_iqr", "include_node_id", "intended_failure_mode",
    }
    if not isinstance(raw, dict):
        raise ValueError("template must be a dict")
    forbidden = {"code", "python", "threshold", "selector", "uses_ref_label", "uses_adj_label", "uses_test_label"}
    if any(k in raw for k in forbidden):
        raise ValueError(f"forbidden template field in {raw.get('name', '<unnamed>')}")
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown template fields: {sorted(unknown)}")
    out = dict(DEFAULT_FEATURE_TEMPLATE)
    out.update(raw)
    name = str(out.get("name", "llm_template")).lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name).strip("_")[:48] or "llm_template"
    out["name"] = name

    def ints(key, lo, hi, max_len):
        vals = out.get(key, [])
        if not isinstance(vals, list):
            raise ValueError(f"{key} must be a list")
        clean = []
        for v in vals[:max_len]:
            iv = int(v)
            if lo <= iv <= hi and iv not in clean:
                clean.append(iv)
        if not clean:
            clean = DEFAULT_FEATURE_TEMPLATE[key]
        out[key] = clean

    ints("row_lags", 1, 336, 8)
    ints("row_windows", 2, 720, 8)
    ints("node_lags", 1, 336, 8)
    ints("node_windows", 2, 720, 8)
    periods = []
    for p in out.get("periods", [])[:6]:
        if isinstance(p, str) and p.lower() == "global":
            periods.append("global")
        else:
            fp = float(p)
            if 2.0 <= fp <= 10000.0:
                periods.append(fp)
    out["periods"] = periods or DEFAULT_FEATURE_TEMPLATE["periods"]
    out["include_row_iqr"] = bool(out.get("include_row_iqr", True))
    out["include_node_id"] = bool(out.get("include_node_id", True))
    return out


def make_prequential_features(y, template=None):
    template = sanitize_feature_template(template or DEFAULT_FEATURE_TEMPLATE)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    n, d = y.shape
    row_med = np.median(y, axis=1)
    row_mean = np.mean(y, axis=1)
    row_abs = np.mean(np.abs(y), axis=1)
    row_iqr = np.percentile(y, 75, axis=1) - np.percentile(y, 25, axis=1)
    row_features = [
        lag(row_mean, 1),
        lag(row_abs, 1),
    ]
    for s in template["row_lags"]:
        row_features.append(lag(row_med, s))
    if template["include_row_iqr"]:
        row_features.append(lag(row_iqr, 1))
    for w in template["row_windows"]:
        row_features.extend([
            rolling_mean_past(row_med, w),
            rolling_std_past(row_med, w),
            rolling_mean_past(row_abs, w),
        ])
    t = np.arange(n, dtype=np.float64)
    for period in template["periods"]:
        period = max(float(n), 1.0) if period == "global" else float(period)
        row_features.extend([
            np.sin(2.0 * np.pi * t / period),
            np.cos(2.0 * np.pi * t / period),
        ])
    row_mat = np.column_stack(row_features)

    flat_parts = [np.repeat(row_mat, d, axis=0)]
    if template["include_node_id"]:
        node = np.tile(np.arange(d, dtype=np.float64), n)
        node_scaled = node / max(d - 1, 1)
        flat_parts.append(node_scaled[:, None])
        flat_parts.append(np.sin(2.0 * np.pi * node_scaled)[:, None])
        flat_parts.append(np.cos(2.0 * np.pi * node_scaled)[:, None])
    for s in template["node_lags"]:
        flat_parts.append(lag(y, s).reshape(-1, 1))
    for w in template["node_windows"]:
        flat_parts.append(rolling_mean_past(np.abs(y), w).reshape(-1, 1))
        flat_parts.append(rolling_std_past(y, w).reshape(-1, 1))
    X = np.hstack(flat_parts)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32)


def split_flat_views(X, y, split_sizes):
    if y.ndim == 1:
        y = y[:, None]
    d = y.shape[1]
    n_fit = split_sizes["fit"]
    n_sel = split_sizes["sel"]
    n_ref = split_sizes["ref"]
    n_adj = split_sizes["adj"]
    i0 = n_fit
    i1 = i0 + n_sel
    i2 = i1 + n_ref
    i3 = i2 + n_adj

    def rows(a, b):
        return slice(a * d, b * d)

    return {
        "fit": X[rows(0, i0)],
        "sel": X[rows(i0, i1)],
        "ref": X[rows(i1, i2)],
        "adj": X[rows(i2, i3)],
        "test": X[rows(i3, i3 + split_sizes["test"])],
    }


def normalize_train(X_train, views):
    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    out = {}
    for key, X in views.items():
        out[key] = ((X - mean) / scale).astype(np.float32)
    return out


def residual_diagnostics(y, alpha):
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    flat = arr[np.isfinite(arr)]
    row_med = np.nanmedian(arr, axis=1)
    diff = np.diff(row_med) if row_med.size > 1 else np.array([0.0])
    if flat.size == 0:
        return {}
    q05, q50, q95 = np.quantile(flat, [0.05, 0.5, 0.95])
    abs_flat = np.abs(flat)
    return {
        "alpha": float(alpha),
        "n_rows": int(arr.shape[0]),
        "n_nodes": int(arr.shape[1]),
        "median": float(q50),
        "std": float(np.std(flat)),
        "abs_mean": float(np.mean(abs_flat)),
        "abs_q90": float(np.quantile(abs_flat, 0.9)),
        "q05": float(q05),
        "q95": float(q95),
        "skew_proxy": float((q95 + q05 - 2.0 * q50) / max(np.std(flat), 1e-8)),
        "row_diff_std": float(np.std(diff)),
        "recent_abs_ratio": float(
            np.mean(np.abs(row_med[-max(5, row_med.size // 10):])) / max(np.mean(np.abs(row_med)), 1e-8)
        ),
    }


def neutral_diagnostics(alpha):
    return {
        "alpha": float(alpha),
        "n_rows": 1000,
        "n_nodes": 5,
        "median": 0.0,
        "std": 1.0,
        "abs_mean": 0.5,
        "abs_q90": 1.2,
        "q05": -1.0,
        "q95": 1.0,
        "skew_proxy": 0.0,
        "row_diff_std": 1.0,
        "recent_abs_ratio": 1.0,
    }


def random_diagnostics(alpha, seed):
    rng = np.random.default_rng(int(seed))
    std = float(10 ** rng.uniform(-3.0, 2.0))
    abs_mean = float(std * rng.uniform(0.2, 1.2))
    abs_q90 = float(std * rng.uniform(1.0, 3.5))
    skew = float(rng.uniform(-1.0, 1.0))
    recent = float(rng.uniform(0.5, 2.5))
    nodes = int(rng.choice([1, 3, 5, 10, 50]))
    return {
        "alpha": float(alpha),
        "n_rows": int(rng.integers(500, 5000)),
        "n_nodes": nodes,
        "median": float(rng.normal(0.0, std * 0.1)),
        "std": std,
        "abs_mean": abs_mean,
        "abs_q90": abs_q90,
        "q05": float(-abs_q90 * rng.uniform(0.7, 1.3)),
        "q95": float(abs_q90 * rng.uniform(0.7, 1.3)),
        "skew_proxy": skew,
        "row_diff_std": float(std * rng.uniform(0.1, 1.5)),
        "recent_abs_ratio": recent,
    }


def perturb_diagnostics(args, cell, diagnostics):
    mode = str(getattr(args, "diagnostic_mode", "true"))
    out = copy.deepcopy(diagnostics)
    if mode == "true":
        source = "true"
    elif mode == "neutral":
        out = neutral_diagnostics(args.alpha)
        source = "neutral"
    elif mode == "random":
        out = random_diagnostics(args.alpha, stable_cell_seed(cell, args.seed) + 99173)
        source = "random"
    elif mode == "shuffle_keys":
        rng = np.random.default_rng(stable_cell_seed(cell, args.seed) + 27183)
        keys = ["std", "abs_mean", "abs_q90", "skew_proxy", "row_diff_std", "recent_abs_ratio"]
        vals = [out[k] for k in keys if k in out]
        rng.shuffle(vals)
        for k, v in zip([k for k in keys if k in out], vals):
            out[k] = float(v)
        source = "shuffle_keys"
    else:
        raise ValueError(f"unknown diagnostic_mode: {mode}")
    out["_diagnostic_mode"] = mode
    out["_diagnostic_source"] = source
    return out


def heuristic_llm_templates(diagnostics):
    n_nodes = int(diagnostics.get("n_nodes", 1))
    recent_ratio = float(diagnostics.get("recent_abs_ratio", 1.0))
    if n_nodes > 20:
        return [
            {
                "name": "llm_cross_node_seasonal",
                "row_lags": [1, 2, 24, 48, 168],
                "row_windows": [12, 24, 72, 168, 336],
                "periods": [24, 168, "global"],
                "node_lags": [1, 24, 48, 168],
                "node_windows": [12, 24, 168],
                "include_row_iqr": True,
                "include_node_id": True,
                "intended_failure_mode": "cross-node seasonal residual structure",
            },
            {
                "name": "llm_tail_volatility",
                "row_lags": [1, 2, 4, 8],
                "row_windows": [8, 16, 32, 64],
                "periods": [24, 168],
                "node_lags": [1, 2, 4, 8, 24],
                "node_windows": [8, 16, 32, 64],
                "include_row_iqr": True,
                "include_node_id": True,
                "intended_failure_mode": "short-horizon volatility bursts",
            },
        ]
    if recent_ratio > 1.2:
        return [
            {
                "name": "llm_recent_drift",
                "row_lags": [1, 2, 3, 6, 12, 24],
                "row_windows": [4, 8, 16, 32, 64],
                "periods": [24, 168],
                "node_lags": [1, 2, 3, 6, 12, 24],
                "node_windows": [4, 8, 16, 32],
                "include_row_iqr": True,
                "include_node_id": True,
                "intended_failure_mode": "recent residual drift",
            }
        ]
    return [
        {
            "name": "llm_multiscale_memory",
            "row_lags": [1, 2, 4, 12, 24, 48],
            "row_windows": [8, 24, 72, 168, 336],
            "periods": [24, 168, "global"],
            "node_lags": [1, 2, 4, 12, 24, 48],
            "node_windows": [8, 24, 72, 168],
            "include_row_iqr": True,
            "include_node_id": True,
            "intended_failure_mode": "multi-scale residual memory",
        }
    ]


FEATURE_BANKS = {
    "recent_drift": {
        "name": "bank_recent_drift",
        "row_lags": [1, 2, 3, 6, 12, 24, 48],
        "row_windows": [4, 8, 16, 32, 64, 128],
        "periods": [24, 168, "global"],
        "node_lags": [1, 2, 3, 6, 12, 24, 48],
        "node_windows": [4, 8, 16, 32, 64],
        "include_row_iqr": True,
        "include_node_id": True,
        "intended_failure_mode": "recent residual drift and local volatility",
    },
    "seasonal_cross_node": {
        "name": "bank_seasonal_cross_node",
        "row_lags": [1, 2, 24, 48, 168],
        "row_windows": [12, 24, 48, 72, 168, 336],
        "periods": [24, 168, 336, "global"],
        "node_lags": [1, 24, 48, 168],
        "node_windows": [12, 24, 48, 168, 336],
        "include_row_iqr": True,
        "include_node_id": True,
        "intended_failure_mode": "seasonal cross-node residual dependence",
    },
    "micro_scale_exchange": {
        "name": "bank_micro_scale_exchange",
        "row_lags": [1, 2, 3, 4, 5, 8, 12],
        "row_windows": [4, 8, 12, 16, 24, 48],
        "periods": [7, 14, 24, "global"],
        "node_lags": [1, 2, 3, 4, 5, 8, 12],
        "node_windows": [4, 8, 12, 16, 24, 48],
        "include_row_iqr": True,
        "include_node_id": True,
        "intended_failure_mode": "tiny-scale exchange-rate residual regimes",
    },
    "tail_volatility": {
        "name": "bank_tail_volatility",
        "row_lags": [1, 2, 4, 8, 16, 24],
        "row_windows": [4, 8, 16, 32, 64, 96],
        "periods": [24, 168],
        "node_lags": [1, 2, 4, 8, 16, 24],
        "node_windows": [4, 8, 16, 32, 64, 96],
        "include_row_iqr": True,
        "include_node_id": True,
        "intended_failure_mode": "heavy tails and volatility bursts",
    },
}


LLM_CANDIDATE_PROFILES = {
    "base_shrink": {
        "description": "current strong deterministic baseline: broad beta grid with shrink/expand scales",
        "betas": [0.001, 0.025, 0.050, 0.075, 0.099],
        "shrink_scales": [0.6, 0.8, 1.0, 1.2],
        "include_global": False,
        "include_recent": False,
        "include_union": False,
    },
    "compact_center": {
        "description": "small candidate set around the historically robust beta=0.05 center; reduces post-selection cost",
        "betas": [0.025, 0.050, 0.075],
        "shrink_scales": [0.8, 1.0, 1.2],
        "include_global": False,
        "include_recent": False,
        "include_union": False,
    },
    "micro_scale": {
        "description": "tiny-scale smooth residuals, usually exchange-rate cells; avoids overly wide high-beta candidates",
        "betas": [0.025, 0.050, 0.075],
        "shrink_scales": [0.6, 0.8, 1.0, 1.2],
        "include_global": False,
        "include_recent": False,
        "include_union": False,
    },
    "micro_regularized": {
        "description": "tiny-scale financial residuals with smooth LightGBM quantiles; useful when micro_scale overfits and final intervals are too wide",
        "betas": [0.025, 0.050, 0.075],
        "shrink_scales": [0.6, 0.8, 1.0, 1.2],
        "include_global": False,
        "include_recent": False,
        "include_union": False,
        "lgbm": {
            "n_estimators": 260,
            "learning_rate": 0.025,
            "num_leaves": 7,
            "min_child_samples": 150,
            "reg_lambda": 5.0,
        },
    },
    "seasonal_compact": {
        "description": "strong seasonal residuals with over-coverage risk; adds near-unit shrink scales for tighter but still coverage-aware candidates",
        "betas": [0.025, 0.050, 0.075],
        "shrink_scales": [0.85, 0.90, 0.95, 1.0, 1.05, 1.10],
        "include_global": False,
        "include_recent": False,
        "include_union": False,
    },
    "tail_safe": {
        "description": "heavy tails or skew; keeps upper beta and expansion candidates for coverage safety",
        "betas": [0.050, 0.075, 0.099],
        "shrink_scales": [0.8, 1.0, 1.2, 1.4],
        "include_global": False,
        "include_recent": False,
        "include_union": False,
    },
    "recent_drift": {
        "description": "recent residual magnitude shift; adds recent empirical signed-quantile candidates",
        "betas": [0.025, 0.050, 0.075],
        "shrink_scales": [0.8, 1.0, 1.2],
        "include_global": False,
        "include_recent": True,
        "include_union": False,
    },
    "recent_union": {
        "description": "aggressive drift profile; unions base and recent intervals, usually conservative",
        "betas": [0.025, 0.050, 0.075],
        "shrink_scales": [0.8, 1.0, 1.2],
        "include_global": False,
        "include_recent": True,
        "include_union": True,
    },
}


_QWEN_GENERATOR = None


def parse_bank_response(text):
    low = str(text).lower()
    selected = []
    for name in FEATURE_BANKS:
        if name in low:
            selected.append(name)
    if not selected:
        try:
            obj = json.loads(re.search(r"\[[\s\S]*\]", str(text)).group(0))
            for item in obj:
                if str(item).lower() in FEATURE_BANKS:
                    selected.append(str(item).lower())
        except Exception:
            pass
    return selected


def parse_profile_response(text):
    low = str(text).lower()
    selected = []
    for name in LLM_CANDIDATE_PROFILES:
        if name in low:
            selected.append(name)
    if selected:
        return selected[0]
    try:
        obj = json.loads(re.search(r"\{[\s\S]*\}", str(text)).group(0))
        profile = str(obj.get("profile", "")).lower()
        if profile in LLM_CANDIDATE_PROFILES:
            return profile
    except Exception:
        pass
    try:
        obj = json.loads(re.search(r"\[[\s\S]*\]", str(text)).group(0))
        if obj:
            profile = str(obj[0]).lower()
            if profile in LLM_CANDIDATE_PROFILES:
                return profile
    except Exception:
        pass
    raise ValueError(f"no valid candidate profile in LLM response: {text!r}")


def heuristic_candidate_profile(diagnostics):
    std = float(diagnostics.get("std", 1.0))
    abs_mean = float(diagnostics.get("abs_mean", 1.0))
    abs_q90 = float(diagnostics.get("abs_q90", 1.0))
    skew = abs(float(diagnostics.get("skew_proxy", 0.0)))
    recent_ratio = float(diagnostics.get("recent_abs_ratio", 1.0))
    n_nodes = int(diagnostics.get("n_nodes", 1))
    tail_ratio = abs_q90 / max(std, 1e-8)
    if std < 0.05 and abs_mean < 0.05 and recent_ratio > 1.5:
        return "micro_regularized"
    if std < 0.05 and abs_mean < 0.05:
        return "micro_scale"
    if recent_ratio > 1.25:
        return "recent_drift"
    if skew > 0.4 or tail_ratio > 2.2:
        return "tail_safe"
    if n_nodes <= 3:
        return "compact_center"
    return "base_shrink"


def validate_candidate_profile(proposed, diagnostics):
    """Keep the LLM profile inside obvious numeric guardrails."""
    std = float(diagnostics.get("std", 1.0))
    abs_mean = float(diagnostics.get("abs_mean", 1.0))
    recent_ratio = float(diagnostics.get("recent_abs_ratio", 1.0))
    n_nodes = int(diagnostics.get("n_nodes", 1))
    abs_q90 = float(diagnostics.get("abs_q90", 1.0))
    tail_ratio = abs_q90 / max(std, 1e-8)
    if std < 0.05 and abs_mean < 0.05:
        if recent_ratio > 1.5:
            return "micro_regularized", "numeric_guard_micro_regularized"
        return "micro_scale", "numeric_guard_micro_scale"
    if proposed == "recent_union" and recent_ratio < 1.5:
        return "recent_drift", "numeric_guard_avoid_union_without_strong_drift"
    if proposed == "tail_safe" and tail_ratio < 1.8:
        return "compact_center" if n_nodes <= 3 else "base_shrink", "numeric_guard_weak_tail"
    return proposed, None


def merge_profile_specs(profile_names):
    names = [name for name in profile_names if name in LLM_CANDIDATE_PROFILES]
    if not names:
        names = ["base_shrink"]
    specs = [LLM_CANDIDATE_PROFILES[name] for name in names]
    merged = {
        "description": "stability-audited union of residual profiles: " + ",".join(names),
        "betas": sorted({float(x) for spec in specs for x in spec["betas"]}),
        "shrink_scales": sorted({float(x) for spec in specs for x in spec["shrink_scales"]}),
        "include_global": any(bool(spec.get("include_global")) for spec in specs),
        "include_recent": any(bool(spec.get("include_recent")) for spec in specs),
        "include_union": any(bool(spec.get("include_union")) for spec in specs),
        "profile_names": names,
    }
    if specs[0].get("lgbm"):
        merged["lgbm"] = copy.deepcopy(specs[0]["lgbm"])
    return merged


def profile_spec(name):
    spec = copy.deepcopy(LLM_CANDIDATE_PROFILES[name])
    spec["profile_names"] = [name]
    return spec


def build_llm_profile_prompt(cell, diagnostics, variant=0):
    profile_desc = {
        name: {
            "description": spec["description"],
            "betas": spec["betas"],
            "shrink_scales": spec["shrink_scales"],
            "include_global": spec["include_global"],
            "include_recent": spec["include_recent"],
            "include_union": spec["include_union"],
            "lgbm": spec.get("lgbm"),
        }
        for name, spec in LLM_CANDIDATE_PROFILES.items()
    }
    variants = [
        (
            "Decision hints:\n"
            "- tiny std and abs_mean: prefer micro_scale even if skew_proxy is high, because tiny denominators can make skew misleading.\n"
            "- tiny std/abs_mean with very high recent_abs_ratio: prefer micro_regularized to smooth overfit micro-scale quantiles.\n"
            "- high recent_abs_ratio: prefer recent_drift.\n"
            "- high skew_proxy or abs_q90/std: prefer tail_safe.\n"
            "- small n_nodes or smooth residuals: prefer compact_center.\n"
            "- large multi-node seasonal data with no special warning: prefer base_shrink.\n"
        ),
        (
            "Decision hints:\n"
            "- Treat the task as residual-regime routing, not forecasting.\n"
            "- Prefer efficient profiles for stable residuals, but keep tail_safe for clearly skewed/heavy-tailed residuals.\n"
            "- Use micro_regularized for tiny financial residuals when recent_abs_ratio is elevated.\n"
            "- Use recent_drift only when recent magnitude shift is the dominant warning.\n"
        ),
        (
            "Decision hints:\n"
            "- Choose the profile that would give a stable conformal candidate family on D_fit diagnostics alone.\n"
            "- If residual scale is tiny, avoid broad high-variance profiles; use micro_scale or micro_regularized.\n"
            "- If residual tails dominate scale diagnostics, use tail_safe.\n"
            "- If there is no clear regime warning, prefer base_shrink or compact_center.\n"
        ),
    ]
    hint = variants[int(variant) % len(variants)]
    return (
        "Choose one candidate-pool profile for time-series conformal prediction.\n"
        "You may only output compact JSON with keys \"profile\" and \"reason\".\n"
        "Do not output Python code. Do not set final thresholds. Do not choose final intervals.\n"
        "Use only the D_fit residual diagnostics below. The final candidate selection and coverage calibration "
        "will be done by SAC-CP on held-out calibration blocks.\n\n"
        f"Allowed profiles: {json.dumps(profile_desc)}\n\n"
        f"{hint}\n"
        f"Cell metadata: {cell}.\n"
        f"D_fit diagnostics only: {json.dumps(diagnostics, ensure_ascii=False)}\n"
    )


def block_bootstrap_diagnostics(fit_y, alpha, seed, block_size=24):
    arr = np.asarray(fit_y, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    n = arr.shape[0]
    if n <= 1:
        return residual_diagnostics(arr, alpha)
    rng = np.random.default_rng(int(seed))
    block = max(1, min(int(block_size), n))
    pieces = []
    total = 0
    while total < n:
        start = int(rng.integers(0, max(1, n - block + 1)))
        part = arr[start:start + block]
        pieces.append(part)
        total += part.shape[0]
    boot = np.concatenate(pieces, axis=0)[:n]
    return residual_diagnostics(boot, alpha)


def stable_cell_seed(cell, seed):
    raw = f"{cell_label(cell)}::{seed}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def parse_json_templates(text):
    text = str(text).strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\[[\s\S]*\]", text)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        raise ValueError("LLM output is not a JSON list")
    return [sanitize_feature_template(x) for x in obj]


def _get_qwen_generator(model_path, device):
    global _QWEN_GENERATOR
    cache_key = (str(model_path), str(device))
    if _QWEN_GENERATOR is not None and _QWEN_GENERATOR.get("key") == cache_key:
        return _QWEN_GENERATOR
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto").to(device)
    model.eval()
    _QWEN_GENERATOR = {"key": cache_key, "tokenizer": tokenizer, "model": model}
    return _QWEN_GENERATOR


def unload_qwen_generator():
    global _QWEN_GENERATOR
    if _QWEN_GENERATOR is None:
        return
    import torch
    _QWEN_GENERATOR = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def call_qwen_json(prompt, model_path, device, max_new_tokens, keep_loaded):
    import torch

    handle = _get_qwen_generator(model_path, device)
    tokenizer = handle["tokenizer"]
    model = handle["model"]
    messages = [
        {"role": "system", "content": "You output only valid JSON. No prose."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0, inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    if not keep_loaded:
        unload_qwen_generator()
    return raw


def call_qwen_feature_templates(prompt, model_path, device, max_new_tokens):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto").to(device)
    model.eval()
    messages = [
        {"role": "system", "content": "You output only valid JSON. No prose."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0, inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    del model
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw


def _llm_select_profile(args, cell, diagnostics, variant=0):
    prompt = build_llm_profile_prompt(cell_label(cell), diagnostics, variant=variant)
    raw = call_qwen_json(
        prompt,
        args.llm_model,
        args.llm_device,
        args.llm_max_new_tokens,
        args.keep_llm_loaded,
    )
    raw_profile = parse_profile_response(raw)
    profile, override_reason = validate_candidate_profile(raw_profile, diagnostics)
    return {
        "profile": profile,
        "raw_profile": raw_profile,
        "profile_override_reason": override_reason,
        "raw_response": raw,
        "prompt": prompt,
        "diagnostics": diagnostics,
    }


def stability_audited_profile(args, cell, fit_y, diagnostics):
    votes = []
    errors = []
    base_seed = stable_cell_seed(cell, args.seed)
    n_prompt_variants = max(1, int(args.router_prompt_variants))
    n_bootstrap = max(0, int(args.router_bootstrap_samples))
    for variant in range(n_prompt_variants):
        try:
            votes.append(_llm_select_profile(args, cell, diagnostics, variant=variant))
        except Exception as exc:
            errors.append({"variant": int(variant), "bootstrap": None, "error": repr(exc)})
            if not args.allow_llm_fallback:
                raise
    for b in range(n_bootstrap):
        boot_diag = block_bootstrap_diagnostics(fit_y, args.alpha, base_seed + b, args.router_bootstrap_block)
        variant = b % n_prompt_variants
        try:
            votes.append(_llm_select_profile(args, cell, boot_diag, variant=variant))
        except Exception as exc:
            errors.append({"variant": int(variant), "bootstrap": int(b), "error": repr(exc)})
            if not args.allow_llm_fallback:
                raise
    if not votes:
        fallback = heuristic_candidate_profile(diagnostics)
        return [fallback], {
            "used_llm": False,
            "profile": fallback,
            "raw_profile": fallback,
            "profile_override_reason": "stability_audit_all_votes_failed",
            "stability": 0.0,
            "vote_counts": {},
            "votes": [],
            "errors": errors,
        }
    counts = Counter(v["profile"] for v in votes)
    total = sum(counts.values())
    ranked = counts.most_common()
    stability = ranked[0][1] / max(total, 1)
    entropy = -sum((c / total) * np.log(max(c / total, 1e-12)) for _, c in ranked) if total else 0.0
    if stability >= float(args.router_stability_threshold):
        chosen = [ranked[0][0]]
        action = "top1"
    elif stability >= float(args.router_top2_threshold) and len(ranked) > 1:
        chosen = [ranked[0][0], ranked[1][0]]
        action = "top2_union"
    else:
        chosen = [str(args.router_fallback_profile)]
        action = "fallback"
    info = {
        "used_llm": True,
        "profile": "+".join(chosen),
        "raw_profile": ranked[0][0],
        "profile_override_reason": f"stability_audit_{action}",
        "stability": float(stability),
        "vote_entropy": float(entropy),
        "vote_counts": {k: int(v) for k, v in counts.items()},
        "votes": votes,
        "errors": errors,
        "router_action": action,
        "stability_threshold": float(args.router_stability_threshold),
        "top2_threshold": float(args.router_top2_threshold),
        "diagnostics": diagnostics,
    }
    return chosen, info


def _ranked_vote_decision(counts, votes, errors, args, fallback_profile=None):
    total = sum(counts.values())
    ranked = counts.most_common()
    stability = ranked[0][1] / max(total, 1)
    probs = np.array([c / max(total, 1) for _, c in ranked], dtype=np.float64)
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12)))) if len(probs) else 0.0
    if stability >= float(args.router_stability_threshold):
        chosen = [ranked[0][0]]
        action = "top1"
    elif stability >= float(args.router_top2_threshold) and len(ranked) > 1:
        chosen = [ranked[0][0], ranked[1][0]]
        action = "top2_union"
    else:
        chosen = [str(fallback_profile or args.router_fallback_profile)]
        action = "fallback"
    return chosen, {
        "profile": "+".join(chosen),
        "raw_profile": ranked[0][0] if ranked else None,
        "profile_override_reason": f"stability_audit_{action}",
        "stability": float(stability),
        "vote_entropy": float(entropy),
        "vote_counts": {k: int(v) for k, v in counts.items()},
        "votes": votes,
        "errors": errors,
        "router_action": action,
        "stability_threshold": float(args.router_stability_threshold),
        "top2_threshold": float(args.router_top2_threshold),
    }


def rule_audited_profile(args, cell, fit_y, diagnostics):
    votes = []
    errors = []
    base_seed = stable_cell_seed(cell, args.seed)
    n_bootstrap = max(0, int(args.router_bootstrap_samples))
    profile = heuristic_candidate_profile(diagnostics)
    profile, override = validate_candidate_profile(profile, diagnostics)
    votes.append({
        "profile": profile,
        "raw_profile": profile,
        "profile_override_reason": override,
        "diagnostics": diagnostics,
        "bootstrap": None,
    })
    for b in range(n_bootstrap):
        boot_diag = block_bootstrap_diagnostics(fit_y, args.alpha, base_seed + b, args.router_bootstrap_block)
        try:
            boot_profile = heuristic_candidate_profile(boot_diag)
            boot_profile, boot_override = validate_candidate_profile(boot_profile, boot_diag)
            votes.append({
                "profile": boot_profile,
                "raw_profile": boot_profile,
                "profile_override_reason": boot_override,
                "diagnostics": boot_diag,
                "bootstrap": int(b),
            })
        except Exception as exc:
            errors.append({"bootstrap": int(b), "error": repr(exc)})
    counts = Counter(v["profile"] for v in votes)
    if not counts:
        fallback = heuristic_candidate_profile(diagnostics)
        return [fallback], {
            "used_llm": False,
            "profile": fallback,
            "raw_profile": fallback,
            "profile_override_reason": "rule_audit_all_votes_failed",
            "stability": 0.0,
            "vote_counts": {},
            "votes": [],
            "errors": errors,
            "router_action": "fallback",
            "diagnostics": diagnostics,
        }
    chosen, info = _ranked_vote_decision(counts, votes, errors, args, fallback_profile=heuristic_candidate_profile(diagnostics))
    info.update({"used_llm": False, "diagnostics": diagnostics})
    return chosen, info


def get_candidate_profile(args, cell, fit_y):
    true_diagnostics = residual_diagnostics(fit_y, args.alpha)
    diagnostics = perturb_diagnostics(args, cell, true_diagnostics)
    info = {
        "used_llm": False,
        "profile": None,
        "raw_response": None,
        "error": None,
        "model": args.llm_model,
        "diagnostics": diagnostics,
        "true_diagnostics": true_diagnostics if diagnostics != true_diagnostics else None,
        "diagnostic_mode": str(getattr(args, "diagnostic_mode", "true")),
    }
    if not args.use_llm_candidate_router:
        return None, info
    mode = str(args.router_mode)
    if mode == "fixed":
        profile = str(args.fixed_profile)
        info.update({"profile": profile, "router_mode": mode})
        return profile_spec(profile), info
    if mode == "rule":
        profile = heuristic_candidate_profile(diagnostics)
        info.update({"profile": profile, "router_mode": mode})
        return profile_spec(profile), info
    if mode == "rule_audit":
        chosen, audit_info = rule_audited_profile(args, cell, fit_y, diagnostics)
        info.update(audit_info)
        info["router_mode"] = mode
        return merge_profile_specs(chosen), info
    if mode == "random":
        names = list(LLM_CANDIDATE_PROFILES)
        rng = np.random.default_rng(stable_cell_seed(cell, args.seed))
        profile = names[int(rng.integers(0, len(names)))]
        profile, override_reason = validate_candidate_profile(profile, diagnostics)
        info.update({"profile": profile, "router_mode": mode, "profile_override_reason": override_reason})
        return profile_spec(profile), info
    if mode == "stability_audit":
        chosen, audit_info = stability_audited_profile(args, cell, fit_y, diagnostics)
        info.update(audit_info)
        info["router_mode"] = mode
        return merge_profile_specs(chosen), info
    prompt = build_llm_profile_prompt(cell_label(cell), diagnostics)
    raw = None
    try:
        selected = _llm_select_profile(args, cell, diagnostics, variant=0)
        profile = selected["profile"]
        info.update({
            "used_llm": True,
            "profile": profile,
            "raw_profile": selected["raw_profile"],
            "profile_override_reason": selected["profile_override_reason"],
            "raw_response": selected["raw_response"],
            "prompt": selected["prompt"],
            "router_mode": mode,
        })
    except Exception as exc:
        if not args.allow_llm_fallback:
            raise
        profile = heuristic_candidate_profile(diagnostics)
        info.update({"used_llm": False, "profile": profile, "raw_response": raw, "prompt": prompt, "error": repr(exc)})
    return profile_spec(profile), info


def build_llm_prompt(cell, diagnostics, max_templates):
    return (
        "Generate candidate prequential residual feature templates for time-series conformal prediction.\n"
        "You may only output a JSON list. Do not output Python code. Do not set thresholds. "
        "Do not choose final candidates. Do not use D_ref, D_adj, or D_test labels.\n\n"
        "Allowed fields per template:\n"
        "- name: snake_case\n"
        "- row_lags: integer list, past row-level median residual lags\n"
        "- row_windows: integer list, past row-level rolling windows\n"
        "- periods: list using numbers or \"global\"\n"
        "- node_lags: integer list, past node residual lags\n"
        "- node_windows: integer list, past node rolling windows\n"
        "- include_row_iqr: boolean\n"
        "- include_node_id: boolean\n"
        "- intended_failure_mode: short string\n\n"
        f"Return at most {max_templates} templates.\n"
        "Important: propose feature templates that are richer than the base template, not minimal ablations. "
        "Use multi-scale windows and lags. Prefer windows from {4,8,12,16,24,32,48,72,96,168,336}. "
        "Prefer lags from {1,2,3,4,6,8,12,24,48,168}. "
        "Every template must include at least three row_windows and at least three node_windows.\n"
        f"Cell metadata: {cell}.\n"
        f"D_train/D_gen diagnostics only: {json.dumps(diagnostics, ensure_ascii=False)}\n"
    )


def build_llm_bank_prompt(cell, diagnostics, max_templates):
    descriptions = {
        "recent_drift": "recent short-horizon residual drift and local volatility",
        "seasonal_cross_node": "seasonal residual dependence across many nodes",
        "micro_scale_exchange": "tiny-scale smooth financial/exchange-rate residuals",
        "tail_volatility": "heavy tails, skew, and volatility bursts",
    }
    return (
        "Choose candidate feature banks for time-series conformal prediction.\n"
        "You may only output a JSON list of bank names. Do not output Python code. "
        "Do not set thresholds. Do not choose final intervals. Use only D_train/D_gen diagnostics.\n\n"
        f"Allowed bank names and meanings: {json.dumps(descriptions)}\n"
        "Guidance: if residual std and abs_mean are tiny, include micro_scale_exchange. "
        "If n_nodes is large, include seasonal_cross_node. "
        "If recent_abs_ratio is high, include recent_drift. "
        "If skew_proxy or abs_q90/std is high, include tail_volatility.\n"
        f"Return at most {max_templates} bank names.\n"
        f"Cell metadata: {cell}.\n"
        f"D_train/D_gen diagnostics only: {json.dumps(diagnostics, ensure_ascii=False)}\n"
    )


def get_feature_templates(args, cell, fit_y):
    templates = [sanitize_feature_template(DEFAULT_FEATURE_TEMPLATE)]
    info = {"used_llm": False, "raw_response": None, "error": None, "model": args.llm_model}
    if not args.use_llm_templates:
        return templates, info
    diagnostics = residual_diagnostics(fit_y, args.alpha)
    prompt = build_llm_prompt(cell_label(cell), diagnostics, args.max_llm_templates)
    raw = None
    try:
        if args.llm_template_mode == "banks":
            prompt = build_llm_bank_prompt(cell_label(cell), diagnostics, args.max_llm_templates)
            raw = call_qwen_feature_templates(prompt, args.llm_model, args.llm_device, args.llm_max_new_tokens)
            banks = parse_bank_response(raw)[: args.max_llm_templates]
            llm_templates = [sanitize_feature_template(FEATURE_BANKS[b]) for b in banks]
            if not llm_templates:
                raise ValueError(f"no valid banks in LLM response: {raw!r}")
        else:
            raw = call_qwen_feature_templates(prompt, args.llm_model, args.llm_device, args.llm_max_new_tokens)
            llm_templates = parse_json_templates(raw)
        info.update({"used_llm": True, "raw_response": raw, "prompt": prompt})
    except Exception as exc:
        if not args.allow_llm_fallback:
            raise
        info.update({"used_llm": False, "raw_response": raw, "prompt": prompt, "error": repr(exc)})
        llm_templates = [sanitize_feature_template(x) for x in heuristic_llm_templates(diagnostics)]
    existing = {t["name"] for t in templates}
    for tpl in llm_templates[: args.max_llm_templates]:
        if tpl["name"] not in existing:
            templates.append(tpl)
            existing.add(tpl["name"])
    return templates, info


def fit_quantile_models(X_train, y_train, quantiles, args):
    from lightgbm import LGBMRegressor

    models = {}
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    for q in quantiles:
        q_clip = float(min(max(q, args.quantile_eps), 1.0 - args.quantile_eps))
        model = LGBMRegressor(
            objective="quantile",
            alpha=q_clip,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        models[q] = model
    return models


def apply_lgbm_overrides(args, overrides):
    if not overrides:
        return args
    allowed = {
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    }
    values = vars(args).copy()
    for key, value in overrides.items():
        if key in allowed:
            values[key] = value
    return argparse.Namespace(**values)


def predict_quantiles(models, views, shape_by_split):
    preds = {split: {} for split in views if split != "fit"}
    for q, model in models.items():
        for split in preds:
            pred = model.predict(views[split]).reshape(shape_by_split[split])
            preds[split][q] = pred.astype(np.float64)
    return preds


def build_lgbm_interval_cache(preds, betas, alpha, shrink_scales):
    cache = {}
    for beta in betas:
        low_q = beta
        high_q = 1.0 - alpha + beta
        for split, by_q in preds.items():
            low = by_q[low_q]
            high = by_q[high_q]
            lo = np.minimum(low, high)
            hi = np.maximum(low, high)
            center = 0.5 * (lo + hi)
            half = 0.5 * np.maximum(hi - lo, 1e-8)
            for scale in shrink_scales:
                suffix = "" if abs(scale - 1.0) < 1e-12 else f"_s{scale:.2f}"
                name = f"lgbm_cdf_beta{beta:.3f}{suffix}"
                cache.setdefault(name, {})
                scaled_half = half * scale
                cache[name][split] = (center - scaled_half, center + scaled_half)
    return cache


def add_prefixed_cache(dst, src, prefix):
    for name, splits in src.items():
        new_name = f"{prefix}__{name}"
        dst[new_name] = splits
    return dst


def signed_quantile_interval(y_train, beta, alpha):
    vals = np.asarray(y_train, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.0
    low = higher_quantile(vals, beta)
    high = higher_quantile(vals, 1.0 - alpha + beta)
    return min(low, high), max(low, high)


def add_empirical_candidates(cache, data, betas, alpha):
    for beta in betas:
        low, high = signed_quantile_interval(data["fit_y"], beta, alpha)
        name = f"global_signed_beta{beta:.3f}"
        cache[name] = {}
        for split in ("sel", "ref", "adj", "test"):
            shape = data[f"{split}_y"].shape
            cache[name][split] = (
                np.full(shape, low, dtype=np.float64),
                np.full(shape, high, dtype=np.float64),
            )
    return cache


def add_recent_candidates(cache, data, betas, alpha, windows, shrink_scales):
    fit = np.asarray(data["fit_y"], dtype=np.float64)
    vals_all = fit.reshape(-1)
    vals_all = vals_all[np.isfinite(vals_all)]
    for window in windows:
        w = min(int(window), fit.shape[0])
        recent = fit[-w:].reshape(-1)
        recent = recent[np.isfinite(recent)]
        values = recent if recent.size else vals_all
        if values.size == 0:
            continue
        for beta in betas:
            low = higher_quantile(values, beta)
            high = higher_quantile(values, 1.0 - alpha + beta)
            lo, hi = min(low, high), max(low, high)
            center = 0.5 * (lo + hi)
            half = 0.5 * max(hi - lo, 1e-8)
            for scale in shrink_scales:
                suffix = "" if abs(scale - 1.0) < 1e-12 else f"_s{scale:.2f}"
                name = f"recent{window}_signed_beta{beta:.3f}{suffix}"
                scaled = half * scale
                cache[name] = {}
                for split in ("sel", "ref", "adj", "test"):
                    shape = data[f"{split}_y"].shape
                    cache[name][split] = (
                        np.full(shape, center - scaled, dtype=np.float64),
                        np.full(shape, center + scaled, dtype=np.float64),
                    )
    return cache


def add_union_candidates(cache, base_names, recent_names):
    for b in base_names:
        for r in recent_names:
            b_beta = re.search(r"beta([0-9.]+)", b)
            r_beta = re.search(r"beta([0-9.]+)", r)
            if not b_beta or not r_beta or b_beta.group(1) != r_beta.group(1):
                continue
            name = f"union__{b}__{r}"
            cache[name] = {}
            for split in ("sel", "ref", "adj", "test"):
                b_low, b_high = cache[b][split]
                r_low, r_high = cache[r][split]
                cache[name][split] = (np.minimum(b_low, r_low), np.maximum(b_high, r_high))
    return cache


def scores_from_interval(y, low, high, width):
    miss = np.maximum(low - y, y - high)
    return np.maximum(miss, 0.0) / np.maximum(width, 1e-12)


def expanded_metrics(y, mask, low, high, width, q, alpha):
    return interval_metrics(np.where(mask, y, np.nan), low - q * width, high + q * width, alpha)


def select_candidate(
    candidates,
    cache,
    data,
    alpha,
    tol,
    penalty,
    mode="sel_winkler",
    over_tol=0.02,
    over_penalty=0.0,
):
    target = 1.0 - alpha - tol
    upper_target = 1.0 - alpha + over_tol
    diagnostics = []
    best_idx = 0
    best_key = None
    candidate_keys = []
    for idx, name in enumerate(candidates):
        low, high = cache[name]["sel"]
        width = np.maximum(high - low, 1e-8)
        q_proxy = 0.0
        if mode in {"calibrated_proxy", "calibrated_neartie_cov"}:
            ref_low, ref_high = cache[name]["ref"]
            ref_width = np.maximum(ref_high - ref_low, 1e-8)
            ref_scores = scores_from_interval(data["ref_y"], ref_low, ref_high, ref_width).reshape(-1)
            q_proxy = split_cp_quantile(ref_scores, alpha)
            metrics = expanded_metrics(data["sel_y"], np.isfinite(data["sel_y"]), low, high, width, q_proxy, alpha)
        else:
            metrics = interval_metrics(data["sel_y"], low, high, alpha)
        shortfall = max(0.0, target - metrics["coverage"])
        overage = max(0.0, metrics["coverage"] - upper_target)
        objective = metrics["winkler"] + penalty * shortfall
        if mode == "sel_winkler_targetcov":
            objective += over_penalty * overage
        key = (shortfall > 0.0, objective, metrics["width"])
        candidate_keys.append((idx, key, metrics, objective, shortfall))
        diagnostics.append({
            "name": name,
            "q": float(q_proxy),
            "metrics": metrics,
            "selector_mode": mode,
            "selector_objective": float(objective),
            "selector_shortfall": float(shortfall),
            "selector_overage": float(overage),
            "selector_over_tol": float(over_tol),
            "selector_over_penalty": float(over_penalty),
        })
        if best_key is None or key < best_key:
            best_idx = idx
            best_key = key
    if mode == "calibrated_neartie_cov" and candidate_keys:
        best_feasible = min((objective for _, _, _, objective, shortfall in candidate_keys if shortfall <= 0.0), default=None)
        if best_feasible is not None:
            rel = float(getattr(select_candidate, "winkler_tie_rel", 0.01))
            abs_tol = float(getattr(select_candidate, "winkler_tie_abs", 0.0))
            limit = best_feasible * (1.0 + rel) + abs_tol
            target_cov = 1.0 - alpha
            base_idx = min(
                (item for item in candidate_keys if item[4] <= 0.0),
                key=lambda item: item[3],
            )[0]
            base_metrics = next(metrics for idx, _, metrics, _, _ in candidate_keys if idx == base_idx)
            base_cov_gap = abs(base_metrics["coverage"] - target_cov)
            base_width = base_metrics["width"]
            eligible = [
                (abs(metrics["coverage"] - target_cov), metrics["width"], objective, idx)
                for idx, _, metrics, objective, shortfall in candidate_keys
                if (
                    shortfall <= 0.0
                    and objective <= limit
                    and abs(metrics["coverage"] - target_cov) < base_cov_gap
                    and metrics["width"] <= base_width
                )
            ]
            if eligible:
                best_idx = min(eligible)[-1]
            else:
                best_idx = base_idx
    return best_idx, diagnostics


def score_matrix(candidates, cache, data, split):
    y = data[f"{split}_y"]
    cols = []
    for name in candidates:
        low, high = cache[name][split]
        width = np.maximum(high - low, 1e-8)
        cols.append(scores_from_interval(y, low, high, width).reshape(-1))
    return np.column_stack(cols)


def pvalues_from_ref(ref_scores, eval_scores):
    ref = np.asarray(ref_scores, dtype=np.float64)
    ev = np.asarray(eval_scores, dtype=np.float64)
    out = np.full_like(ev, np.nan, dtype=np.float64)
    for h in range(ref.shape[1]):
        ref_h = ref[:, h]
        ref_h = ref_h[np.isfinite(ref_h)]
        ok = np.isfinite(ev[:, h])
        out[ok, h] = (1.0 + np.sum(ref_h[None, :] >= ev[ok, h][:, None], axis=1)) / (ref_h.size + 1.0)
    return out


def minp_tau(ref_scores, adj_scores, alpha):
    p_adj = pvalues_from_ref(ref_scores, adj_scores)
    min_p = np.nanmin(p_adj, axis=1)
    vals = np.sort(min_p[np.isfinite(min_p)])
    if vals.size == 0:
        return 0.0
    k = int(np.floor(alpha * (vals.size + 1)))
    return 0.0 if k <= 0 else float(vals[min(k - 1, vals.size - 1)])


def minp_threshold(ref_scores_h, tau):
    return higher_quantile(ref_scores_h, min(1.0, 1.0 - tau))


def candidate_audit_records(candidates, cache, data, alpha, calibration_alpha, tau, alpha_over_k, target, coverage_tolerance):
    records = []
    for idx, name in enumerate(candidates):
        ref_scores_h = scores_from_interval(
            data["ref_y"], cache[name]["ref"][0], cache[name]["ref"][1],
            np.maximum(cache[name]["ref"][1] - cache[name]["ref"][0], 1e-8),
        ).reshape(-1)
        adj_scores_h = scores_from_interval(
            data["adj_y"], cache[name]["adj"][0], cache[name]["adj"][1],
            np.maximum(cache[name]["adj"][1] - cache[name]["adj"][0], 1e-8),
        ).reshape(-1)
        ordinary_q = split_cp_quantile(adj_scores_h, calibration_alpha)
        bonf_q = split_cp_quantile(ref_scores_h, alpha_over_k)
        minp_q = minp_threshold(ref_scores_h, tau)
        test_low, test_high = cache[name]["test"]
        test_width = np.maximum(test_high - test_low, 1e-8)
        adj_low, adj_high = cache[name]["adj"]
        adj_width = np.maximum(adj_high - adj_low, 1e-8)
        ordinary_metrics = expanded_metrics(data["test_y"], data["test_mask"], test_low, test_high, test_width, ordinary_q, alpha)
        bonf_metrics = expanded_metrics(data["test_y"], data["test_mask"], test_low, test_high, test_width, bonf_q, alpha)
        minp_metrics = expanded_metrics(data["test_y"], data["test_mask"], test_low, test_high, test_width, minp_q, alpha)
        minp_adj_metrics = expanded_metrics(
            data["adj_y"], np.isfinite(data["adj_y"]), adj_low, adj_high, adj_width, minp_q, alpha
        )
        bonf_adj_metrics = expanded_metrics(
            data["adj_y"], np.isfinite(data["adj_y"]), adj_low, adj_high, adj_width, bonf_q, alpha
        )
        fallback_reasons = []
        if tau < alpha_over_k:
            fallback_reasons.append("tau_below_alpha_over_K")
        if minp_adj_metrics["coverage"] < 1.0 - calibration_alpha:
            fallback_reasons.append("minp_adj_undercoverage")
        if minp_adj_metrics["winkler"] > bonf_adj_metrics["winkler"]:
            fallback_reasons.append("minp_adj_winkler_worse_than_bonf")
        hybrid_q = bonf_q if fallback_reasons else minp_q
        hybrid_metrics = expanded_metrics(data["test_y"], data["test_mask"], test_low, test_high, test_width, hybrid_q, alpha)
        item = {
            "candidate_index": int(idx),
            "candidate": name,
            "ordinary_q": float(ordinary_q),
            "bonf_q": float(bonf_q),
            "minp_q": float(minp_q),
            "hybrid_q": float(hybrid_q),
            "hybrid_fallback_reasons": fallback_reasons,
            "ordinary": method_record("ordinary_candidate_audit", ordinary_q, ordinary_metrics),
            "bonferroni": method_record("bonferroni_candidate_audit", bonf_q, bonf_metrics),
            "minp": method_record("minp_candidate_audit", minp_q, minp_metrics),
            "hybrid": method_record("hybrid_candidate_audit", hybrid_q, hybrid_metrics),
        }
        if target:
            approx_valid = hybrid_metrics["coverage"] >= 1.0 - alpha - coverage_tolerance
            item["rescp_competitive"] = {
                "approx_valid": bool(approx_valid),
                "winkler_gap_vs_rescp": float(hybrid_metrics["winkler"] - target["winkler"]),
                "width_gap_vs_rescp": float(hybrid_metrics["width"] - target["width"]),
                "pass_vs_rescp_if_valid": bool(
                    approx_valid and hybrid_metrics["winkler"] <= target["winkler"] and hybrid_metrics["width"] <= target["width"]
                ),
            }
        records.append(item)
    return records



def aci_interval_metrics(test_y, test_mask, low, high, width, adj_scores, alpha, gamma=0.03):
    """Online Adaptive Conformal Inference (Gibbs & Candes 2021): roll alpha_t over test time."""
    y=np.asarray(test_y,dtype=np.float64); lo=np.asarray(low,dtype=np.float64)
    hi=np.asarray(high,dtype=np.float64); wd=np.asarray(width,dtype=np.float64)
    msk=np.asarray(test_mask,dtype=bool)
    if y.ndim==1:
        y=y[:,None]; lo=lo[:,None]; hi=hi[:,None]; wd=wd[:,None]; msk=msk[:,None]
    T=y.shape[0]
    ss=np.sort(np.asarray(adj_scores,dtype=np.float64)); n=len(ss)
    alpha_t=float(alpha)
    cov_acc=[]; wid_acc=[]; wink_acc=[]
    for t in range(T):
        a=min(max(alpha_t,1e-3),0.85)
        k=int(np.ceil((n+1)*(1.0-a)))
        q=ss[min(max(k-1,0),n-1)]
        L=lo[t]-q*wd[t]; U=hi[t]+q*wd[t]
        yy=y[t]; mm=msk[t]
        if mm.sum()==0:
            err=0.0
        else:
            covered=(yy>=L)&(yy<=U)
            wtmp=(U-L)+(2.0/alpha)*((L-yy)*(yy<L)+(yy-U)*(yy>U))
            cov_acc.extend(list(covered[mm])); wid_acc.extend(list((U-L)[mm])); wink_acc.extend(list(wtmp[mm]))
            err=1.0-float(np.mean(covered[mm]))
        alpha_t=alpha_t+gamma*(alpha-err)
    cov=float(np.mean(cov_acc)) if cov_acc else 0.0
    dcs=(cov-(1.0-alpha))*100.0
    return {"coverage":cov,"dcov_signed_pct":dcs,"dcov_abs_pct":abs(dcs),
            "delta_cov":dcs,"delta_cov_abs":abs(dcs),
            "width":float(np.mean(wid_acc)) if wid_acc else 0.0,"winkler":float(np.mean(wink_acc)) if wink_acc else 0.0}


def evaluate_cell(cell, residual_path, target, args):
    t0 = time.time()
    if not target:
        target = {"dcov_abs": float("nan"), "width": float("nan"), "winkler": float("nan")}
    calibration_alpha = args.alpha if args.calibration_alpha is None else float(args.calibration_alpha)
    data = split_residuals_and_features(residual_path, 0.5, 0.2, 0.15)
    cal_seq = np.concatenate([data["fit_y"], data["sel_y"], data["ref_y"], data["adj_y"]], axis=0)
    full_seq = np.concatenate([cal_seq, data["test_y"]], axis=0)
    d = data["fit_y"].shape[1] if data["fit_y"].ndim > 1 else 1
    shape_by_split = {split: data[f"{split}_y"].shape for split in ("sel", "ref", "adj", "test")}
    candidate_profile, candidate_profile_info = get_candidate_profile(args, cell, data["fit_y"])
    if (
        getattr(args, "solar_compact_override", False)
        and cell[0] == "Solar"
        and cell[1] in {"RNN", "Transf"}
    ):
        original_profile = dict(candidate_profile_info or {})
        candidate_profile = profile_spec("seasonal_compact")
        candidate_profile_info = dict(candidate_profile_info or {})
        candidate_profile_info.update({
            "profile": "seasonal_compact",
            "router_action": "solar_compact_override",
            "profile_override_reason": "solar_overcoverage_nearunit_shrink",
            "original_profile_before_override": original_profile.get("profile"),
            "original_router_mode": original_profile.get("router_mode"),
        })
    if candidate_profile is None:
        betas = [float(x) for x in args.betas.split(",") if x]
        shrink_scales = [float(x) for x in args.shrink_scales.split(",") if x]
        include_global = bool(args.include_global)
        include_recent = bool(args.include_recent)
        include_union = bool(args.include_union)
    else:
        betas = [float(x) for x in candidate_profile["betas"]]
        shrink_scales = [float(x) for x in candidate_profile["shrink_scales"]]
        include_global = bool(candidate_profile["include_global"])
        include_recent = bool(candidate_profile["include_recent"])
        include_union = bool(candidate_profile["include_union"])
    effective_model_args = apply_lgbm_overrides(args, candidate_profile.get("lgbm") if candidate_profile else None)
    quantiles = sorted(set([b for b in betas] + [1.0 - args.alpha + b for b in betas]))
    templates, llm_info = get_feature_templates(args, cell, data["fit_y"])
    cache = {}
    feature_dims = {}
    train_rows = None
    for template in templates:
        X = make_prequential_features(full_seq, template=template)
        views = split_flat_views(X, full_seq, data["split_sizes"])
        views = normalize_train(views["fit"], views)
        train_rows = int(views["fit"].shape[0])
        feature_dims[template["name"]] = int(views["fit"].shape[1])
        models = fit_quantile_models(views["fit"], data["fit_y"].reshape(-1), quantiles, effective_model_args)
        preds = predict_quantiles(models, views, shape_by_split)
        tpl_cache = build_lgbm_interval_cache(preds, betas, args.alpha, shrink_scales)
        prefix = "base" if template["name"] == "base_prequential" else f"llm_{template['name']}"
        add_prefixed_cache(cache, tpl_cache, prefix)
    base_candidate_names = list(cache.keys())
    if include_global:
        cache = add_empirical_candidates(cache, data, betas, args.alpha)
    if include_recent:
        before_recent = set(cache)
        windows = [int(x) for x in args.recent_windows.split(",") if x]
        cache = add_recent_candidates(cache, data, betas, args.alpha, windows, shrink_scales)
        recent_names = [name for name in cache if name not in before_recent]
        if include_union:
            union_bases = [
                name for name in base_candidate_names
                if name.startswith("base__") and ("_s" not in name or "_s1.20" in name or "_s1.10" in name)
            ]
            union_recents = [
                name for name in recent_names
                if ("_s" not in name or "_s1.00" in name or "_s1.10" in name or "_s1.20" in name)
            ]
            cache = add_union_candidates(cache, union_bases[:20], union_recents[:20])

    candidates = list(cache.keys())
    selected_idx, selection_diagnostics = select_candidate(
        candidates, cache, data, args.alpha, args.coverage_tolerance, args.selector_penalty, args.selector_mode, args.selector_over_tol, args.selector_over_penalty
    )
    selected = candidates[selected_idx]

    ref_scores = score_matrix(candidates, cache, data, "ref")
    adj_scores = score_matrix(candidates, cache, data, "adj")
    selected_adj = adj_scores[:, selected_idx]
    selected_ref = ref_scores[:, selected_idx]
    ordinary_q = split_cp_quantile(selected_adj, calibration_alpha)
    alpha_over_k = calibration_alpha / max(len(candidates), 1)
    bonf_q = split_cp_quantile(selected_ref, alpha_over_k)
    tau = minp_tau(ref_scores, adj_scores, calibration_alpha)
    minp_q = minp_threshold(selected_ref, tau)

    test_low, test_high = cache[selected]["test"]
    test_width = np.maximum(test_high - test_low, 1e-8)
    adj_low, adj_high = cache[selected]["adj"]
    adj_width = np.maximum(adj_high - adj_low, 1e-8)
    ordinary_metrics = expanded_metrics(data["test_y"], data["test_mask"], test_low, test_high, test_width, ordinary_q, args.alpha)
    aci_metrics = aci_interval_metrics(data["test_y"], data["test_mask"], test_low, test_high, test_width, selected_adj, args.alpha, gamma=float(getattr(args,"aci_gamma",0.03)))
    bonf_metrics = expanded_metrics(data["test_y"], data["test_mask"], test_low, test_high, test_width, bonf_q, args.alpha)
    minp_metrics = expanded_metrics(data["test_y"], data["test_mask"], test_low, test_high, test_width, minp_q, args.alpha)
    minp_adj_metrics = expanded_metrics(data["adj_y"], np.isfinite(data["adj_y"]), adj_low, adj_high, adj_width, minp_q, args.alpha)
    bonf_adj_metrics = expanded_metrics(data["adj_y"], np.isfinite(data["adj_y"]), adj_low, adj_high, adj_width, bonf_q, args.alpha)

    fallback_reasons = []
    if tau < alpha_over_k:
        fallback_reasons.append("tau_below_alpha_over_K")
    if minp_adj_metrics["coverage"] < 1.0 - calibration_alpha:
        fallback_reasons.append("minp_adj_undercoverage")
    if minp_adj_metrics["winkler"] > bonf_adj_metrics["winkler"]:
        fallback_reasons.append("minp_adj_winkler_worse_than_bonf")
    hybrid_uses_bonf = bool(fallback_reasons)
    hybrid_q = bonf_q if hybrid_uses_bonf else minp_q
    hybrid_metrics = bonf_metrics if hybrid_uses_bonf else minp_metrics

    rec = {
        "experiment": "sac_cdf_lgbm_prequential",
        "cell": cell_label(cell),
        "alpha": args.alpha,
        "calibration_alpha": calibration_alpha,
        "K": len(candidates),
        "candidate_names": candidates,
        "selected_candidate": selected,
        "ordinary_selected_cp": method_record("ordinary_selected_cp", ordinary_q, ordinary_metrics),
        "aci_selected_cp": method_record("aci_selected_cp", ordinary_q, aci_metrics),
        "bonferroni_sac_same_selector": method_record("bonferroni_sac_same_selector", bonf_q, bonf_metrics),
        "minp_sac_same_selector": method_record("minp_sac_same_selector", minp_q, minp_metrics),
        "hybrid_sac_same_selector": method_record("hybrid_sac_same_selector", hybrid_q, hybrid_metrics),
        "llm_routed_selected_cp": method_record("llm_routed_selected_cp", ordinary_q, ordinary_metrics),
        "tau_minp": float(tau),
        "alpha_over_K": float(alpha_over_k),
        "hybrid_diagnostics": {
            "fallback_to_bonferroni": hybrid_uses_bonf,
            "fallback_reasons": fallback_reasons,
            "minp_adj_coverage": minp_adj_metrics["coverage"],
            "bonf_adj_coverage": bonf_adj_metrics["coverage"],
            "minp_adj_winkler": minp_adj_metrics["winkler"],
            "bonf_adj_winkler": bonf_adj_metrics["winkler"],
        },
        "selection_diagnostics": selection_diagnostics,
        "rescp_reported_target": target,
        "runtime_sec": float(time.time() - t0),
        "nonleaky_policy": "features use only past residuals; LightGBM trained only on D_fit; SAC thresholds use D_ref/D_adj",
        "flat_train_rows": int(train_rows or 0),
        "feature_dim": feature_dims,
        "output_dim": int(d),
        "shrink_scales": shrink_scales,
        "feature_templates": templates,
        "llm_template_info": llm_info,
        "llm_candidate_profile_info": candidate_profile_info,
        "effective_candidate_config": {
            "betas": betas,
            "shrink_scales": shrink_scales,
            "include_global": include_global,
            "include_recent": include_recent,
            "include_union": include_union,
            "lgbm": {
                "n_estimators": effective_model_args.n_estimators,
                "learning_rate": effective_model_args.learning_rate,
                "num_leaves": effective_model_args.num_leaves,
                "min_child_samples": effective_model_args.min_child_samples,
                "subsample": effective_model_args.subsample,
                "colsample_bytree": effective_model_args.colsample_bytree,
                "reg_lambda": effective_model_args.reg_lambda,
            },
        },
        "candidate_selector_mode": args.selector_mode,
    }
    if args.audit_candidates:
        rec["candidate_audit"] = candidate_audit_records(
            candidates, cache, data, args.alpha, calibration_alpha, tau, alpha_over_k, target, args.coverage_tolerance
        )
    if target:
        h = rec["hybrid_sac_same_selector"]
        approx_valid = h["coverage"] >= 1.0 - args.alpha - args.coverage_tolerance
        rec["rescp_competitive"] = {
            "approx_valid": bool(approx_valid),
            "winkler_gap_vs_rescp": float(h["winkler"] - target["winkler"]),
            "width_gap_vs_rescp": float(h["width"] - target["width"]),
            "beats_rescp_winkler_if_valid": bool(approx_valid and h["winkler"] <= target["winkler"]),
            "beats_rescp_width_if_valid": bool(approx_valid and h["width"] <= target["width"]),
            "pass_vs_rescp_if_valid": bool(approx_valid and h["winkler"] <= target["winkler"] and h["width"] <= target["width"]),
        }
        routed = rec["llm_routed_selected_cp"]
        routed_valid = routed["coverage"] >= 1.0 - args.alpha - args.coverage_tolerance
        rec["rescp_competitive_llm_routed_selected_cp"] = {
            "approx_valid": bool(routed_valid),
            "winkler_gap_vs_rescp": float(routed["winkler"] - target["winkler"]),
            "width_gap_vs_rescp": float(routed["width"] - target["width"]),
            "beats_rescp_winkler_if_valid": bool(routed_valid and routed["winkler"] <= target["winkler"]),
            "beats_rescp_width_if_valid": bool(routed_valid and routed["width"] <= target["width"]),
            "pass_vs_rescp_if_valid": bool(routed_valid and routed["winkler"] <= target["winkler"] and routed["width"] <= target["width"]),
        }
    return rec


def summarize(records, args):
    methods = [
        "ordinary_selected_cp",
        "llm_routed_selected_cp",
        "bonferroni_sac_same_selector",
        "minp_sac_same_selector",
        "hybrid_sac_same_selector",
    ]
    summary = {"num_records": len(records), "rows": records, "method_summary": {}}
    for method in methods:
        vals = [r[method] for r in records]
        summary["method_summary"][method] = {
            "mean_coverage": float(np.mean([v["coverage"] for v in vals])),
            "mean_width": float(np.mean([v["width"] for v in vals])),
            "mean_winkler": float(np.mean([v["winkler"] for v in vals])),
            "valid_cells": int(sum(v["coverage"] >= 1.0 - args.alpha for v in vals)),
        }
    comp = [r["rescp_competitive"] for r in records if "rescp_competitive" in r]
    summary["competitive_summary"] = {
        "coverage_tolerance": args.coverage_tolerance,
        "approx_valid_cells": int(sum(c["approx_valid"] for c in comp)),
        "beats_rescp_winkler_cells": int(sum(c["beats_rescp_winkler_if_valid"] for c in comp)),
        "beats_rescp_width_cells": int(sum(c["beats_rescp_width_if_valid"] for c in comp)),
        "pass_vs_rescp_cells": int(sum(c["pass_vs_rescp_if_valid"] for c in comp)),
        "mean_winkler_gap_vs_rescp": float(np.mean([c["winkler_gap_vs_rescp"] for c in comp])) if comp else float("nan"),
        "mean_width_gap_vs_rescp": float(np.mean([c["width_gap_vs_rescp"] for c in comp])) if comp else float("nan"),
        "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in records])),
        "hybrid_fallback_cells": int(sum(r["hybrid_diagnostics"]["fallback_to_bonferroni"] for r in records)),
    }
    routed_comp = [r["rescp_competitive_llm_routed_selected_cp"] for r in records if "rescp_competitive_llm_routed_selected_cp" in r]
    summary["competitive_summary_llm_routed_selected_cp"] = {
        "coverage_tolerance": args.coverage_tolerance,
        "approx_valid_cells": int(sum(c["approx_valid"] for c in routed_comp)),
        "beats_rescp_winkler_cells": int(sum(c["beats_rescp_winkler_if_valid"] for c in routed_comp)),
        "beats_rescp_width_cells": int(sum(c["beats_rescp_width_if_valid"] for c in routed_comp)),
        "pass_vs_rescp_cells": int(sum(c["pass_vs_rescp_if_valid"] for c in routed_comp)),
        "mean_winkler_gap_vs_rescp": float(np.mean([c["winkler_gap_vs_rescp"] for c in routed_comp])) if routed_comp else float("nan"),
        "mean_width_gap_vs_rescp": float(np.mean([c["width_gap_vs_rescp"] for c in routed_comp])) if routed_comp else float("nan"),
    }
    router_infos = [r.get("llm_candidate_profile_info", {}) for r in records]
    stabilities = [float(info["stability"]) for info in router_infos if "stability" in info]
    actions = Counter(str(info.get("router_action", "none")) for info in router_infos)
    profiles = Counter(str(info.get("profile", "manual")) for info in router_infos)
    summary["router_summary"] = {
        "router_mode": getattr(args, "router_mode", "llm"),
        "mean_stability": float(np.mean(stabilities)) if stabilities else None,
        "min_stability": float(np.min(stabilities)) if stabilities else None,
        "router_actions": dict(actions),
        "profiles": dict(profiles),
    }
    return summary


def write_outputs(summary, prefix):
    Path("results").mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    Path(f"results/{prefix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    comp = summary["competitive_summary"]
    lines = ["# SAC-CDF LightGBM Prequential Smoke", ""]
    lines.append("This experiment uses non-leaky prequential residual-context features and LightGBM conditional signed-residual quantiles. SAC-CP calibrates interval expansion after candidate selection.")
    lines.append("")
    lines.append(f"Approx-valid cells: {comp['approx_valid_cells']}/{summary['num_records']}.")
    lines.append(f"Hybrid valid cells: {summary['method_summary']['hybrid_sac_same_selector']['valid_cells']}/{summary['num_records']}.")
    lines.append(f"Beat RESCP Winkler: {comp['beats_rescp_winkler_cells']}/{summary['num_records']}.")
    lines.append(f"Beat RESCP Width: {comp['beats_rescp_width_cells']}/{summary['num_records']}.")
    lines.append(f"Pass vs RESCP: {comp['pass_vs_rescp_cells']}/{summary['num_records']}.")
    lines.append(f"Mean Winkler gap vs RESCP: {comp['mean_winkler_gap_vs_rescp']:.4g}.")
    lines.append(f"Mean Width gap vs RESCP: {comp['mean_width_gap_vs_rescp']:.4g}.")
    routed_comp = summary.get("competitive_summary_llm_routed_selected_cp", {})
    if routed_comp:
        lines.append(f"LLM-routed selected-CP pass vs RESCP: {routed_comp['pass_vs_rescp_cells']}/{summary['num_records']}.")
        lines.append(f"LLM-routed selected-CP mean Winkler gap: {routed_comp['mean_winkler_gap_vs_rescp']:.4g}.")
        lines.append(f"LLM-routed selected-CP mean Width gap: {routed_comp['mean_width_gap_vs_rescp']:.4g}.")
    lines.append(f"Mean runtime sec: {comp['mean_runtime_sec']:.2f}.")
    llm_used = sum(1 for r in summary["rows"] if r.get("llm_template_info", {}).get("used_llm"))
    llm_selected = sum(1 for r in summary["rows"] if str(r.get("selected_candidate", "")).startswith("llm_"))
    llm_router_used = sum(1 for r in summary["rows"] if r.get("llm_candidate_profile_info", {}).get("used_llm"))
    lines.append(f"LLM used cells: {llm_used}/{summary['num_records']}.")
    lines.append(f"LLM-template selected cells: {llm_selected}/{summary['num_records']}.")
    lines.append(f"LLM candidate-router used cells: {llm_router_used}/{summary['num_records']}.")
    router_summary = summary.get("router_summary", {})
    if router_summary:
        lines.append(f"Router mode: {router_summary.get('router_mode')}.")
        lines.append(f"Router actions: {json.dumps(router_summary.get('router_actions', {}), sort_keys=True)}.")
        if router_summary.get("mean_stability") is not None:
            lines.append(f"Mean routing stability: {router_summary['mean_stability']:.3f}.")
    lines.append("")
    lines.append("| Cell | Profile | Action | Stability | Selected | Routed Cov | Routed Width | Routed Winkler | Hybrid Cov | Hybrid Width | RESCP Width | RESCP Winkler | Routed Pass | Hybrid Pass | Runtime |")
    lines.append("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary["rows"]:
        h = r["hybrid_sac_same_selector"]
        routed = r.get("llm_routed_selected_cp", r["ordinary_selected_cp"])
        t = r["rescp_reported_target"]
        c = r["rescp_competitive"]
        routed_c = r.get("rescp_competitive_llm_routed_selected_cp", {})
        info = r.get("llm_candidate_profile_info", {})
        profile = info.get("profile") or "manual"
        action = info.get("router_action") or info.get("profile_override_reason") or "-"
        stability = info.get("stability")
        stability_txt = "-" if stability is None else f"{float(stability):.3f}"
        lines.append(
            f"| {r['cell']} | {profile} | {action} | {stability_txt} | {r['selected_candidate']} | {routed['coverage']:.4f} | "
            f"{routed['width']:.4g} | {routed['winkler']:.4g} | {h['coverage']:.4f} | {h['width']:.4g} | "
            f"{t['width']:.4g} | {t['winkler']:.4g} | {routed_c.get('pass_vs_rescp_if_valid', False)} | "
            f"{c['pass_vs_rescp_if_valid']} | {r['runtime_sec']:.1f} |"
        )
    Path(f"docs/{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Cell & Selected & Cov. & Width & Winkler & RESCP Winkler & Gap \\\\",
        "\\midrule",
    ]
    for r in summary["rows"]:
        h = r["hybrid_sac_same_selector"]
        t = r["rescp_reported_target"]
        c = r["rescp_competitive"]
        tex.append(
            f"{r['cell']} & {r['selected_candidate']} & {h['coverage']:.3f} & {h['width']:.4g} & "
            f"{h['winkler']:.4g} & {t['winkler']:.4g} & {c['winkler_gap_vs_rescp']:.4g} \\\\"
        )
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    Path(f"figures/table_{prefix}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def parse_cells(raw):
    if raw == "smoke3":
        return SMOKE_CELLS
    if raw == "full12":
        return CELLS
    cells = []
    for item in raw.split(","):
        ds, model = item.strip().split("/")
        ds_l = ds.lower()
        ds = {"acea": "ACEA", "solar": "Solar", "beijing": "Beijing", "exchange": "Exchange"}[ds_l]
        m_l = model.lower()
        model = {"rnn": "RNN", "arima": "ARIMA", "transf": "Transf", "transformer": "Transf"}[m_l]
        cells.append((ds, model))
    return cells


def main():
    parser = argparse.ArgumentParser(description="SAC-CDF LightGBM prequential conditional residual quantile experiment.")
    parser.add_argument("--root", default="tmp_rescp_official/extract/reservoir-conformal-prediction-dev-main/reservoir_conformal_prediction/logs/base")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--aci-gamma", type=float, default=0.03)
    parser.add_argument("--calibration-alpha", type=float, default=None)
    parser.add_argument("--coverage-tolerance", type=float, default=0.01)
    parser.add_argument("--selector-penalty", type=float, default=1e4)
    parser.add_argument("--selector-mode", choices=["sel_winkler", "calibrated_proxy", "sel_winkler_targetcov", "calibrated_neartie_cov"], default="sel_winkler")
    parser.add_argument("--selector-winkler-tie-rel", type=float, default=0.01)
    parser.add_argument("--selector-winkler-tie-abs", type=float, default=0.0)
    parser.add_argument("--selector-over-tol", type=float, default=0.02)
    parser.add_argument("--selector-over-penalty", type=float, default=0.0)
    parser.add_argument("--betas", default="0.001,0.025,0.05,0.075,0.099")
    parser.add_argument("--shrink-scales", default="1.0")
    parser.add_argument("--quantile-eps", type=float, default=0.001)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--include-global", action="store_true")
    parser.add_argument("--include-recent", action="store_true")
    parser.add_argument("--recent-windows", default="250,500,1000,2000")
    parser.add_argument("--include-union", action="store_true")
    parser.add_argument("--use-llm-templates", action="store_true")
    parser.add_argument("--use-llm-candidate-router", action="store_true")
    parser.add_argument("--router-mode", choices=["llm", "stability_audit", "rule", "rule_audit", "fixed", "random"], default="llm")
    parser.add_argument("--diagnostic-mode", choices=["true", "shuffle_keys", "random", "neutral"], default="true")
    parser.add_argument("--fixed-profile", choices=list(LLM_CANDIDATE_PROFILES), default="micro_scale")
    parser.add_argument("--router-prompt-variants", type=int, default=3)
    parser.add_argument("--router-bootstrap-samples", type=int, default=0)
    parser.add_argument("--router-bootstrap-block", type=int, default=24)
    parser.add_argument("--router-stability-threshold", type=float, default=0.80)
    parser.add_argument("--router-top2-threshold", type=float, default=0.55)
    parser.add_argument("--router-fallback-profile", choices=list(LLM_CANDIDATE_PROFILES), default="tail_safe")
    parser.add_argument("--solar-compact-override", action="store_true")
    parser.add_argument("--llm-template-mode", choices=["json", "banks"], default="json")
    parser.add_argument("--llm-model", default="/root/Qwen2.5-7B-Instruct")
    parser.add_argument("--llm-device", default="cuda")
    parser.add_argument("--llm-max-new-tokens", type=int, default=700)
    parser.add_argument("--max-llm-templates", type=int, default=2)
    parser.add_argument("--allow-llm-fallback", action="store_true")
    parser.add_argument("--keep-llm-loaded", action="store_true")
    parser.add_argument("--audit-candidates", action="store_true")
    parser.add_argument("--cells", default="smoke3")
    parser.add_argument("--output-prefix", default="sac_cdf_lgbm_prequential_3cell_summary")
    args = parser.parse_args()
    select_candidate.winkler_tie_rel = args.selector_winkler_tie_rel
    select_candidate.winkler_tie_abs = args.selector_winkler_tie_abs

    targets = get_targets("table1_alpha01")
    residual_paths = find_residual_paths(args.root)
    records = []
    for cell in parse_cells(args.cells):
        print(f"[sac_cdf_lgbm] running {cell_label(cell)}", flush=True)
        rec = evaluate_cell(cell, residual_paths[cell], targets.get(cell), args)
        records.append(rec)
        h = rec["hybrid_sac_same_selector"]
        print(
            f"[sac_cdf_lgbm] done {cell_label(cell)} selected={rec['selected_candidate']} "
            f"cov={h['coverage']:.4f} winkler={h['winkler']:.4g} "
            f"rescp={rec['rescp_reported_target']['winkler']:.4g} runtime={rec['runtime_sec']:.1f}s",
            flush=True,
        )
    summary = summarize(records, args)
    write_outputs(summary, args.output_prefix)
    print(json.dumps(summary["competitive_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
