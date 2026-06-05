import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_rescp_table6 import get_targets  # noqa: E402
from models.uncertainty.sac_cp import (  # noqa: E402
    BonferroniCalibrator,
    DGCPGuardSelector,
    FeatureKNNResidualScaleScore,
    MinPSelectionAwareCalibrator,
    ResidualScaleScore,
    ValidationScoreSelector,
    interval_metrics,
)
from models.uncertainty.sac_cp.candidate_score import is_per_candidate_features  # noqa: E402
from models.uncertainty.sac_cp.metrics import split_cp_quantile  # noqa: E402
try:
    from models.uncertainty.sac_cp.signed_quantile_candidates import CoverageConstrainedWinklerSelector
except Exception:  # pragma: no cover - optional competitive candidates
    CoverageConstrainedWinklerSelector = None
from official_residual_quickscan import DATASET_MAP, MODEL_MAP  # noqa: E402


CELLS = [
    ("Solar", "RNN"),
    ("Solar", "Transf"),
    ("Solar", "ARIMA"),
    ("Beijing", "RNN"),
    ("Beijing", "Transf"),
    ("Beijing", "ARIMA"),
    ("Exchange", "RNN"),
    ("Exchange", "Transf"),
    ("Exchange", "ARIMA"),
    ("ACEA", "RNN"),
    ("ACEA", "Transf"),
    ("ACEA", "ARIMA"),
]


def infer_keys(path):
    parts = [p.lower() for p in Path(path).parts]
    dataset = next((DATASET_MAP[p] for p in parts if p in DATASET_MAP), None)
    model = next((MODEL_MAP[p] for p in parts if p in MODEL_MAP), None)
    return dataset, model


def canonical_cell(dataset, model):
    return f"{dataset}/{model}".lower().replace("transformer", "transf")


def find_residual_paths(root):
    out = {}
    for residual_path in sorted(Path(root).glob("*/*/*/*/residuals.h5")):
        dataset, model = infer_keys(residual_path)
        if dataset is None or model is None:
            continue
        out.setdefault((dataset, model), residual_path)
    return out


def split_residuals_and_features(residual_path, fit_frac, sel_frac, ref_frac):
    residual_path = Path(residual_path)
    residuals = pd.read_hdf(residual_path, key="target")
    features = pd.read_hdf(residual_path, key="input")
    mask = pd.read_hdf(residual_path, key="target_mask").astype(bool)
    indices = np.load(residual_path.parent / "indices.npz")
    valid_target_indices = indices["valid_target_indices"]
    pos_by_abs = {int(v): i for i, v in enumerate(valid_target_indices)}
    calib_pos = [pos_by_abs[int(i)] for i in indices["calib_indices"] if int(i) in pos_by_abs]
    test_pos = [pos_by_abs[int(i)] for i in indices["test_indices"] if int(i) in pos_by_abs]

    cal_y = residuals.iloc[calib_pos].to_numpy(dtype=np.float64)
    cal_x = features.iloc[calib_pos].to_numpy(dtype=np.float64)
    test_y = residuals.iloc[test_pos].to_numpy(dtype=np.float64)
    test_x = features.iloc[test_pos].to_numpy(dtype=np.float64)
    test_mask = mask.iloc[test_pos].to_numpy(dtype=bool)

    n = cal_y.shape[0]
    n_fit = max(10, int(round(n * fit_frac)))
    n_sel = max(10, int(round(n * sel_frac)))
    n_ref = max(10, int(round(n * ref_frac)))
    if n_fit + n_sel + n_ref >= n:
        raise ValueError(
            f"Calibration split too small: n={n}, fit={n_fit}, sel={n_sel}, ref={n_ref}"
        )
    i0 = n_fit
    i1 = i0 + n_sel
    i2 = i1 + n_ref
    return {
        "fit_y": cal_y[:i0],
        "fit_x": cal_x[:i0],
        "sel_y": cal_y[i0:i1],
        "sel_x": cal_x[i0:i1],
        "ref_y": cal_y[i1:i2],
        "ref_x": cal_x[i1:i2],
        "adj_y": cal_y[i2:],
        "adj_x": cal_x[i2:],
        "test_y": test_y,
        "test_x": test_x,
        "test_mask": test_mask,
        "split_sizes": {
            "fit": int(i0),
            "sel": int(i1 - i0),
            "ref": int(i2 - i1),
            "adj": int(n - i2),
            "test": int(test_y.shape[0]),
        },
    }


def build_candidates(family):
    family = family.lower()
    small = [
        ResidualScaleScore("standard_abs", center_mode="zero", scale_mode="unit"),
        ResidualScaleScore("time_decay_abs", center_mode="global_median", scale_mode="global_mad", decay=200.0),
        FeatureKNNResidualScaleScore(
            "knn_regime_100", k=100, window=1000, center_mode="node_median", scale_mode="node_mad"
        ),
    ]
    if family == "small":
        return small
    full_extra = [
        FeatureKNNResidualScaleScore(
            "knn_regime_50", k=50, window=1000, center_mode="node_median", scale_mode="node_mad"
        ),
        ResidualScaleScore("residual_mad_scale", center_mode="node_median", scale_mode="node_mad"),
        ResidualScaleScore("residual_iqr_scale", center_mode="global_median", scale_mode="global_iqr"),
        FeatureKNNResidualScaleScore(
            "context_pool_score", k=200, window=1000, center_mode="node_median", scale_mode="node_mad"
        ),
    ]
    return small[:2] + [full_extra[0], small[2]] + full_extra[1:]


def masked_metrics(residuals, mask, low, high, alpha):
    return interval_metrics(np.where(mask, residuals, np.nan), low, high, alpha)


def eval_candidate(candidate, residuals, features, mask, threshold, alpha):
    low, high = candidate.interval(residuals.shape, threshold, features=features)
    return masked_metrics(residuals, mask, low, high, alpha)


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


def mean_abs_score_correlation(candidates, residuals, features):
    per_candidate_features = is_per_candidate_features(features, len(candidates))
    cols = []
    for i, c in enumerate(candidates):
        feat = features[i] if per_candidate_features else features
        cols.append(c.score_batch(residuals, features=feat).reshape(-1))
    mat = np.column_stack(cols)
    keep = np.all(np.isfinite(mat), axis=1)
    mat = mat[keep]
    if mat.shape[0] < 3 or mat.shape[1] < 2:
        return float("nan")
    corr = np.corrcoef(mat, rowvar=False)
    vals = np.abs(corr[np.triu_indices_from(corr, k=1)])
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


class WidthSelector:
    def __init__(self, alpha):
        self.alpha = float(alpha)

    def select(self, candidates, d_sel, x_sel=None):
        best_idx = 0
        best_key = None
        diagnostics = []
        per_candidate_features = is_per_candidate_features(x_sel, len(candidates))
        for idx, candidate in enumerate(candidates):
            features = x_sel[idx] if per_candidate_features else x_sel
            scores = candidate.score_batch(d_sel, features=features).reshape(-1)
            q = split_cp_quantile(scores, self.alpha)
            low, high = candidate.interval(d_sel.shape, q, features=features)
            metrics = interval_metrics(d_sel, low, high, self.alpha)
            key = (metrics["width"], metrics["winkler"], metrics["dcov_abs_pct"])
            diagnostics.append({"name": candidate.name, "q": q, "metrics": metrics})
            if best_key is None or key < best_key:
                best_idx, best_key = idx, key
        return best_idx, diagnostics


class WinklerSelector(ValidationScoreSelector):
    pass


def make_selector(name, alpha):
    name = name.lower()
    if name == "validation_score":
        return ValidationScoreSelector(alpha)
    if name == "width":
        return WidthSelector(alpha)
    if name == "winkler":
        return WinklerSelector(alpha)
    if name == "diagnostic":
        return DGCPGuardSelector(alpha)
    if name == "coverage_winkler":
        if CoverageConstrainedWinklerSelector is None:
            raise ValueError("coverage_winkler selector requires signed_quantile_candidates.py")
        return CoverageConstrainedWinklerSelector(alpha, tol=0.01)
    raise ValueError(f"Unknown selector: {name}")


def run_cell(residual_path, alpha, family, selector_name, target, extra_candidates=None):
    dataset, model = infer_keys(residual_path)
    data = split_residuals_and_features(residual_path, 0.5, 0.2, 0.15)
    candidates = build_candidates(family)
    if extra_candidates:
        existing = {c.name for c in candidates}
        for candidate in extra_candidates:
            if candidate.name in existing:
                raise ValueError(f"Duplicate candidate name: {candidate.name}")
            candidates.append(candidate)
            existing.add(candidate.name)
    for candidate in candidates:
        candidate.fit(data["fit_y"], features=data["fit_x"])

    sel_features = data["sel_x"]
    ref_features = data["ref_x"]
    adj_features = data["adj_x"]
    test_features = data["test_x"]

    selector = make_selector(selector_name, alpha)
    selected_idx, selection_diagnostics = selector.select(candidates, data["sel_y"], x_sel=sel_features)
    selected = candidates[selected_idx]

    dg_selector = DGCPGuardSelector(alpha)
    dg_idx, dg_diagnostics = dg_selector.select(candidates, data["sel_y"], x_sel=sel_features)
    dg_selected = candidates[dg_idx]

    selected_adj_scores = selected.score_batch(data["adj_y"], features=adj_features).reshape(-1)
    ordinary_q = split_cp_quantile(selected_adj_scores, alpha)
    ordinary_metrics = eval_candidate(selected, data["test_y"], test_features, data["test_mask"], ordinary_q, alpha)

    bonf = BonferroniCalibrator(alpha).fit(candidates, data["ref_y"], x_ref=ref_features)
    bonf_q = bonf.threshold_for_candidate(selected_idx)
    bonf_metrics = eval_candidate(selected, data["test_y"], test_features, data["test_mask"], bonf_q, alpha)

    minp = MinPSelectionAwareCalibrator(alpha).fit(
        candidates, data["ref_y"], data["adj_y"], x_ref=ref_features, x_adj=adj_features
    )
    minp_q = minp.threshold_for_candidate(selected_idx)
    minp_metrics = eval_candidate(selected, data["test_y"], test_features, data["test_mask"], minp_q, alpha)
    minp_adj_metrics = eval_candidate(
        selected,
        data["adj_y"],
        adj_features,
        np.isfinite(data["adj_y"]),
        minp_q,
        alpha,
    )
    bonf_adj_metrics = eval_candidate(
        selected,
        data["adj_y"],
        adj_features,
        np.isfinite(data["adj_y"]),
        bonf_q,
        alpha,
    )
    hybrid_fallback_reasons = []
    if minp.tau < alpha / max(len(candidates), 1):
        hybrid_fallback_reasons.append("tau_below_alpha_over_K")
    if minp_adj_metrics["coverage"] < 1.0 - alpha:
        hybrid_fallback_reasons.append("minp_adj_undercoverage")
    if minp_adj_metrics["winkler"] > bonf_adj_metrics["winkler"]:
        hybrid_fallback_reasons.append("minp_adj_winkler_worse_than_bonf")
    hybrid_uses_bonf = bool(hybrid_fallback_reasons)
    hybrid_q = bonf_q if hybrid_uses_bonf else minp_q
    hybrid_metrics = bonf_metrics if hybrid_uses_bonf else minp_metrics

    dg_minp_q = minp.threshold_for_candidate(dg_idx)
    dg_minp_metrics = eval_candidate(dg_selected, data["test_y"], test_features, data["test_mask"], dg_minp_q, alpha)

    k = len(candidates)
    rec = {
        "experiment": "sac_cp_timeseries",
        "cell": f"{dataset}/{model}",
        "dataset": dataset,
        "model": model,
        "path": str(residual_path),
        "alpha": alpha,
        "family": family,
        "selector": selector_name,
        "K": k,
        "candidate_names": [c.name for c in candidates],
        "selected_candidate": selected.name,
        "diagnostic_selected_candidate": dg_selected.name,
        "candidate_score_correlation": mean_abs_score_correlation(candidates, data["ref_y"], ref_features),
        "tau_minp": float(minp.tau),
        "alpha_over_K": float(alpha / max(k, 1)),
        "tau_over_alphaK": float(minp.tau / (alpha / max(k, 1))) if k else float("nan"),
        "ordinary_selected_cp": method_record("ordinary_selected_cp", ordinary_q, ordinary_metrics),
        "bonferroni_sac_same_selector": method_record("bonferroni_sac_same_selector", bonf_q, bonf_metrics),
        "minp_sac_same_selector": method_record("minp_sac_same_selector", minp_q, minp_metrics),
        "hybrid_sac_same_selector": method_record("hybrid_sac_same_selector", hybrid_q, hybrid_metrics),
        "hybrid_diagnostics": {
            "fallback_to_bonferroni": hybrid_uses_bonf,
            "fallback_reasons": hybrid_fallback_reasons,
            "minp_adj_coverage": minp_adj_metrics["coverage"],
            "bonf_adj_coverage": bonf_adj_metrics["coverage"],
            "minp_adj_winkler": minp_adj_metrics["winkler"],
            "bonf_adj_winkler": bonf_adj_metrics["winkler"],
            "minp_adj_width": minp_adj_metrics["width"],
            "bonf_adj_width": bonf_adj_metrics["width"],
        },
        "minp_sac_diagnostic_selector": method_record(
            "minp_sac_diagnostic_selector", dg_minp_q, dg_minp_metrics
        ),
        "same_candidate_width_reduction": float(bonf_metrics["width"] - minp_metrics["width"]),
        "rescp_reported_target": target,
        "split": data["split_sizes"],
        "selection_diagnostics": selection_diagnostics,
        "diagnostic_selection_diagnostics": dg_diagnostics,
    }
    if target:
        rec["pass_vs_rescp"] = {}
        for method_key in [
            "ordinary_selected_cp",
            "bonferroni_sac_same_selector",
            "minp_sac_same_selector",
            "hybrid_sac_same_selector",
            "minp_sac_diagnostic_selector",
        ]:
            m = rec[method_key]
            rec["pass_vs_rescp"][method_key] = {
                "delta_cov": abs(m["delta_cov"]) <= target["dcov_abs"],
                "width": m["width"] <= target["width"],
                "winkler": m["winkler"] <= target["winkler"],
                "all": (
                    abs(m["delta_cov"]) <= target["dcov_abs"]
                    and m["width"] <= target["width"]
                    and m["winkler"] <= target["winkler"]
                ),
            }
    return rec


def summarize_records(records):
    methods = [
        "ordinary_selected_cp",
        "bonferroni_sac_same_selector",
        "minp_sac_same_selector",
        "hybrid_sac_same_selector",
        "minp_sac_diagnostic_selector",
    ]
    def is_valid(record, metrics):
        return metrics["coverage"] >= 1.0 - float(record["alpha"])

    summary = {"num_records": len(records), "method_summary": {}, "rows": records}
    for method in methods:
        vals = [r[method] for r in records]
        summary["method_summary"][method] = {
            "mean_coverage": float(np.mean([v["coverage"] for v in vals])),
            "mean_width": float(np.mean([v["width"] for v in vals])),
            "mean_winkler": float(np.mean([v["winkler"] for v in vals])),
            "valid_cells": int(sum(is_valid(r, r[method]) for r in records)),
            "pass_vs_rescp": int(sum(r.get("pass_vs_rescp", {}).get(method, {}).get("all", False) for r in records)),
        }
    summary["ordinary_undercover_cells"] = int(
        sum(not is_valid(r, r["ordinary_selected_cp"]) for r in records)
    )
    summary["bonferroni_valid_cells"] = int(
        sum(is_valid(r, r["bonferroni_sac_same_selector"]) for r in records)
    )
    summary["minp_valid_cells"] = int(sum(is_valid(r, r["minp_sac_same_selector"]) for r in records))
    summary["hybrid_valid_cells"] = int(
        sum(is_valid(r, r["hybrid_sac_same_selector"]) for r in records)
    )
    summary["hybrid_fallback_cells"] = int(
        sum(r.get("hybrid_diagnostics", {}).get("fallback_to_bonferroni", False) for r in records)
    )
    summary["minp_narrower_than_bonf_cells"] = int(sum(r["same_candidate_width_reduction"] > 0 for r in records))
    summary["hybrid_narrower_than_bonf_cells"] = int(
        sum(r["hybrid_sac_same_selector"]["width"] < r["bonferroni_sac_same_selector"]["width"] for r in records)
    )
    summary["minp_better_winkler_than_bonf_cells"] = int(
        sum(r["minp_sac_same_selector"]["winkler"] < r["bonferroni_sac_same_selector"]["winkler"] for r in records)
    )
    summary["hybrid_better_winkler_than_bonf_cells"] = int(
        sum(r["hybrid_sac_same_selector"]["winkler"] < r["bonferroni_sac_same_selector"]["winkler"] for r in records)
    )
    summary["hybrid_width_reduction_vs_bonferroni"] = float(
        np.mean([
            r["bonferroni_sac_same_selector"]["width"] - r["hybrid_sac_same_selector"]["width"]
            for r in records
        ])
    )
    summary["hybrid_winkler_reduction_vs_bonferroni"] = float(
        np.mean([
            r["bonferroni_sac_same_selector"]["winkler"] - r["hybrid_sac_same_selector"]["winkler"]
            for r in records
        ])
    )
    summary["mean_tau_over_alphaK"] = float(np.mean([r["tau_over_alphaK"] for r in records]))
    summary["mean_candidate_score_correlation"] = float(np.mean([r["candidate_score_correlation"] for r in records]))
    return summary


def write_main_outputs(summary, summary_json, summary_md, table_tex, latex_includes):
    Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# SAC-CP Time-Series 12-Cell Summary", ""]
    lines.append(
        f"Ordinary under-cover cells: {summary['ordinary_undercover_cells']}/{summary['num_records']}."
    )
    lines.append(
        f"Bonferroni valid cells: {summary['bonferroni_valid_cells']}/{summary['num_records']}."
    )
    lines.append(f"Min-p valid cells: {summary['minp_valid_cells']}/{summary['num_records']}.")
    lines.append(
        f"Min-p narrower than same-candidate Bonferroni: {summary['minp_narrower_than_bonf_cells']}/{summary['num_records']}."
    )
    lines.append(
        f"Min-p better Winkler than Bonferroni: {summary['minp_better_winkler_than_bonf_cells']}/{summary['num_records']}."
    )
    lines.append("")
    lines.append(
        "| Cell | Selected | Corr | tau/(alpha/K) | Ordinary Cov | Ordinary Winkler | Bonf Cov | Bonf Winkler | Min-p Cov | Min-p Winkler | RESCP Winkler |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary["rows"]:
        target = r.get("rescp_reported_target") or {}
        lines.append(
            "| {cell} | {sel} | {corr:.3g} | {tau:.3g} | {oc:.4f} | {ow:.4g} | {bc:.4f} | {bw:.4g} | {mc:.4f} | {mw:.4g} | {rw:.4g} |".format(
                cell=r["cell"],
                sel=r["selected_candidate"],
                corr=r["candidate_score_correlation"],
                tau=r["tau_over_alphaK"],
                oc=r["ordinary_selected_cp"]["coverage"],
                ow=r["ordinary_selected_cp"]["winkler"],
                bc=r["bonferroni_sac_same_selector"]["coverage"],
                bw=r["bonferroni_sac_same_selector"]["winkler"],
                mc=r["minp_sac_same_selector"]["coverage"],
                mw=r["minp_sac_same_selector"]["winkler"],
                rw=target.get("winkler", float("nan")),
            )
        )
    lines.append("")
    lines.append("## Method Summary")
    lines.append("| Method | Mean Cov | Mean Width | Mean Winkler | Valid Cells | Pass vs RESCP |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method, row in summary["method_summary"].items():
        lines.append(
            f"| {method} | {row['mean_coverage']:.4f} | {row['mean_width']:.4g} | "
            f"{row['mean_winkler']:.4g} | {row['valid_cells']} | {row['pass_vs_rescp']} |"
        )
    Path(summary_md).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Cell & Ord. Cov. & Ord. Wink. & Bonf. Cov. & Bonf. Wink. & Min-p Cov. & Min-p Wink. & RESCP Wink. \\\\",
        "\\midrule",
    ]
    for r in summary["rows"]:
        target = r.get("rescp_reported_target") or {}
        tex.append(
            f"{r['cell']} & {r['ordinary_selected_cp']['coverage']:.3f} & {r['ordinary_selected_cp']['winkler']:.4g} & "
            f"{r['bonferroni_sac_same_selector']['coverage']:.3f} & {r['bonferroni_sac_same_selector']['winkler']:.4g} & "
            f"{r['minp_sac_same_selector']['coverage']:.3f} & {r['minp_sac_same_selector']['winkler']:.4g} & "
            f"{target.get('winkler', float('nan')):.4g} \\\\"
        )
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    Path(table_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(table_tex).write_text("\n".join(tex) + "\n", encoding="utf-8")
    Path(latex_includes).parent.mkdir(parents=True, exist_ok=True)
    Path(latex_includes).write_text("\\input{figures/table_sac_cp_timeseries_12cell.tex}\n", encoding="utf-8")


def write_hybrid_outputs(summary, summary_json, summary_md, table_tex):
    Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Hybrid SAC-CP 12-Cell Summary", ""]
    lines.append(f"Ordinary valid cells: {summary['method_summary']['ordinary_selected_cp']['valid_cells']}/{summary['num_records']}.")
    lines.append(f"Bonferroni valid cells: {summary['bonferroni_valid_cells']}/{summary['num_records']}.")
    lines.append(f"Min-p valid cells: {summary['minp_valid_cells']}/{summary['num_records']}.")
    lines.append(f"Hybrid valid cells: {summary['hybrid_valid_cells']}/{summary['num_records']}.")
    lines.append(f"Hybrid fallback cells: {summary['hybrid_fallback_cells']}/{summary['num_records']}.")
    lines.append(
        f"Hybrid mean width reduction vs Bonferroni: {summary['hybrid_width_reduction_vs_bonferroni']:.4g}."
    )
    lines.append(
        f"Hybrid mean Winkler reduction vs Bonferroni: {summary['hybrid_winkler_reduction_vs_bonferroni']:.4g}."
    )
    lines.append("")
    lines.append(
        "| Cell | Fallback? | Reasons | Ordinary Cov | Bonf Cov | Min-p Cov | Hybrid Cov | Bonf W | Min-p W | Hybrid W | Bonf Winkler | Min-p Winkler | Hybrid Winkler |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary["rows"]:
        diag = r.get("hybrid_diagnostics", {})
        reasons = ",".join(diag.get("fallback_reasons", [])) or "-"
        lines.append(
            "| {cell} | {fb} | {reasons} | {oc:.4f} | {bc:.4f} | {mc:.4f} | {hc:.4f} | "
            "{bw:.4g} | {mw:.4g} | {hw:.4g} | {bwi:.4g} | {mwi:.4g} | {hwi:.4g} |".format(
                cell=r["cell"],
                fb=diag.get("fallback_to_bonferroni", False),
                reasons=reasons,
                oc=r["ordinary_selected_cp"]["coverage"],
                bc=r["bonferroni_sac_same_selector"]["coverage"],
                mc=r["minp_sac_same_selector"]["coverage"],
                hc=r["hybrid_sac_same_selector"]["coverage"],
                bw=r["bonferroni_sac_same_selector"]["width"],
                mw=r["minp_sac_same_selector"]["width"],
                hw=r["hybrid_sac_same_selector"]["width"],
                bwi=r["bonferroni_sac_same_selector"]["winkler"],
                mwi=r["minp_sac_same_selector"]["winkler"],
                hwi=r["hybrid_sac_same_selector"]["winkler"],
            )
        )
    lines.append("")
    lines.append("## Method Summary")
    lines.append("| Method | Mean Cov | Mean Width | Mean Winkler | Valid Cells | Pass vs RESCP |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method in [
        "ordinary_selected_cp",
        "bonferroni_sac_same_selector",
        "minp_sac_same_selector",
        "hybrid_sac_same_selector",
    ]:
        row = summary["method_summary"][method]
        lines.append(
            f"| {method} | {row['mean_coverage']:.4f} | {row['mean_width']:.4g} | "
            f"{row['mean_winkler']:.4g} | {row['valid_cells']} | {row['pass_vs_rescp']} |"
        )
    Path(summary_md).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Cell & Ord. Cov. & Bonf. Cov. & Min-p Cov. & Hybrid Cov. & Bonf. W & Min-p W & Hybrid W & Fallback \\\\",
        "\\midrule",
    ]
    for r in summary["rows"]:
        tex.append(
            f"{r['cell']} & {r['ordinary_selected_cp']['coverage']:.3f} & "
            f"{r['bonferroni_sac_same_selector']['coverage']:.3f} & "
            f"{r['minp_sac_same_selector']['coverage']:.3f} & "
            f"{r['hybrid_sac_same_selector']['coverage']:.3f} & "
            f"{r['bonferroni_sac_same_selector']['width']:.4g} & "
            f"{r['minp_sac_same_selector']['width']:.4g} & "
            f"{r['hybrid_sac_same_selector']['width']:.4g} & "
            f"{str(r.get('hybrid_diagnostics', {}).get('fallback_to_bonferroni', False))} \\\\"
        )
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    Path(table_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(table_tex).write_text("\n".join(tex) + "\n", encoding="utf-8")


def write_ablation_outputs(summary, output_json, output_md, output_tex, group_key):
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = summary["rows"]
    lines = [f"# SAC-CP {group_key.replace('_', ' ').title()} Ablation", ""]
    lines.append(
        f"| {group_key} | K | Ordinary Cov | Bonf Cov | Min-p Cov | Min-p Width | Min-p Winkler | tau/(alpha/K) | Corr |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        ordinary = r["ordinary_selected_cp"]
        bonf = r["bonferroni_sac_same_selector"]
        minp = r["minp_sac_same_selector"]
        lines.append(
            f"| {r[group_key]} | {r['K']} | {ordinary['mean_coverage']:.4f} | "
            f"{bonf['mean_coverage']:.4f} | {minp['mean_coverage']:.4f} | "
            f"{minp['mean_width']:.4g} | {minp['mean_winkler']:.4g} | "
            f"{r['tau_over_alphaK']:.3g} | {r['candidate_score_correlation']:.3g} |"
        )
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        f"{group_key} & K & Ord. Cov. & Bonf. Cov. & Min-p Cov. & Min-p Wink. & $\\tau/(\\alpha/K)$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        ordinary = r["ordinary_selected_cp"]
        bonf = r["bonferroni_sac_same_selector"]
        minp = r["minp_sac_same_selector"]
        tex.append(
            f"{r[group_key]} & {r['K']} & {ordinary['mean_coverage']:.3f} & "
            f"{bonf['mean_coverage']:.3f} & {minp['mean_coverage']:.3f} & "
            f"{minp['mean_winkler']:.4g} & {r['tau_over_alphaK']:.2f} \\\\"
        )
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    Path(output_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(output_tex).write_text("\n".join(tex) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run SAC-CP time-series main experiments and ablations.")
    parser.add_argument(
        "--root",
        default="tmp_rescp_official/extract/reservoir-conformal-prediction-dev-main/reservoir_conformal_prediction/logs/base",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--target-table", default="table1_alpha01")
    parser.add_argument(
        "--mode",
        choices=["main", "hybrid", "candidate_ablation", "selector_ablation", "all"],
        default="all",
    )
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    targets = get_targets(args.target_table)
    residual_paths = find_residual_paths(args.root)
    selected_cells = CELLS
    if args.only:
        wanted = args.only.lower().replace("transformer", "transf")
        selected_cells = [c for c in CELLS if canonical_cell(*c) == wanted]
        if not selected_cells:
            raise ValueError(f"Unknown --only cell: {args.only}")

    missing = [cell for cell in selected_cells if cell not in residual_paths]
    if missing:
        raise FileNotFoundError(f"Missing residual artifacts for cells: {missing}")

    if args.mode in {"main", "all"}:
        records = []
        for cell in selected_cells:
            print(f"[main] running {cell[0]}/{cell[1]}", flush=True)
            records.append(run_cell(residual_paths[cell], args.alpha, "full", "validation_score", targets.get(cell)))
            print(f"[main] done {cell[0]}/{cell[1]}", flush=True)
        summary = summarize_records(records)
        write_main_outputs(
            summary,
            "results/sac_cp_timeseries_12cell_summary.json",
            "docs/sac_cp_timeseries_12cell_summary.md",
            "figures/table_sac_cp_timeseries_12cell.tex",
            "figures/sac_cp_timeseries_latex_includes.tex",
        )
        print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))

    if args.mode == "hybrid":
        records = []
        for cell in selected_cells:
            print(f"[hybrid] running {cell[0]}/{cell[1]}", flush=True)
            records.append(run_cell(residual_paths[cell], args.alpha, "full", "validation_score", targets.get(cell)))
            print(f"[hybrid] done {cell[0]}/{cell[1]}", flush=True)
        summary = summarize_records(records)
        write_hybrid_outputs(
            summary,
            "results/sac_cp_hybrid_12cell_summary.json",
            "docs/sac_cp_hybrid_12cell_summary.md",
            "figures/table_sac_cp_hybrid_12cell.tex",
        )
        print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))

    if args.mode in {"candidate_ablation", "all"}:
        rows = []
        for family in ["small", "full", "full_diagnostic"]:
            fam = "full" if family == "full_diagnostic" else family
            selector = "diagnostic" if family == "full_diagnostic" else "validation_score"
            recs = []
            for cell in selected_cells:
                print(f"[candidate_ablation:{family}] running {cell[0]}/{cell[1]}", flush=True)
                recs.append(run_cell(residual_paths[cell], args.alpha, fam, selector, targets.get(cell)))
                print(f"[candidate_ablation:{family}] done {cell[0]}/{cell[1]}", flush=True)
            s = summarize_records(recs)
            row = {
                "family": family,
                "K": recs[0]["K"],
                "ordinary_selected_cp": s["method_summary"]["ordinary_selected_cp"],
                "bonferroni_sac_same_selector": s["method_summary"]["bonferroni_sac_same_selector"],
                "minp_sac_same_selector": s["method_summary"]["minp_sac_same_selector"],
                "tau_over_alphaK": s["mean_tau_over_alphaK"],
                "candidate_score_correlation": s["mean_candidate_score_correlation"],
            }
            rows.append(row)
        summary = {"num_families": len(rows), "rows": rows}
        write_ablation_outputs(
            summary,
            "results/sac_cp_candidate_family_ablation.json",
            "docs/sac_cp_candidate_family_ablation.md",
            "figures/table_sac_cp_candidate_family_ablation.tex",
            "family",
        )
        print(json.dumps(summary, indent=2))

    if args.mode in {"selector_ablation", "all"}:
        rows = []
        for selector in ["validation_score", "width", "winkler", "diagnostic"]:
            recs = []
            for cell in selected_cells:
                print(f"[selector_ablation:{selector}] running {cell[0]}/{cell[1]}", flush=True)
                recs.append(run_cell(residual_paths[cell], args.alpha, "full", selector, targets.get(cell)))
                print(f"[selector_ablation:{selector}] done {cell[0]}/{cell[1]}", flush=True)
            s = summarize_records(recs)
            row = {
                "selector": selector,
                "K": recs[0]["K"],
                "ordinary_selected_cp": s["method_summary"]["ordinary_selected_cp"],
                "bonferroni_sac_same_selector": s["method_summary"]["bonferroni_sac_same_selector"],
                "minp_sac_same_selector": s["method_summary"]["minp_sac_same_selector"],
                "tau_over_alphaK": s["mean_tau_over_alphaK"],
                "candidate_score_correlation": s["mean_candidate_score_correlation"],
            }
            rows.append(row)
        summary = {"num_selectors": len(rows), "rows": rows}
        write_ablation_outputs(
            summary,
            "results/sac_cp_selector_ablation.json",
            "docs/sac_cp_selector_ablation.md",
            "figures/table_sac_cp_selector_ablation.tex",
            "selector",
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
