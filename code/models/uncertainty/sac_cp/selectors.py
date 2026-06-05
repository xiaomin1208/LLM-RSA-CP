import numpy as np

from .candidate_score import is_per_candidate_features
from .metrics import interval_metrics, split_cp_quantile


class ValidationScoreSelector:
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
            key = (metrics["winkler"], metrics["width"], metrics["dcov_abs_pct"])
            diagnostics.append({"name": candidate.name, "q": q, "metrics": metrics})
            if best_key is None or key < best_key:
                best_idx, best_key = idx, key
        return best_idx, diagnostics


class DGCPGuardSelector:
    def __init__(self, alpha):
        self.alpha = float(alpha)
        self.base_selector = ValidationScoreSelector(alpha)

    def select(self, candidates, d_sel, x_sel=None):
        idx, diagnostics = self.base_selector.select(candidates, d_sel, x_sel=x_sel)
        # Conservative placeholder: prefer node-local score when validation winner over-covers strongly.
        winner = diagnostics[idx]
        if winner["metrics"]["dcov_signed_pct"] > 2.0:
            for j, diag in enumerate(diagnostics):
                if "node" in diag["name"] and diag["metrics"]["dcov_abs_pct"] <= winner["metrics"]["dcov_abs_pct"]:
                    return j, diagnostics
        return idx, diagnostics
