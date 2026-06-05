import numpy as np

from .candidate_score import is_per_candidate_features
from .metrics import higher_quantile


class MinPSelectionAwareCalibrator:
    def __init__(self, alpha):
        self.alpha = float(alpha)
        self.tau = None
        self.ref_scores = None
        self.candidate_names = None

    @staticmethod
    def pvalues_from_ref(ref_scores, eval_scores):
        ref = np.asarray(ref_scores, dtype=np.float64)
        ev = np.asarray(eval_scores, dtype=np.float64)
        out = np.full_like(ev, np.nan, dtype=np.float64)
        for h in range(ref.shape[1]):
            ref_h = ref[:, h]
            ref_h = ref_h[np.isfinite(ref_h)]
            if ref_h.size == 0:
                continue
            ev_h = ev[:, h]
            ok = np.isfinite(ev_h)
            out[ok, h] = (1.0 + np.sum(ref_h[None, :] >= ev_h[ok, None], axis=1)) / (ref_h.size + 1.0)
        return out

    def fit(self, candidates, d_ref, d_adj, x_ref=None, x_adj=None):
        self.candidate_names = [c.name for c in candidates]
        ref_per_candidate = is_per_candidate_features(x_ref, len(candidates))
        adj_per_candidate = is_per_candidate_features(x_adj, len(candidates))
        self.ref_scores = np.column_stack([
            c.score_batch(d_ref, features=(x_ref[i] if ref_per_candidate else x_ref)).reshape(-1)
            for i, c in enumerate(candidates)
        ])
        adj_scores = np.column_stack([
            c.score_batch(d_adj, features=(x_adj[i] if adj_per_candidate else x_adj)).reshape(-1)
            for i, c in enumerate(candidates)
        ])
        p_adj = self.pvalues_from_ref(self.ref_scores, adj_scores)
        m_adj = np.nanmin(p_adj, axis=1)
        vals = np.sort(m_adj[np.isfinite(m_adj)])
        n = vals.size
        if n == 0:
            self.tau = 0.0
        else:
            k = int(np.floor(self.alpha * (n + 1)))
            self.tau = 0.0 if k <= 0 else float(vals[min(k - 1, n - 1)])
        return self

    def threshold_for_candidate(self, candidate_index):
        if self.ref_scores is None or self.tau is None:
            raise RuntimeError("Calibrator must be fit before threshold_for_candidate.")
        return higher_quantile(self.ref_scores[:, candidate_index], min(1.0, 1.0 - self.tau))
