import numpy as np

from .candidate_score import is_per_candidate_features
from .metrics import split_cp_quantile


class BonferroniCalibrator:
    def __init__(self, alpha):
        self.alpha = float(alpha)
        self.alpha_per_candidate = None
        self.ref_scores = None

    def fit(self, candidates, d_ref, x_ref=None):
        self.alpha_per_candidate = self.alpha / max(len(candidates), 1)
        ref_per_candidate = is_per_candidate_features(x_ref, len(candidates))
        self.ref_scores = np.column_stack([
            c.score_batch(d_ref, features=(x_ref[i] if ref_per_candidate else x_ref)).reshape(-1)
            for i, c in enumerate(candidates)
        ])
        return self

    def threshold_for_candidate(self, candidate_index):
        if self.ref_scores is None:
            raise RuntimeError("Calibrator must be fit before threshold_for_candidate.")
        return split_cp_quantile(self.ref_scores[:, candidate_index], self.alpha_per_candidate)
