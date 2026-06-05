from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .metrics import weighted_quantile


def _flat_valid(arr):
    vals = np.asarray(arr, dtype=np.float64).reshape(-1)
    return vals[np.isfinite(vals)]


def _mad_scale(vals):
    vals = _flat_valid(vals)
    if vals.size == 0:
        return 1.0
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) / 0.6744897501960817
    if np.isfinite(mad) and mad > 1e-12:
        return float(mad)
    std = np.std(vals)
    return float(std if std > 1e-12 else 1.0)


def _safe_scale(scale, eps):
    scale = np.asarray(scale, dtype=np.float64)
    return np.maximum(scale, eps)


@dataclass
class CandidateScore:
    name: str

    def fit(self, residuals, features=None):
        return self

    def center_scale(self, residuals, features=None):
        raise NotImplementedError

    def score_batch(self, residuals, features=None):
        center, scale = self.center_scale(residuals, features=features)
        return np.abs(np.asarray(residuals, dtype=np.float64) - center) / scale

    def interval(self, residual_shape, threshold, features=None):
        center, scale = self.center_scale(np.full(residual_shape, np.nan), features=features)
        half = float(threshold) * scale
        return center - half, center + half


def is_per_candidate_features(features, n_candidates):
    if not isinstance(features, (list, tuple)):
        return False
    return len(features) == n_candidates


class ResidualScaleScore(CandidateScore):
    def __init__(
        self,
        name,
        center_mode="zero",
        scale_mode="global_mad",
        window: Optional[int] = None,
        decay: Optional[float] = None,
        knn_k: Optional[int] = None,
        eps=1e-8,
    ):
        super().__init__(name=name)
        self.center_mode = center_mode
        self.scale_mode = scale_mode
        self.window = window
        self.decay = decay
        self.knn_k = knn_k
        self.eps = float(eps)
        self.history = None
        self.global_center = 0.0
        self.global_scale = 1.0
        self.node_center = None
        self.node_scale = None

    def fit(self, residuals, features=None):
        hist = np.asarray(residuals, dtype=np.float64)
        self.history = hist.copy()
        vals = _flat_valid(hist)
        self.global_center = float(np.median(vals)) if vals.size else 0.0
        self.global_scale = _mad_scale(vals)
        self.node_center = np.nanmedian(hist, axis=0)
        self.node_center = np.where(np.isfinite(self.node_center), self.node_center, self.global_center)
        self.node_scale = np.array([_mad_scale(hist[:, j]) for j in range(hist.shape[1])], dtype=np.float64)
        self.node_scale = _safe_scale(self.node_scale, self.eps)
        return self

    def _window_values(self):
        hist = self.history
        if hist is None or hist.size == 0:
            return np.array([0.0])
        if self.window is None or self.window <= 0:
            return hist
        return hist[-min(self.window, hist.shape[0]):]

    def _decay_weights(self, n):
        if self.decay is None or self.decay <= 0:
            return None
        age = np.arange(n - 1, -1, -1, dtype=np.float64)
        weights = np.exp(-age / self.decay)
        return weights / max(np.sum(weights), 1e-12)

    def _knn_values(self, residuals):
        # Legacy residual-magnitude retrieval. Prefer FeatureKNNResidualScaleScore
        # when side information x is available; this path should not be used for
        # final evaluation because it depends on the residual being scored.
        hist = self.history
        if hist is None or self.knn_k is None or self.knn_k <= 0:
            return self._window_values()
        current = np.asarray(residuals, dtype=np.float64)
        valid_current = current[np.isfinite(current)]
        target_abs = float(np.median(np.abs(valid_current))) if valid_current.size else float(np.nanmedian(np.abs(hist)))
        hist_abs = np.nanmedian(np.abs(hist), axis=1)
        dist = np.abs(hist_abs - target_abs)
        order = np.argsort(np.where(np.isfinite(dist), dist, np.inf))
        idx = order[: min(self.knn_k, hist.shape[0])]
        return hist[idx]

    def _center_scale_from_values(self, values):
        values = np.asarray(values, dtype=np.float64)
        weights = self._decay_weights(values.shape[0])
        flat = values.reshape(-1)
        flat_weights = None if weights is None else np.repeat(weights, values.shape[1])
        if self.center_mode == "zero":
            center = 0.0
        elif self.center_mode == "global_median":
            center = weighted_quantile(flat, 0.5, flat_weights)
        elif self.center_mode == "node_median":
            center = np.nanmedian(values, axis=0)
            center = np.where(np.isfinite(center), center, self.global_center)
        else:
            center = self.global_center

        if self.scale_mode == "unit":
            scale = 1.0
        elif self.scale_mode == "global_mad":
            loc = 0.0 if np.isscalar(center) else np.nanmedian(center)
            scale = _mad_scale(flat - loc)
        elif self.scale_mode == "global_iqr":
            q25 = weighted_quantile(flat, 0.25, flat_weights)
            q75 = weighted_quantile(flat, 0.75, flat_weights)
            scale = max((q75 - q25) / 1.349, self.eps)
        elif self.scale_mode == "node_mad":
            center_arr = center if not np.isscalar(center) else np.full(values.shape[1], center)
            scale = np.array([_mad_scale(values[:, j] - center_arr[j]) for j in range(values.shape[1])])
        else:
            scale = self.global_scale
        return center, _safe_scale(scale, self.eps)

    def center_scale(self, residuals, features=None):
        if self.scale_mode == "unit" and self.center_mode == "zero":
            shape = np.asarray(residuals).shape
            return np.zeros(shape[1], dtype=np.float64), np.ones(shape[1], dtype=np.float64)
        values = self._knn_values(residuals) if self.knn_k else self._window_values()
        center, scale = self._center_scale_from_values(values)
        if np.isscalar(center):
            center = np.full(np.asarray(residuals).shape[1], float(center), dtype=np.float64)
        if np.isscalar(scale):
            scale = np.full(np.asarray(residuals).shape[1], float(scale), dtype=np.float64)
        return center, _safe_scale(scale, self.eps)


class FeatureKNNResidualScaleScore(ResidualScaleScore):
    def __init__(
        self,
        name,
        k=100,
        window: Optional[int] = None,
        center_mode="node_median",
        scale_mode="node_mad",
        eps=1e-8,
    ):
        super().__init__(
            name=name,
            center_mode=center_mode,
            scale_mode=scale_mode,
            window=window,
            eps=eps,
        )
        self.k = int(k)
        self.feature_history = None
        self.feature_mean = None
        self.feature_scale = None

    @staticmethod
    def _as_feature_matrix(features):
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr

    def fit(self, residuals, features=None):
        super().fit(residuals, features=features)
        if features is None:
            return self
        feat = self._as_feature_matrix(features)
        self.feature_history = feat.copy()
        self.feature_mean = np.nanmean(feat, axis=0)
        self.feature_mean = np.where(np.isfinite(self.feature_mean), self.feature_mean, 0.0)
        self.feature_scale = np.nanstd(feat, axis=0)
        self.feature_scale = np.where(self.feature_scale > self.eps, self.feature_scale, 1.0)
        return self

    def _normalized_features(self, features):
        feat = self._as_feature_matrix(features)
        return (np.nan_to_num(feat, nan=0.0) - self.feature_mean) / self.feature_scale

    def _candidate_history(self):
        hist = self.history
        feat = self.feature_history
        if self.window is not None and self.window > 0:
            w = min(self.window, hist.shape[0])
            hist = hist[-w:]
            feat = feat[-w:]
        return hist, feat

    def center_scale(self, residuals, features=None):
        residuals = np.asarray(residuals, dtype=np.float64)
        if features is None or self.feature_history is None:
            return super().center_scale(residuals, features=features)

        hist, hist_feat = self._candidate_history()
        hist_feat = (np.nan_to_num(hist_feat, nan=0.0) - self.feature_mean) / self.feature_scale
        eval_feat = self._normalized_features(features)
        centers = np.zeros_like(residuals, dtype=np.float64)
        scales = np.ones_like(residuals, dtype=np.float64)
        k = min(max(self.k, 1), hist.shape[0])

        if self.center_mode == "node_median" and self.scale_mode == "node_mad":
            hist_sq = np.sum(hist_feat * hist_feat, axis=1)
            feat_dim = max(hist_feat.shape[1], 1)
            chunk = 256
            for start in range(0, residuals.shape[0], chunk):
                end = min(start + chunk, residuals.shape[0])
                q = eval_feat[start:end]
                q_sq = np.sum(q * q, axis=1, keepdims=True)
                dist = (q_sq + hist_sq[None, :] - 2.0 * q @ hist_feat.T) / feat_dim
                dist = np.where(np.isfinite(dist), dist, np.inf)
                idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
                values = hist[idx]
                center = np.nanmedian(values, axis=1)
                center = np.where(np.isfinite(center), center, self.global_center)
                scale = np.nanmedian(np.abs(values - center[:, None, :]), axis=1) / 0.6744897501960817
                fallback = self.node_scale[None, :]
                scale = np.where(np.isfinite(scale) & (scale > self.eps), scale, fallback)
                centers[start:end] = center
                scales[start:end] = scale
        else:
            for i in range(residuals.shape[0]):
                diff = hist_feat - eval_feat[i][None, :]
                dist = np.sqrt(np.nanmean(diff * diff, axis=1))
                order = np.argsort(np.where(np.isfinite(dist), dist, np.inf))
                values = hist[order[:k]]
                center, scale = self._center_scale_from_values(values)
                if np.isscalar(center):
                    center = np.full(residuals.shape[1], float(center), dtype=np.float64)
                if np.isscalar(scale):
                    scale = np.full(residuals.shape[1], float(scale), dtype=np.float64)
                centers[i] = center
                scales[i] = scale
        return centers, _safe_scale(scales, self.eps)


def build_default_residual_candidates() -> List[ResidualScaleScore]:
    return [
        ResidualScaleScore("standard_abs", center_mode="zero", scale_mode="unit"),
        ResidualScaleScore("global_mad", center_mode="global_median", scale_mode="global_mad"),
        ResidualScaleScore("node_mad", center_mode="node_median", scale_mode="node_mad"),
        ResidualScaleScore("nexcp_decay_200", center_mode="global_median", scale_mode="global_mad", decay=200.0),
        ResidualScaleScore("recent_200", center_mode="global_median", scale_mode="global_iqr", window=200),
        ResidualScaleScore("recent_1000", center_mode="global_median", scale_mode="global_iqr", window=1000),
        FeatureKNNResidualScaleScore("knn_regime_100", k=100, center_mode="node_median", scale_mode="node_mad"),
        FeatureKNNResidualScaleScore("hopcpt_proxy_knn_200", k=200, center_mode="node_median", scale_mode="node_mad"),
        FeatureKNNResidualScaleScore("hopcpt_proxy_recent_knn_200", k=200, window=1000, center_mode="node_median", scale_mode="node_mad"),
    ]
