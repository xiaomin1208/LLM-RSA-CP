import numpy as np

from .candidate_score import CandidateScore, FeatureKNNResidualScaleScore, is_per_candidate_features
from .metrics import interval_metrics, weighted_quantile


def _as_feature_matrix(features):
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _finite_flat(values):
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    return vals[np.isfinite(vals)]


def _safe_weights(weights, n):
    if weights is None:
        return None
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size != n:
        return None
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return None
    return weights / total


def _beta_grid(alpha, steps):
    steps = max(int(steps), 1)
    return np.linspace(0.0, float(alpha), steps + 1)


class SignedQuantileIntervalScore(CandidateScore):
    def __init__(
        self,
        name,
        weight_mode="global",
        k=None,
        window=None,
        decay=None,
        beta_search=True,
        beta_steps=10,
        alpha=0.1,
        reservoir_dim=64,
        reservoir_radius=0.9,
        reservoir_leak=0.5,
        reservoir_temperature=0.2,
        seed=0,
    ):
        super().__init__(name=name)
        self.weight_mode = str(weight_mode)
        self.k = None if k is None else int(k)
        self.window = None if window is None else int(window)
        self.decay = None if decay is None else float(decay)
        self.beta_search = bool(beta_search)
        self.beta_steps = int(beta_steps)
        self.alpha = float(alpha)
        self.reservoir_dim = int(reservoir_dim)
        self.reservoir_radius = float(reservoir_radius)
        self.reservoir_leak = float(reservoir_leak)
        self.reservoir_temperature = float(reservoir_temperature)
        self.seed = int(seed)
        self.history = None
        self.feature_history = None
        self.feature_mean = None
        self.feature_scale = None
        self.global_low = 0.0
        self.global_high = 0.0
        self.global_beta = 0.0
        self._reservoir_W = None
        self._reservoir_U = None
        self._reservoir_states = None
        self._fallback_scale = 1.0
        self._interval_cache = {}

    def fit(self, residuals, features=None):
        self.history = np.asarray(residuals, dtype=np.float64).copy()
        vals = _finite_flat(self.history)
        self._fallback_scale = float(np.std(vals)) if vals.size else 1.0
        if not np.isfinite(self._fallback_scale) or self._fallback_scale <= 1e-12:
            self._fallback_scale = 1.0
        if features is not None:
            feat = _as_feature_matrix(features)
            self.feature_history = feat.copy()
            self.feature_mean = np.nanmean(feat, axis=0)
            self.feature_mean = np.where(np.isfinite(self.feature_mean), self.feature_mean, 0.0)
            self.feature_scale = np.nanstd(feat, axis=0)
            self.feature_scale = np.where(self.feature_scale > 1e-8, self.feature_scale, 1.0)
        if self.weight_mode.startswith("reservoir"):
            self._fit_reservoir()
        self.global_low, self.global_high, self.global_beta = self._best_interval(vals, None)
        self._interval_cache = {}
        return self

    def _fit_reservoir(self):
        x = np.nanmedian(self.history, axis=1)
        x = x[np.isfinite(x)]
        if x.size == 0:
            self._reservoir_states = None
            return
        rng = np.random.default_rng(self.seed)
        W = rng.normal(size=(self.reservoir_dim, self.reservoir_dim))
        eig = np.max(np.abs(np.linalg.eigvals(W)))
        W = W / max(float(eig), 1e-6) * self.reservoir_radius
        U = rng.normal(scale=1.0 / np.sqrt(self.reservoir_dim), size=(self.reservoir_dim,))
        states = np.zeros((x.size, self.reservoir_dim), dtype=np.float64)
        h = np.zeros(self.reservoir_dim, dtype=np.float64)
        scale = np.std(x)
        scale = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
        for i, value in enumerate(x / scale):
            pre = W @ h + U * value
            h = (1.0 - self.reservoir_leak) * h + self.reservoir_leak * np.tanh(pre)
            states[i] = h
        norm = np.linalg.norm(states, axis=1, keepdims=True)
        self._reservoir_W = W
        self._reservoir_U = U
        self._reservoir_states = states / np.maximum(norm, 1e-12)

    def _normalized_features(self, features):
        feat = _as_feature_matrix(features)
        if self.feature_mean is None:
            return np.nan_to_num(feat, nan=0.0)
        return (np.nan_to_num(feat, nan=0.0) - self.feature_mean) / self.feature_scale

    def _candidate_history(self):
        hist = self.history
        feat = self.feature_history
        states = self._reservoir_states
        if self.window is not None and self.window > 0:
            w = min(self.window, hist.shape[0])
            hist = hist[-w:]
            feat = None if feat is None else feat[-w:]
            states = None if states is None else states[-min(w, states.shape[0]):]
        return hist, feat, states

    def _time_weights(self, n):
        if self.decay is None or self.decay <= 0:
            return np.ones(n, dtype=np.float64)
        age = np.arange(n - 1, -1, -1, dtype=np.float64)
        return np.exp(-age / self.decay)

    def _feature_weights(self, eval_feature, hist_feat):
        if eval_feature is None or hist_feat is None or self.k is None:
            return np.ones(hist_feat.shape[0] if hist_feat is not None else self.history.shape[0])
        q = self._normalized_features(eval_feature[None, :])[0]
        h = (np.nan_to_num(hist_feat, nan=0.0) - self.feature_mean) / self.feature_scale
        dist = np.nanmean((h - q[None, :]) ** 2, axis=1)
        dist = np.where(np.isfinite(dist), dist, np.inf)
        k = min(max(self.k, 1), dist.size)
        idx = np.argpartition(dist, kth=k - 1)[:k]
        weights = np.zeros(dist.size, dtype=np.float64)
        weights[idx] = 1.0
        return weights

    def _knn_intervals_batch(self, residual_shape, features):
        n, d = residual_shape
        if features is None or self.feature_history is None or self.k is None:
            low, high, _ = self._best_interval(self.history.reshape(-1), None)
            return np.full((n, d), low), np.full((n, d), high)
        hist, hist_feat, _ = self._candidate_history()
        h = (np.nan_to_num(hist_feat, nan=0.0) - self.feature_mean) / self.feature_scale
        q = self._normalized_features(features)
        hist_sq = np.sum(h * h, axis=1)
        k = min(max(self.k, 1), h.shape[0])
        lows = np.zeros((n, d), dtype=np.float64)
        highs = np.zeros((n, d), dtype=np.float64)
        chunk = 256
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            qq = q[start:end]
            q_sq = np.sum(qq * qq, axis=1, keepdims=True)
            dist = q_sq + hist_sq[None, :] - 2.0 * qq @ h.T
            dist = np.where(np.isfinite(dist), dist, np.inf)
            idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
            for row, hist_idx in enumerate(idx):
                vals = hist[hist_idx].reshape(-1)
                low, high, _ = self._best_interval(vals, None)
                lows[start + row, :] = low
                highs[start + row, :] = high
        return lows, highs

    def _reservoir_weights(self, query_index, states):
        if states is None or states.shape[0] == 0:
            return np.ones(self.history.shape[0], dtype=np.float64)
        q_idx = min(max(int(query_index), 0), states.shape[0] - 1)
        q = states[q_idx]
        sim = states @ q
        sim = sim / max(self.reservoir_temperature, 1e-6)
        sim = sim - np.max(sim[np.isfinite(sim)]) if np.any(np.isfinite(sim)) else sim
        weights = np.exp(np.where(np.isfinite(sim), sim, -np.inf))
        return weights

    def _weights_for_query(self, query_idx, eval_feature=None):
        hist, hist_feat, states = self._candidate_history()
        n = hist.shape[0]
        weights = np.ones(n, dtype=np.float64)
        if self.weight_mode in {"time_decay", "decay"} or "decay" in self.weight_mode:
            weights *= self._time_weights(n)
        if self.weight_mode.startswith("knn"):
            weights *= self._feature_weights(eval_feature, hist_feat)
        if self.weight_mode.startswith("reservoir"):
            weights *= self._reservoir_weights(query_idx, states)
        return hist, _safe_weights(weights, n)

    def _best_interval(self, values, weights):
        values = _finite_flat(values)
        if values.size == 0:
            return -self._fallback_scale, self._fallback_scale, 0.0
        weights = _safe_weights(weights, values.size)
        best = None
        betas = _beta_grid(self.alpha, self.beta_steps) if self.beta_search else np.array([self.alpha / 2.0])
        for beta in betas:
            lo_q = min(max(beta, 0.0), 1.0)
            hi_q = min(max(1.0 - self.alpha + beta, 0.0), 1.0)
            low = weighted_quantile(values, lo_q, weights)
            high = weighted_quantile(values, hi_q, weights)
            if high < low:
                low, high = high, low
            width = high - low
            key = (width, abs(beta - self.alpha / 2.0))
            if best is None or key < best[0]:
                best = (key, low, high, beta)
        return float(best[1]), float(best[2]), float(best[3])

    def _interval_arrays(self, residual_shape, features=None):
        cache_key = (int(residual_shape[0]), int(residual_shape[1]), id(features))
        if cache_key in self._interval_cache:
            return self._interval_cache[cache_key]
        n, d = residual_shape
        lows = np.zeros((n, d), dtype=np.float64)
        highs = np.zeros((n, d), dtype=np.float64)
        if self.weight_mode == "global" and self.window is None and self.decay is None:
            lows[:] = self.global_low
            highs[:] = self.global_high
            self._interval_cache[cache_key] = (lows, highs)
            return lows, highs
        if not self.weight_mode.startswith("knn") and not self.weight_mode.startswith("reservoir"):
            hist, weights = self._weights_for_query(0, None)
            vals = hist.reshape(-1)
            flat_weights = None if weights is None else np.repeat(weights, hist.shape[1])
            low, high, _ = self._best_interval(vals, flat_weights)
            lows[:] = low
            highs[:] = high
            self._interval_cache[cache_key] = (lows, highs)
            return lows, highs
        if self.weight_mode.startswith("knn"):
            lows, highs = self._knn_intervals_batch(residual_shape, features)
            self._interval_cache[cache_key] = (lows, highs)
            return lows, highs
        eval_feat = None if features is None else _as_feature_matrix(features)
        for i in range(n):
            feat_i = None if eval_feat is None else eval_feat[i]
            hist, weights = self._weights_for_query(i, feat_i)
            vals = hist.reshape(-1)
            flat_weights = None if weights is None else np.repeat(weights, hist.shape[1])
            low, high, _ = self._best_interval(vals, flat_weights)
            lows[i, :] = low
            highs[i, :] = high
        self._interval_cache[cache_key] = (lows, highs)
        return lows, highs

    def interval(self, residual_shape, threshold, features=None):
        low, high = self._interval_arrays(residual_shape, features=features)
        expand = max(float(threshold), 0.0) * self._fallback_scale
        return low - expand, high + expand

    def score_batch(self, residuals, features=None):
        residuals = np.asarray(residuals, dtype=np.float64)
        low, high = self._interval_arrays(residuals.shape, features=features)
        miss = np.maximum(low - residuals, residuals - high)
        return np.maximum(miss, 0.0) / max(self._fallback_scale, 1e-12)


class CoverageConstrainedWinklerSelector:
    def __init__(self, alpha, tol=0.01, penalty=1e4):
        self.alpha = float(alpha)
        self.tol = float(tol)
        self.penalty = float(penalty)

    def select(self, candidates, d_sel, x_sel=None):
        diagnostics = []
        best_idx = 0
        best_key = None
        per_candidate_features = is_per_candidate_features(x_sel, len(candidates))
        target = 1.0 - self.alpha - self.tol
        for idx, candidate in enumerate(candidates):
            features = x_sel[idx] if per_candidate_features else x_sel
            low, high = candidate.interval(d_sel.shape, 0.0, features=features)
            metrics = interval_metrics(d_sel, low, high, self.alpha)
            shortfall = max(0.0, target - metrics["coverage"])
            key = (shortfall > 0.0, metrics["winkler"] + self.penalty * shortfall, metrics["width"])
            diagnostics.append({"name": candidate.name, "q": 0.0, "metrics": metrics})
            if best_key is None or key < best_key:
                best_idx, best_key = idx, key
        return best_idx, diagnostics


def build_rescp_competitive_candidates(alpha=0.1, include_reservoir=False):
    base = [
        SignedQuantileIntervalScore("signed_quantile_scp", "global", alpha=alpha, beta_search=True),
        SignedQuantileIntervalScore("signed_quantile_nexcp", "time_decay", decay=500, alpha=alpha, beta_search=True),
        SignedQuantileIntervalScore("signed_quantile_recent500", "time_decay", window=500, decay=200, alpha=alpha, beta_search=True),
        SignedQuantileIntervalScore("signed_quantile_recent1000", "time_decay", window=1000, decay=500, alpha=alpha, beta_search=True),
        SignedQuantileIntervalScore("knn_signed_quantile_k50", "knn", k=50, window=1000, alpha=alpha, beta_search=True),
        SignedQuantileIntervalScore("knn_signed_quantile_k100", "knn", k=100, window=1000, alpha=alpha, beta_search=True),
        SignedQuantileIntervalScore("knn_signed_quantile_k200", "knn", k=200, window=1000, alpha=alpha, beta_search=True),
    ]
    if not include_reservoir:
        return base
    base.extend([
        SignedQuantileIntervalScore("reservoir_signed_quantile", "reservoir", alpha=alpha, beta_search=True),
        SignedQuantileIntervalScore(
            "reservoir_decay_signed_quantile", "reservoir_decay", decay=500, alpha=alpha, beta_search=True
        ),
        SignedQuantileIntervalScore(
            "reservoir_window_signed_quantile", "reservoir", window=1000, alpha=alpha, beta_search=True
        ),
        SignedQuantileIntervalScore(
            "reservoir_decay_window_signed_quantile",
            "reservoir_decay",
            window=1000,
            decay=500,
            alpha=alpha,
            beta_search=True,
        ),
    ])
    return base


def build_rescp_fast_candidates(alpha=0.1):
    return [
        SignedQuantileIntervalScore("signed_quantile_scp", "global", alpha=alpha, beta_search=True),
        SignedQuantileIntervalScore("signed_quantile_nexcp", "time_decay", decay=200, alpha=alpha, beta_search=True),
        SignedQuantileIntervalScore("signed_quantile_decay500", "time_decay", decay=500, alpha=alpha, beta_search=True),
        SignedQuantileIntervalScore("signed_quantile_recent500", "time_decay", window=500, decay=200, alpha=alpha, beta_search=True),
        SignedQuantileIntervalScore("signed_quantile_recent1000", "time_decay", window=1000, decay=500, alpha=alpha, beta_search=True),
    ]
