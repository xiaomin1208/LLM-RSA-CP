import sys
sys.path.insert(0, "scripts")
import run_aci as M
# model-route lgbm (same as before) + FINE-GRAINED shrink scales near 1.0 to give selector candidates around target coverage
LGBM = {
 "tail_safe":        {"n_estimators":400,"learning_rate":0.03, "num_leaves":63,"min_child_samples":40,"reg_lambda":0.5},
 "micro_scale":      {"n_estimators":300,"learning_rate":0.03, "num_leaves":31,"min_child_samples":60,"reg_lambda":1.5},
 "seasonal_compact": {"n_estimators":500,"learning_rate":0.025,"num_leaves":63,"min_child_samples":30,"reg_lambda":0.5},
 "compact_center":   {"n_estimators":300,"learning_rate":0.03, "num_leaves":31,"min_child_samples":50,"reg_lambda":1.0},
}
FINE=[0.80,0.85,0.90,0.95,1.0]
for n,p in M.LLM_CANDIDATE_PROFILES.items():
    if n in LGBM: p["lgbm"]=LGBM[n]
    # add fine scales around 1.0 (keep existing, add missing)
    sc=set(round(s,2) for s in p["shrink_scales"]) | set(FINE)
    p["shrink_scales"]=sorted(sc)
print("[finegrain] patched; example base_shrink scales:", M.LLM_CANDIDATE_PROFILES["base_shrink"]["shrink_scales"], flush=True)
M.main()
