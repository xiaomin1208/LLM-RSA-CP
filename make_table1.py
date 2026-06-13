# -*- coding: utf-8 -*-
"""Assemble the main-experiment results table (alpha=0.1) from the 3-seed LLM-RSA-CP runs.

Reads results/abl_fine_g0{,_seed1,_seed2}.json (ordinary_selected_cp) and prints the
LLM-RSA-CP metrics (ΔCov / PI-Width / Winkler) as mean +/- std over the 3 seeds.

No numbers are hard-coded: every value is computed at run time from the result JSONs
produced by run_main_table.sh. Baseline comparison numbers are NOT included here — pull
them from the respective baselines' own reports when assembling the paper table.

Usage:  python make_table1.py            # reads ./results/
        python make_table1.py RESULTS_DIR
"""
import json, sys, os
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
SEEDS = ["abl_fine_g0", "abl_fine_g0_seed1", "abl_fine_g0_seed2"]
ORDER = ["Solar/RNN", "Solar/Transf", "Solar/ARIMA",
         "Beijing/RNN", "Beijing/Transf", "Beijing/ARIMA",
         "Exchange/RNN", "Exchange/Transf", "Exchange/ARIMA",
         "ACEA/RNN", "ACEA/Transf", "ACEA/ARIMA"]


def fmt(mean, std, micro):
    if micro:
        return f"{mean:.4f}±{std:.4f}" if std >= 5e-5 else f"{mean:.4f}±0.0000"
    return f"{mean:.2f}±{std:.2f}"


# ---- compute the LLM-RSA-CP metrics from the 3 seed JSONs ----
acc = {}
for s in SEEDS:
    d = json.load(open(os.path.join(RES, s + ".json")))
    for r in d["rows"]:
        o = r["ordinary_selected_cp"]
        acc.setdefault(r["cell"], {"cov": [], "wid": [], "w": []})
        acc[r["cell"]]["cov"].append(o["coverage"])
        acc[r["cell"]]["wid"].append(o["width"])
        acc[r["cell"]]["w"].append(o["winkler"])

print("# Main-experiment results — LLM-RSA-CP (alpha=0.1, mean±std over 3 seeds)\n")
print("Computed at run time from results/abl_fine_g0{,_seed1,_seed2}.json.\n")
print("| Dataset | Model | ΔCov | PI-Width | Winkler |")
print("|---|---|---|---|---|")
for cell in ORDER:
    ds, model = cell.split("/")
    a = acc[cell]
    cov = np.array(a["cov"]); wid = np.array(a["wid"]); w = np.array(a["w"])
    micro = wid.mean() < 1.0
    dcov = (cov - 0.9) * 100.0
    print(f"| {ds} | {model} | "
          f"{dcov.mean():+.2f}±{dcov.std():.2f} | "
          f"{fmt(wid.mean(), wid.std(), micro)} | "
          f"{fmt(w.mean(), w.std(), micro)} |")
