# -*- coding: utf-8 -*-
"""Assemble main-experiment Table 1 (alpha=0.1) from the 3-seed LLM-RSA-CP runs.

Reads results/abl_fine_g0{,_seed1,_seed2}.json (ordinary_selected_cp), computes the
LLM-RSA-CP column as mean +/- std over the 3 seeds, and prints Table 1 (Markdown) with
the fixed baseline columns (taken from the ResCP same-protocol report).

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
BASELINES = ["SPCI", "HopCPT", "CP-QRNN", "RESCQR", "SCP", "NexCP", "ResCP"]

# Fixed baseline values (ResCP same-protocol report): cell -> {method: (dcov, width, winkler)}
# Strings preserve +/-std and undercoverage markers; () marks values excluded from "best" due to undercoverage.
B = {
 "Solar/RNN": {"SPCI":("0.05±0.17","70.41±1.73","151.50±1.18"),"HopCPT":("-1.64±1.18†","60.49±2.10","112.46±9.34"),"CP-QRNN":("-0.26±0.92","55.74±0.98","78.42±0.20"),"RESCQR":("-1.10±0.91†","59.99±1.72","82.76±0.26"),"SCP":("0.37","79.15","171.41"),"NexCP":("1.46","89.56","164.96"),"ResCP":("0.74±0.24","62.25±0.75","104.24±0.79")},
 "Solar/Transf": {"SPCI":("-0.16±0.46","71.36±3.21","152.85±0.67"),"HopCPT":("1.32±0.62","61.49±1.74","107.59±6.07"),"CP-QRNN":("1.37±2.09","55.70±0.95","77.61±0.25"),"RESCQR":("-3.51±16.26‡","59.56±1.59","(82.16±0.32)"),"SCP":("0.35","79.03","169.64"),"NexCP":("1.46","89.21","163.34"),"ResCP":("3.09±0.35","63.34±1.11","103.13±0.58")},
 "Solar/ARIMA": {"SPCI":("0.51±0.36","91.71±1.18","148.86±0.34"),"HopCPT":("2.88±2.13","143.32±7.36","173.49±5.15"),"CP-QRNN":("-0.41±0.42","59.17±0.41","77.34±0.30"),"RESCQR":("-2.03±0.62‡","66.19±0.81","85.38±0.45"),"SCP":("0.16","124.19","215.77"),"NexCP":("1.76","137.53","207.58"),"ResCP":("0.68±0.95","77.17±2.07","110.38±4.03")},
 "Beijing/RNN": {"SPCI":("-1.73±0.67†","67.93±1.43","124.51±2.25"),"HopCPT":("-5.18±12.67§","68.47±13.30","(140.50±43.64)"),"CP-QRNN":("-1.86±2.16†","61.71±4.91","104.03±0.99"),"RESCQR":("-1.21±1.65†","65.53±4.09","105.43±0.85"),"SCP":("-0.32","67.99","126.41"),"NexCP":("0.00","69.79","124.10"),"ResCP":("-0.70±0.77","65.96±2.50","106.07±0.47")},
 "Beijing/Transf": {"SPCI":("-0.98±1.04","69.96±4.53","125.41±1.43"),"HopCPT":("-8.05±16.41§","61.76±14.39","(140.30±36.01)"),"CP-QRNN":("-1.07±0.69†","62.41±1.66","102.81±0.44"),"RESCQR":("-1.43±1.10†","64.41±2.72","105.97±1.21"),"SCP":("-0.41","67.46","126.63"),"NexCP":("0.03","69.64","124.35"),"ResCP":("-0.49±0.59","64.06±1.74","103.64±0.21")},
 "Beijing/ARIMA": {"SPCI":("-0.23±0.40","74.68±1.21","130.59±0.59"),"HopCPT":("-1.37±0.26†","67.78±0.50","122.48±4.36"),"CP-QRNN":("-1.54±0.77†","61.80±1.68","101.84±0.67"),"RESCQR":("-1.42±1.15†","66.01±3.01","107.20±1.21"),"SCP":("-0.24","75.72","135.07"),"NexCP":("-0.16","76.45","132.03"),"ResCP":("0.63±0.22","70.43±0.86","108.75±0.31")},
 "Exchange/RNN": {"SPCI":("2.98±0.65","0.0241±0.0007","0.0287±0.0007"),"HopCPT":("2.75±0.08","0.0404±0.0001","0.0482±0.0001"),"CP-QRNN":("-1.07±2.52†","0.0341±0.0018","0.0461±0.0005"),"RESCQR":("3.18±1.25","0.0383±0.0008","0.0464±0.0005"),"SCP":("2.29","0.0444","0.0517"),"NexCP":("1.64","0.0405","0.0492"),"ResCP":("1.13±0.27","0.0210±0.0001","0.0264±0.0002")},
 "Exchange/Transf": {"SPCI":("4.44±0.35","0.0255±0.0005","0.0300±0.0005"),"HopCPT":("2.98±0.07","0.0399±0.0001","0.0479±0.0001"),"CP-QRNN":("-0.57±1.58","0.0337±0.0016","0.0480±0.0009"),"RESCQR":("0.82±1.89","0.0365±0.0013","0.0475±0.0008"),"SCP":("4.57","0.0544","0.0620"),"NexCP":("3.25","0.0509","0.0602"),"ResCP":("1.46±0.18","0.0229±0.0001","0.0294±0.0001")},
 "Exchange/ARIMA": {"SPCI":("3.49±0.41","0.0242±0.0006","0.0289±0.0003"),"HopCPT":("2.07±0.08","0.0379±0.0000","0.0456±0.0001"),"CP-QRNN":("-1.22±1.78†","0.0330±0.0007","0.0455±0.0003"),"RESCQR":("0.68±1.58","0.0351±0.0009","0.0455±0.0006"),"SCP":("3.08","0.0387","0.0462"),"NexCP":("2.13","0.0356","0.0447"),"ResCP":("0.38±0.41","0.0207±0.0001","0.0268±0.0001")},
 "ACEA/RNN": {"SPCI":("-0.78±1.88","8.99±0.68","14.27±0.19"),"HopCPT":("-2.18±0.00‡","18.90±0.00","27.56±0.00"),"CP-QRNN":("-12.37±8.98§","15.86±1.99","(32.61±5.69)"),"RESCQR":("-18.86±7.44§","15.23±1.96","(34.61±3.53)"),"SCP":("-0.99","19.63","27.60"),"NexCP":("-0.33","20.15","26.83"),"ResCP":("1.56±0.62","9.61±0.26","12.91±0.23")},
 "ACEA/Transf": {"SPCI":("-1.41±1.29†","9.10±0.23","14.58±0.36"),"HopCPT":("-2.51±0.00‡","18.29±0.00","(27.53±0.00)"),"CP-QRNN":("-13.35±9.85§","14.82±2.02","(33.47±7.18)"),"RESCQR":("-26.92±7.68§","13.20±1.47","(39.98±4.27)"),"SCP":("-5.52§","16.53","(29.24)"),"NexCP":("-0.45","20.20","27.47"),"ResCP":("3.54±0.32","10.10±0.16","12.90±0.16")},
 "ACEA/ARIMA": {"SPCI":("1.41±0.90","12.46±0.35","17.36±0.13"),"HopCPT":("-3.58±0.00‡","34.84±0.00","(44.69±0.00)"),"CP-QRNN":("-29.35±11.01§","18.16±2.43","(53.89±9.33)"),"RESCQR":("-27.10±8.65§","17.39±2.20","(48.49±6.89)"),"SCP":("-0.75","38.13","45.99"),"NexCP":("-0.40","36.08","43.70"),"ResCP":("5.02±0.40","13.63±0.55","16.21±0.53")},
}

def fmt(mean, std, micro):
    if micro:
        return f"{mean:.4f}±{std:.4f}" if std >= 5e-5 else f"{mean:.4f}±0.0000"
    return f"{mean:.2f}±{std:.2f}"

# ---- compute LLM-RSA-CP column from the 3 seed JSONs ----
acc = {}
for s in SEEDS:
    d = json.load(open(os.path.join(RES, s + ".json")))
    for r in d["rows"]:
        o = r["ordinary_selected_cp"]
        acc.setdefault(r["cell"], {"cov": [], "wid": [], "w": []})
        acc[r["cell"]]["cov"].append(o["coverage"]); acc[r["cell"]]["wid"].append(o["width"]); acc[r["cell"]]["w"].append(o["winkler"])

def parse_wink(s):
    """numeric Winkler for bold comparison; () excluded -> None."""
    if s.startswith("("):
        return None
    return float(s.split("±")[0])

rows = []
hdr = "| 数据集 | 模型 | 指标 | " + " | ".join(BASELINES) + " | **LLM-RSA-CP** |"
sep = "|" + "---|" * (3 + len(BASELINES) + 1)
print("# 表 1  α=0.1 主实验性能对比（LLM-RSA-CP = abl_fine 3 seed 均值±std）\n")
print("Winkler 加粗为覆盖有效方法中的最小值；†轻度欠覆盖，‡/§显著欠覆盖，()因欠覆盖不计入最优。\n")
print(hdr); print(sep)
for cell in ORDER:
    ds, model = cell.split("/")
    a = acc[cell]
    cov = np.array(a["cov"]); wid = np.array(a["wid"]); w = np.array(a["w"])
    micro = wid.mean() < 1.0
    dcov = (cov - 0.9) * 100.0
    ours = {"dcov": f"+{dcov.mean():.2f}±{dcov.std():.2f}",
            "wid": fmt(wid.mean(), wid.std(), micro),
            "w": fmt(w.mean(), w.std(), micro)}
    # bold: lowest valid Winkler among baselines + ours
    cands = {m: parse_wink(B[cell][m][2]) for m in BASELINES}
    cands["__ours__"] = float(w.mean())
    valid = {k: v for k, v in cands.items() if v is not None}
    best = min(valid, key=valid.get)
    def wk(method, val):
        return f"**{val}**" if method == best else val
    for mi, metric in enumerate(["ΔCov", "PI-Width", "Winkler"]):
        cells_out = []
        for m in BASELINES:
            v = B[cell][m][mi]
            cells_out.append(wk(m, v) if metric == "Winkler" else v)
        ov = [ours["dcov"], ours["wid"], ours["w"]][mi]
        cells_out.append(wk("__ours__", ov) if metric == "Winkler" else ov)
        c1 = ds if mi == 0 else ""
        c2 = model if mi == 0 else ""
        print(f"| {c1} | {c2} | {metric} | " + " | ".join(cells_out) + " |")
