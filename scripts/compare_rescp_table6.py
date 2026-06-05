import ast, json, re, sys
from pathlib import Path
from statistics import mean, pstdev

TARGETS = {
    ("Solar", "RNN"): {"dcov_abs": 0.47, "width": 46.37, "winkler": 87.44},
    ("Solar", "Transf"): {"dcov_abs": 0.18, "width": 39.37, "winkler": 75.69},
    ("Solar", "ARIMA"): {"dcov_abs": 0.30, "width": 62.24, "winkler": 98.54},
    ("Beijing", "RNN"): {"dcov_abs": 0.98, "width": 52.21, "winkler": 86.72},
    ("Beijing", "Transf"): {"dcov_abs": 0.75, "width": 50.31, "winkler": 84.47},
    ("Beijing", "ARIMA"): {"dcov_abs": 0.32, "width": 54.70, "winkler": 99.59},
    ("Exchange", "RNN"): {"dcov_abs": 1.77, "width": 0.0180, "winkler": 0.0234},
    ("Exchange", "Transf"): {"dcov_abs": 2.37, "width": 0.0195, "winkler": 0.0262},
    ("Exchange", "ARIMA"): {"dcov_abs": 1.03, "width": 0.0175, "winkler": 0.0234},
    ("ACEA", "RNN"): {"dcov_abs": 1.86, "width": 8.06, "winkler": 11.14},
    ("ACEA", "Transf"): {"dcov_abs": 4.20, "width": 8.58, "winkler": 11.42},
    ("ACEA", "ARIMA"): {"dcov_abs": 1.16, "width": 8.31, "winkler": 11.66},
}

TARGETS_ALPHA_01 = {
    ("Solar", "RNN"): {"dcov_abs": 0.74, "width": 62.25, "winkler": 104.24},
    ("Solar", "Transf"): {"dcov_abs": 3.09, "width": 63.34, "winkler": 103.13},
    ("Solar", "ARIMA"): {"dcov_abs": 0.68, "width": 77.17, "winkler": 110.38},
    ("Beijing", "RNN"): {"dcov_abs": 0.70, "width": 65.96, "winkler": 106.07},
    ("Beijing", "Transf"): {"dcov_abs": 0.49, "width": 64.06, "winkler": 103.64},
    ("Beijing", "ARIMA"): {"dcov_abs": 0.63, "width": 70.43, "winkler": 108.75},
    ("Exchange", "RNN"): {"dcov_abs": 1.13, "width": 0.0210, "winkler": 0.0264},
    ("Exchange", "Transf"): {"dcov_abs": 1.46, "width": 0.0229, "winkler": 0.0294},
    ("Exchange", "ARIMA"): {"dcov_abs": 0.38, "width": 0.0207, "winkler": 0.0268},
    ("ACEA", "RNN"): {"dcov_abs": 1.56, "width": 9.61, "winkler": 12.91},
    ("ACEA", "Transf"): {"dcov_abs": 3.54, "width": 10.10, "winkler": 12.90},
    ("ACEA", "ARIMA"): {"dcov_abs": 5.02, "width": 13.63, "winkler": 16.21},
}


def get_targets(target_table="table6_alpha015"):
    table = str(target_table).lower()
    if table in {"table1_alpha01", "alpha01", "0.1", "table1"}:
        return TARGETS_ALPHA_01
    if table in {"table6_alpha015", "alpha015", "0.15", "table6"}:
        return TARGETS
    raise ValueError(f"Unknown target table: {target_table}")

NAME_MAP = [
    ("solar", "Solar"), ("beijing", "Beijing"), ("air", "Beijing"),
    ("exchange", "Exchange"), ("acea", "ACEA"),
]
MODEL_MAP = [("varima", "ARIMA"), ("arima", "ARIMA"), ("lstm", "RNN"), ("rnn", "RNN"), ("transf", "Transf")]
pat = re.compile(r"Evaluation metrics \(alpha ([0-9.]+): (\{.*\})")

def infer(path):
    s = path.name.lower()
    ds = next((v for k,v in NAME_MAP if k in s), None)
    md = next((v for k,v in MODEL_MAP if k in s), None)
    return ds, md

def parse_log(path):
    rows=[]
    for line in path.read_text(errors='replace').splitlines():
        m=pat.search(line)
        if not m: continue
        alpha=float(m.group(1))
        try: d=ast.literal_eval(m.group(2))
        except Exception: continue
        rows.append({
            'alpha': alpha,
            'dcov_signed_pct': float(d['mean_coverage_eps'])*100.0,
            'dcov_abs_pct': abs(float(d['mean_coverage_eps']))*100.0,
            'width': float(d['mean_pi_width']),
            'winkler': float(d['winkler_score']),
            'coverage': float(d['mean_coverage']),
        })
    return rows

def summarize(paths):
    out=[]
    for p in paths:
        rows=parse_log(p)
        if not rows: continue
        ds, md = infer(p)
        by={}
        for r in rows:
            by.setdefault(r['alpha'], []).append(r)
        for a, vals in sorted(by.items()):
            rec={'log': str(p), 'dataset': ds, 'model': md, 'alpha': a, 'n': len(vals)}
            for key in ['dcov_signed_pct','dcov_abs_pct','width','winkler','coverage']:
                xs=[v[key] for v in vals]
                rec[key+'_mean']=mean(xs)
                rec[key+'_std']=pstdev(xs) if len(xs)>1 else 0.0
            if abs(a-0.15)<1e-9 and ds and md and (ds,md) in TARGETS:
                t=TARGETS[(ds,md)]
                rec['rescp_dcov_abs']=t['dcov_abs']; rec['rescp_width']=t['width']; rec['rescp_winkler']=t['winkler']
                rec['pass_dcov']=abs(rec['dcov_signed_pct_mean']) < t['dcov_abs']
                rec['pass_width']=rec['width_mean'] < t['width']
                rec['pass_winkler']=rec['winkler_mean'] < t['winkler']
                rec['pass_all']=rec['pass_dcov'] and rec['pass_width'] and rec['pass_winkler']
            out.append(rec)
    return out

def main():
    paths = [Path(x) for x in sys.argv[1:]] or sorted(Path('logs/rescp_benchmark').glob('*.log'))
    out = summarize(paths)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
