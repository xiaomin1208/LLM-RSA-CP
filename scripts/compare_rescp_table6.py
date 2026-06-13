import ast, json, re, sys
from pathlib import Path
from statistics import mean, pstdev

TARGETS = {}  # baseline reference numbers removed; supply via logs if needed

TARGETS_ALPHA_01 = {}  # baseline reference numbers removed


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
