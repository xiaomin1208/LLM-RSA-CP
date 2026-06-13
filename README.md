# Main-experiment reproduction — LLM-RSA-CP Table 1 (α=0.1)

Self-contained pipeline that regenerates **only** the main experiment table (Table 1:
LLM-RSA-CP vs. baselines on 12 cells). Ablation / router / generality experiments are
**not** included here by design.

## Contents
```
main_table_repro/
├── scripts/                 # core pipeline (run_aci.py + modules) — copied as-is
├── finegrain_wrapper.py     # main-method config: finegrain candidates + LLM stability-audit router
├── run_main_table.sh        # driver: 3 seeds × 12 cells → results/, then builds the table
├── make_table1.py           # assembles Table 1 (Markdown) from the 3 seed result JSONs
├── data/                    # residual data (ResCP/HopCPT benchmark) — included
├── results/                 # output JSONs (created on run; git-ignored)
└── table1_alpha01.md        # final table (created on run)
```

## Method note (data split)
The main method uses three contiguous calibration blocks **D_fit ≺ D_sel ≺ D_adj**:
D_fit trains the conditional residual-quantile models, D_sel selects the candidate, and
**D_adj (the most-recent block) gives the split-conformal threshold (eq. 15)**. A further
block D_ref is reserved only for the min-p / hybrid robust-threshold variants and is **not
used by the main method** — folding it into D_adj injects older, drifted residuals and
degrades efficiency (verified: Solar Winkler +12~15%), so D_adj is kept as the recent block.

## Data
The base-predictor residuals for the 4 datasets (Solar, Beijing, Exchange, ACEA × RNN /
Transformer / ARIMA) are included under `data/` (split-conformal residuals + calibration/
test indices). They are taken from the **ResCP (Reservoir Conformal Prediction)**
benchmark — we reuse the same base-predictor residuals, data split, and test indices for
cell-wise comparability with ResCP. Datasets: Solar (PV generation), Beijing (air quality),
Exchange (exchange rate), ACEA (electricity). The repository ships the residuals only (not
the raw series or model checkpoints).

## Setup
Requires the base conda env (numpy, pandas, lightgbm, torch, transformers, scikit-learn)
and the frozen Qwen2.5-7B-Instruct in the HF cache. Data is already in `data/` — no extra
download needed (the driver passes `--root data`).

## Run
```bash
bash run_main_table.sh         # ~1 GPU (RTX 5090); 3 seeds × 12 cells
# or, if results/ already populated, just rebuild the table:
python make_table1.py results > table1_alpha01.md
```

## Output
`table1_alpha01.md` — the LLM-RSA-CP metrics (ΔCov / PI-Width / Winkler) per cell as
mean±std over the 3 seeds, computed at run time from the result JSONs. No numbers are
pre-filled in the repo: baseline comparison values are not included — pull them from the
respective baselines' own reports when assembling the paper table.
