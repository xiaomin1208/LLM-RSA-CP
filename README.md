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
├── tmp_rescp_official/       # -> symlink to residual data (see Setup); not duplicated
├── results/                 # output JSONs (created on run)
└── table1_alpha01.md        # final table (created on run)
```

## Method note (data split)
The main method uses three contiguous calibration blocks **D_fit ≺ D_sel ≺ D_adj**:
D_fit trains the conditional residual-quantile models, D_sel selects the candidate, and
**D_adj (the most-recent block) gives the split-conformal threshold (eq. 15)**. A further
block D_ref is reserved only for the min-p / hybrid robust-threshold variants and is **not
used by the main method** — folding it into D_adj injects older, drifted residuals and
degrades efficiency (verified: Solar Winkler +12~15%), so D_adj is kept as the recent block.

## Setup
Requires the base conda env (numpy, pandas, lightgbm, torch, transformers, scikit-learn)
and the frozen Qwen2.5-7B-Instruct in the HF cache. Link the residual data:
```bash
ln -s /root/autodl-tmp/llmcp_llm_router_20260506/tmp_rescp_official ./tmp_rescp_official
```

## Run
```bash
bash run_main_table.sh         # ~1 GPU (RTX 5090); 3 seeds × 12 cells
# or, if results/ already populated, just rebuild the table:
python make_table1.py results > table1_alpha01.md
```

## Output
`table1_alpha01.md` — Table 1 with the LLM-RSA-CP column as mean±std over 3 seeds,
Winkler-best per cell bolded among coverage-valid methods. Baseline columns are the
ResCP same-protocol values (embedded in `make_table1.py`).
