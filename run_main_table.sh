#!/bin/bash
# Reproduce the main-experiment Table 1 (LLM-RSA-CP, alpha=0.1) end-to-end.
# Runs the full pipeline (LLM router -> candidates -> selection -> split-conformal) for
# all 12 cells x 3 seeds, then assembles Table 1.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=${PY:-/root/miniconda3/bin/python}

# Main-method config (identical to the paper): finegrain candidates + frozen-LLM stability-audit router.
C="--cells full12 --root data --use-llm-candidate-router --llm-device cuda --llm-max-new-tokens 120 \
   --allow-llm-fallback --keep-llm-loaded --n-jobs 1 --router-prompt-variants 3 \
   --router-bootstrap-samples 0 --selector-mode sel_winkler_targetcov \
   --selector-over-tol 0.005 --selector-over-penalty 300 --router-mode stability_audit"

mkdir -p results logs
echo "[main_table] START $(date)"
$PY finegrain_wrapper.py $C --seed 0 --output-prefix abl_fine_g0       > logs/g0.log       2>&1; echo "[main_table] seed0 rc=$? $(date)"
$PY finegrain_wrapper.py $C --seed 1 --output-prefix abl_fine_g0_seed1 > logs/g0_seed1.log 2>&1; echo "[main_table] seed1 rc=$? $(date)"
$PY finegrain_wrapper.py $C --seed 2 --output-prefix abl_fine_g0_seed2 > logs/g0_seed2.log 2>&1; echo "[main_table] seed2 rc=$? $(date)"

echo "[main_table] assembling Table 1 ..."
$PY make_table1.py results > table1_alpha01.md
echo "[main_table] DONE -> table1_alpha01.md  $(date)"
