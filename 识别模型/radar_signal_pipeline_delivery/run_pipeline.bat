@echo off
setlocal

if "%~1"=="" (
  set INPUT_FILES=sample1_template_match_pdw_with_pred_label.txt sample2_template_match_pdw_with_pred_label.txt
) else (
  set INPUT_FILES=%*
)

set OMP_NUM_THREADS=1
python -m app.run_pipeline --input %INPUT_FILES% --block-duration 30 --display-interval 30 --llm-labels artifacts/llm_eval/hybrid_v1/hybrid_llm_labels.jsonl
