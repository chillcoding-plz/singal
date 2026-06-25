param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$InputFiles = @(
        "sample1_template_match_pdw_with_pred_label.txt",
        "sample2_template_match_pdw_with_pred_label.txt"
    ),
    [double]$BlockDuration = 30,
    [double]$DisplayInterval = 30,
    [string]$LlmLabels = "artifacts/llm_eval/hybrid_v1/hybrid_llm_labels.jsonl"
)

$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = "1"

$InputFiles = $InputFiles | ForEach-Object { $_ -split "," } | Where-Object { $_ }
if (-not $InputFiles -or $InputFiles.Count -eq 0) {
    $InputFiles = @(
        "sample1_template_match_pdw_with_pred_label.txt",
        "sample2_template_match_pdw_with_pred_label.txt"
    )
}

python -m app.run_pipeline `
    --input $InputFiles `
    --block-duration $BlockDuration `
    --display-interval $DisplayInterval `
    --llm-labels $LlmLabels
