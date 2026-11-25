# The full study in the order of Figure 3.6, for Windows PowerShell.  Every step can also be run on its own;
# finished networks and finished runs are skipped, so the script can be restarted after an interruption.
#
#   .\scripts\pipeline.ps1                      # all steps, 12 workers
#   .\scripts\pipeline.ps1 -Workers 8 -From 6   # start at step 6 (the experiments)
#   .\scripts\pipeline.ps1 -Hours 10            # each experiment stops after 10 hours; run again to continue
#
# Run the pilot study (step 2) first and read its tables before the long steps.
param(
    [int]$Workers = 12,
    [int]$From = 1,
    [double]$Hours = 0,
    [double]$RolloutSequences = 0.01
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:OMP_NUM_THREADS = "1"

function Step([string[]]$Arguments) {
    Write-Host "`n>>> python -m mctsjssp $($Arguments -join ' ')" -ForegroundColor Cyan
    & python -W ignore -m mctsjssp @Arguments
    if ($LASTEXITCODE -ne 0) { throw "step failed: $($Arguments -join ' ')" }
}
$runExtra = @("--set", "recording.rollout_sequences=$RolloutSequences")
if ($Hours -gt 0) { $runExtra += @("--hours", "$Hours") }

if ($From -le 1) { Step @("validate-env", "--config", "configs/base.yaml") }                      # 3.7, checks 1-2
if ($From -le 2) { Step @("pilot", "--config", "configs/pilot.yaml", "--set", "pilot.workers=$Workers") }  # 3.6.3
if ($From -le 3) {                                                                                # 3.6.2 and 3.6.1
    Step @("tune", "--config", "configs/base.yaml", "--configuration", "LR", "--parallel", "$Workers")
    Step @("tune", "--config", "configs/base.yaml", "--configuration", "AZ", "--parallel", "$Workers")
    Step @("train", "--config", "configs/base.yaml", "--all", "--workers", "$Workers")            # incl. fast networks
}
if ($From -le 4) {                                                                                # 3.7 check 3, 3.6.2
    Step @("validate-env", "--config", "configs/base.yaml", "--learning")
    Step @("assess", "--config", "configs/base.yaml", "--all")
}
if ($From -le 5) { Step @("select-entropy", "--config", "configs/base.yaml") }                    # temperatures, Table 3.5
if ($From -le 6) {                                                                                # Table 3.6
    foreach ($rq in "rq1", "rq2", "rq3", "rq4", "rq5", "fast") {
        Step (@("run", "--config", "configs/experiments/$rq.yaml", "--workers", "$Workers") + $runExtra)
    }
    Step @("fixed-states", "--config", "configs/experiments/rq3.yaml")
}
if ($From -le 7) {                                                                                # 3.6.4
    foreach ($rq in "rq1", "rq2", "rq3", "rq4", "rq5", "fast") {
        Step @("analyse", "--config", "configs/experiments/$rq.yaml")
    }
    $all = @("rq1", "rq2", "rq3", "rq4", "fast") | ForEach-Object { @("--config", "configs/experiments/$_.yaml") }
    Write-Host "`n>>> compare and compare-states across the experiments" -ForegroundColor Cyan
    & python -W ignore -m mctsjssp compare @all
    & python -W ignore -m mctsjssp compare-states @all
}
Write-Host "`nDone. Results are in results\, analyses in results\<experiment>\analysis\, comparisons in results\compare\." -ForegroundColor Green
