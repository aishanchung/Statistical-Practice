param(
    [string]$OutputDir = '',
    [ValidateSet('all','train','analyze','finalize')][string]$Stage = 'all',
    [switch]$Smoke
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$python = Join-Path $repo '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Existing Statistical-Practice .venv is required.' }
if (-not $OutputDir) {
    $name = if ($Smoke) { 'binary_classifier_comparison_smoke_release' } else { 'binary_classifier_comparison' }
    $OutputDir = Join-Path $PSScriptRoot "outputs/$name"
}
Push-Location $repo
try {
    if ($Stage -in @('all','train')) {
        $runArgs = @('-u','-m','experiments.congestion_2026.run_binary_classifier_comparison','--out',$OutputDir)
        if ($Smoke) { $runArgs += '--smoke' }
        if (Test-Path -LiteralPath (Join-Path $OutputDir 'manifest.json')) { $runArgs += '--resume' }
        & $python @runArgs
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $LASTEXITCODE" }
    }
    if ($Stage -in @('all','analyze')) {
        $analysisArgs = @('-u','-m','experiments.congestion_2026.analyze_binary_classifier_comparison','--out',$OutputDir)
        if ($Smoke) { $analysisArgs += @('--smoke','--bootstrap-reps','60') }
        & $python @analysisArgs
        if ($LASTEXITCODE -ne 0) { throw "Analysis failed: $LASTEXITCODE" }
    }
    if ((-not $Smoke) -and ($Stage -in @('all','finalize'))) {
        & $python -u -m experiments.congestion_2026.finalize_binary_classifier_comparison --out $OutputDir
        if ($LASTEXITCODE -ne 0) { throw "Delivery checks failed: $LASTEXITCODE" }
    }
} finally { Pop-Location }
