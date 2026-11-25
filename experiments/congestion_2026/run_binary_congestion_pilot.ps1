param(
    [string]$OutputDir = '',
    [switch]$Smoke
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$python = Join-Path $repo '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Existing Statistical-Practice .venv is required.' }
if (-not $OutputDir) {
    $name = if ($Smoke) { 'binary_congestion_pilot_smoke_final' } else { 'binary_congestion_pilot' }
    $OutputDir = Join-Path $PSScriptRoot "outputs/$name"
}
Push-Location $repo
try {
    $runArgs = @('-u', '-m', 'experiments.congestion_2026.run_binary_congestion_pilot', '--out', $OutputDir)
    if ($Smoke) { $runArgs += '--smoke' }
    $manifest = Join-Path $OutputDir 'manifest.json'
    if (Test-Path -LiteralPath $manifest) {
        $runArgs += '--resume'
        $record = Get-Content -LiteralPath $manifest -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($record.reused_model_artifacts) {
            $first = $record.reused_model_artifacts.PSObject.Properties | Select-Object -First 1
            $prior = Split-Path (Split-Path $first.Name -Parent) -Parent
            $runArgs += @('--reuse-models-from', $prior)
        }
    }
    & $python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Pilot failed: $LASTEXITCODE" }
    if (-not $Smoke) {
        & $python -u -m experiments.congestion_2026.finalize_binary_congestion_pilot --out $OutputDir
        if ($LASTEXITCODE -ne 0) { throw "Final validation failed: $LASTEXITCODE" }
    }
} finally { Pop-Location }
