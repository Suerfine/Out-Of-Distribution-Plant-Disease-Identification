param(
    [switch]$Detach,
    [switch]$Resume,
    [switch]$Prepare,
    [switch]$ClearStop,
    [double]$Hours,
    [string]$DeadlineUtc
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path (Split-Path -Parent $repoRoot) 'Plant_Disease\.venv\Scripts\python.exe'
$spec = Join-Path $PSScriptRoot 'configs\optuna_convnext_base_seed42.json'
$specData = Get-Content -LiteralPath $spec -Raw | ConvertFrom-Json
$output = [IO.Path]::GetFullPath((Join-Path $repoRoot $specData.output_dir))
if (-not $output.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Search output must stay inside this repository.'
}
$objectiveCorrection = Join-Path $output 'OBJECTIVE_SUPERSEDED.json'
if (Test-Path -LiteralPath $objectiveCorrection) {
    throw 'This historical study optimized calibrated H and is superseded. Follow OPTUNA_CONVNEXT_BASE_PLAN.md for the corrected gamma-zero objective; this launcher cannot resume the old objective.'
}
New-Item -ItemType Directory -Path $output -Force | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$argsList = @('-X', 'utf8', '-B', '-u', '-m', 'independent_cnn_baseline.optuna_search', '--spec', $spec)
if ($Resume) { $argsList += '--resume' }
if ($Prepare) { $argsList += '--prepare' }
if ($ClearStop) { $argsList += '--clear-stop' }
if ($PSBoundParameters.ContainsKey('Hours')) { $argsList += @('--hours', $Hours.ToString([Globalization.CultureInfo]::InvariantCulture)) }
if ($DeadlineUtc) { $argsList += @('--deadline-utc', $DeadlineUtc) }
if ($Detach) {
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss')
    $stdout = Join-Path $output ("launcher_$stamp.stdout.log")
    $stderr = Join-Path $output ("launcher_$stamp.stderr.log")
    $argsList[$argsList.IndexOf($spec)] = '"' + $spec + '"'
    $process = Start-Process -FilePath $python -ArgumentList $argsList -WorkingDirectory $repoRoot -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    [ordered]@{launcher_pid=$process.Id;launched_at_utc=[DateTime]::UtcNow.ToString('o');python=$python;
        module='independent_cnn_baseline.optuna_search';spec=$spec;resume=[bool]$Resume;stdout=$stdout;stderr=$stderr} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'launch.json') -Encoding UTF8
    Write-Output ("Started hidden ConvNeXt-Base Optuna search, launcher PID {0}. State: {1}" -f $process.Id,(Join-Path $output 'status.json'))
} else {
    Push-Location -LiteralPath $repoRoot
    try {
        & $python @argsList
        if ($LASTEXITCODE -ne 0) { throw "Optuna search exited with code $LASTEXITCODE. Inspect logs before resuming." }
    } finally { Pop-Location }
}
