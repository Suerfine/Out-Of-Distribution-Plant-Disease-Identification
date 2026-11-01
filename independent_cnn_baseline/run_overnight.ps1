param(
    [string]$Python,
    [string]$Spec,
    [double]$Hours = 10,
    [string]$DeadlineUtc,
    [switch]$Resume,
    [switch]$Prepare,
    [switch]$RefitOnly,
    [switch]$Detach
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Spec) { $Spec = Join-Path $PSScriptRoot $(if ($RefitOnly) { 'configs\refit_r0_seed42.json' } else { 'configs\study_seed42.json' }) }
$runnerModule = if ($RefitOnly) { 'independent_cnn_baseline.finalize' } else { 'independent_cnn_baseline.runner' }
$Spec = (Resolve-Path -LiteralPath $Spec).Path
if (-not $Python) {
    $candidates = @((Join-Path $repoRoot '.venv\Scripts\python.exe'),
                    (Join-Path (Split-Path -Parent $repoRoot) 'Plant_Disease\.venv\Scripts\python.exe'))
    $Python = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $Python) { throw 'Supply -Python with the project-local CUDA python.exe path.' }
$Python = (Resolve-Path -LiteralPath $Python).Path
$specData = Get-Content -LiteralPath $Spec -Raw | ConvertFrom-Json
$output = [IO.Path]::GetFullPath((Join-Path $repoRoot $specData.output_dir))
if (-not $output.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Study output must stay inside this repository.'
}
New-Item -ItemType Directory -Path $output -Force | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HOME = Join-Path $repoRoot 'runs\cache\independent_cnn_baseline\huggingface'
$env:MPLCONFIGDIR = Join-Path $repoRoot 'runs\cache\independent_cnn_baseline\matplotlib'
$env:CUDA_CACHE_PATH = Join-Path $repoRoot 'runs\cache\independent_cnn_baseline\cuda'
$dataSpec = $specData
if ($RefitOnly) { $dataSpec = Get-Content -LiteralPath (Join-Path $repoRoot $specData.source_spec) -Raw | ConvertFrom-Json }
if (-not $dataSpec.data_root) { throw 'The source study must specify its dataset root.' }
$env:DPD_ROOT = [IO.Path]::GetFullPath((Join-Path $repoRoot $dataSpec.data_root))
$argList = @('-B', '-u', '-m', $runnerModule, '--spec', ('"' + $Spec + '"'), '--hours', $Hours.ToString([Globalization.CultureInfo]::InvariantCulture))
if ($Resume) { $argList += '--resume' }
if ($Prepare) { $argList += '--prepare' }
if ($DeadlineUtc) { $argList += @('--deadline-utc', $DeadlineUtc) }
if ($Detach) {
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss')
    $stdout = Join-Path $output ("launcher_$stamp.stdout.log")
    $stderr = Join-Path $output ("launcher_$stamp.stderr.log")
    $process = Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $repoRoot `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    [ordered]@{launcher_pid=$process.Id;launched_at_utc=[DateTime]::UtcNow.ToString('o');
        python=$Python;module=$runnerModule;spec=$Spec;hours=$Hours;resume=[bool]$Resume;deadline_utc=$DeadlineUtc;stdout=$stdout;stderr=$stderr} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'launch.json') -Encoding UTF8
    Write-Output ("Started hidden controller, launcher PID {0}. State: {1}" -f $process.Id,(Join-Path $output 'status.json'))
} else {
    Push-Location -LiteralPath $repoRoot
    try {
        $direct = @('-B', '-u', '-m', $runnerModule, '--spec', $Spec, '--hours', $Hours)
        if ($Resume) { $direct += '--resume' }
        if ($Prepare) { $direct += '--prepare' }
        if ($DeadlineUtc) { $direct += @('--deadline-utc', $DeadlineUtc) }
        & $Python @direct
        if ($LASTEXITCODE -ne 0) { throw "Controller exited with code $LASTEXITCODE. Inspect training.log before resuming." }
    } finally { Pop-Location }
}
