$ErrorActionPreference = 'Stop'
$studyPath = $PSScriptRoot
$pythonPath = [IO.Path]::GetFullPath((Join-Path $studyPath '../../Plant_Disease/.venv/Scripts/python.exe'))
$logsPath = Join-Path $studyPath 'logs'
New-Item -ItemType Directory -Force -Path $logsPath | Out-Null
$runPath = Join-Path $studyPath 'runs/full_seed42'
$previousLaunch = Get-Content -LiteralPath (Join-Path $studyPath 'launch.json') -Raw | ConvertFrom-Json
$previousStatus = Get-Content -LiteralPath (Join-Path $runPath 'status.json') -Raw | ConvertFrom-Json
@{recorded_utc=[DateTime]::UtcNow.ToString('o');reason='Checkpoint serialization I/O failure';previous_launch=$previousLaunch;previous_status=$previousStatus;repair='Checked stream writes, fsync, ZIP CRC, atomic replacement and storage-error retries';training_recipe_changed=$false;fixed_total_epochs=15} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runPath 'resume_after_io_failure_20261106.json') -Encoding utf8
$launchStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutPath = Join-Path $logsPath "recovery_$launchStamp.stdout.log"
$stderrPath = Join-Path $logsPath "recovery_$launchStamp.stderr.log"
$runner = Start-Process -FilePath $pythonPath -ArgumentList @('-u','../operations/checkpoint_recovery.py','--experiment','.') -WorkingDirectory $studyPath -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
@{started_utc=[DateTime]::UtcNow.ToString('o');pid=$runner.Id;stdout=$stdoutPath;stderr=$stderrPath;mode='hidden resumable background process with checked checkpoint I/O';progress_interval_minutes=15;chat_notifications=$false;run='runs/full_seed42';training_recipe_changed=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $studyPath 'launch.json') -Encoding utf8
Write-Output "Resumed Base training with checked checkpoint I/O, PID $($runner.Id)"
$runner.WaitForExit()
$runner.Refresh()
if ($runner.ExitCode -ne 0) { throw "Training failed; inspect $stderrPath" }
