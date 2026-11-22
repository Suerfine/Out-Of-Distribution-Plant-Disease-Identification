$ErrorActionPreference = 'Stop'
$studyPath = $PSScriptRoot
$pythonPath = [IO.Path]::GetFullPath((Join-Path $studyPath '../../Plant_Disease/.venv/Scripts/python.exe'))
$logsPath = Join-Path $studyPath 'logs'
New-Item -ItemType Directory -Force -Path $logsPath | Out-Null
$launchStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutPath = Join-Path $logsPath "study_$launchStamp.stdout.log"
$stderrPath = Join-Path $logsPath "study_$launchStamp.stderr.log"
$runner = Start-Process -FilePath $pythonPath -ArgumentList @('-u','train.py','--config','config.json') -WorkingDirectory $studyPath -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
@{started_utc=[DateTime]::UtcNow.ToString('o');pid=$runner.Id;stdout=$stdoutPath;stderr=$stderrPath;mode='hidden resumable background process';progress_interval_minutes=15;chat_notifications=$false;run='runs/full_seed42'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $studyPath 'launch.json') -Encoding utf8
Write-Output "Started crop-conditioned disease training PID $($runner.Id)"
$runner.WaitForExit()
$runner.Refresh()
if ($runner.ExitCode -ne 0) { throw "Training failed; inspect $stderrPath" }
