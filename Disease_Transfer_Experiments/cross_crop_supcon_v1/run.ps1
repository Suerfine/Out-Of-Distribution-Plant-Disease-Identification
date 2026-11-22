$ErrorActionPreference = 'Stop'
$studyPath = $PSScriptRoot
$pythonPath = [IO.Path]::GetFullPath((Join-Path $studyPath '../../Plant_Disease/.venv/Scripts/python.exe'))
$logsPath = Join-Path $studyPath 'logs'
New-Item -ItemType Directory -Force -Path $logsPath | Out-Null
$launchStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutPath = Join-Path $logsPath "study_$launchStamp.stdout.log"
$stderrPath = Join-Path $logsPath "study_$launchStamp.stderr.log"
@{started_utc=[DateTime]::UtcNow.ToString('o');stdout=$stdoutPath;stderr=$stderrPath;mode='resumable persistent exec session'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $studyPath 'launch.json') -Encoding utf8
Set-Location -LiteralPath $studyPath
& $pythonPath -u train.py --config config.json 2>> $stderrPath | Tee-Object -FilePath $stdoutPath
if ($LASTEXITCODE -ne 0) { throw "Contrastive study failed; inspect $stderrPath" }
