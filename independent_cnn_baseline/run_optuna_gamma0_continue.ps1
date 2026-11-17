param([switch]$Detach,[switch]$Resume,[switch]$ClearStop,[string]$DeadlineUtc)
$ErrorActionPreference='Stop'
$taskRepo=Split-Path -Parent $PSScriptRoot
$taskPython=Join-Path (Split-Path -Parent $taskRepo) 'Plant_Disease\.venv\Scripts\python.exe'
$taskSpec=Join-Path $PSScriptRoot 'configs\optuna_convnext_base_gamma0_continue_20261102.json'
$taskConfig=Get-Content -LiteralPath $taskSpec -Raw | ConvertFrom-Json
$taskOutput=[IO.Path]::GetFullPath((Join-Path $taskRepo $taskConfig.output_dir))
if(-not $taskOutput.StartsWith($taskRepo+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Output must stay inside repository.'}
New-Item -ItemType Directory -Path $taskOutput -Force | Out-Null
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$taskArgs=@('-X','utf8','-B','-u','-m','independent_cnn_baseline.optuna_gamma0_continue','--spec',$taskSpec)
if($Resume){$taskArgs+='--resume'}
if($ClearStop){$taskArgs+='--clear-stop'}
if($DeadlineUtc){$taskArgs+=@('--deadline-utc',$DeadlineUtc)}
if($Detach){
    $taskStamp=[DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss')
    $taskStdout=Join-Path $taskOutput ("launcher_$taskStamp.stdout.log")
    $taskStderr=Join-Path $taskOutput ("launcher_$taskStamp.stderr.log")
    $taskArgs[$taskArgs.IndexOf($taskSpec)]='"'+$taskSpec+'"'
    $taskProcess=Start-Process -FilePath $taskPython -ArgumentList $taskArgs -WorkingDirectory $taskRepo -WindowStyle Hidden -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr -PassThru
    [ordered]@{launcher_pid=$taskProcess.Id;launched_at_utc=[DateTime]::UtcNow.ToString('o');python=$taskPython;module='independent_cnn_baseline.optuna_gamma0_continue';spec=$taskSpec;resume=[bool]$Resume;stdout=$taskStdout;stderr=$taskStderr} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskOutput 'launch.json') -Encoding UTF8
    Write-Output ("Started hidden five-hour gamma-zero continuation, launcher PID {0}. Status: {1}" -f $taskProcess.Id,(Join-Path $taskOutput 'status.json'))
}else{
    Push-Location -LiteralPath $taskRepo
    try{& $taskPython @taskArgs;if($LASTEXITCODE -ne 0){throw "Continuation exited with code $LASTEXITCODE; inspect logs before recovery."}}finally{Pop-Location}
}
