"""Start the supervisor outside the tool's process job, then exit immediately."""
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from storage import atomic_json,now
from supervisor import native_python_environment

HERE=Path(__file__).resolve().parent


def main():
    cfg=json.loads((HERE/'config.json').read_text()); run=HERE/'runs'/cfg['run_name']; run.mkdir(parents=True,exist_ok=True)
    stdout=run/f'supervisor_{time.time_ns()}.stdout.log'; stderr=stdout.with_name(stdout.name.replace('stdout','stderr'))
    flags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB
    native,env=native_python_environment()
    with stdout.open('w') as out,stderr.open('w') as err:
        child=subprocess.Popen([native,'-u',str(HERE/'supervisor.py')],cwd=str(HERE),stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=flags,env=env)
    atomic_json(run/'launch.json',{'utc':now(),'supervisor_pid':child.pid,'launcher_pid':os.getpid(),'detached_from_tool_job':True,
        'stdout':str(stdout),'stderr':str(stderr),'sleep_prevention':'while supervisor alive','progress_interval_minutes':15,'chat_notifications':False})
    print(json.dumps({'launched':True,'supervisor_pid':child.pid,'launcher_exits_now':True}),flush=True)


if __name__=='__main__': main()
