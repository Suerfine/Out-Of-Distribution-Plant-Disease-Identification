"""Detached overnight controller: sleep prevention, hang detection and recovery."""
import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from storage import atomic_json,event,now

HERE=Path(__file__).resolve().parent


def native_python_environment():
    # Windows venv python.exe is a shim that spawns another PID. Launch the
    # actual interpreter with the same site-packages so the watched PID is the
    # worker itself and all DataLoader descendants inherit its killable job.
    native=getattr(sys,'_base_executable',sys.executable)
    env=os.environ.copy()
    site_path=Path(env.get('AML_VENV_SITE',str(Path(sys.prefix)/'Lib/site-packages')))
    if site_path.is_dir():
        env['AML_VENV_SITE']=str(site_path)
        entries=[str(site_path)]+[p for p in env.get('PYTHONPATH','').split(os.pathsep) if p and p!=str(site_path)]
        env['PYTHONPATH']=os.pathsep.join(entries)
    return native,env


class WorkerJob:
    def __init__(self,process,enabled=True):
        self.handle=None
        if os.name!='nt' or not enabled: return
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_=[('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),('flags',w.DWORD),
                ('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),('active',w.DWORD),('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
        class IO(ctypes.Structure): _fields_=[(n,ctypes.c_ulonglong) for n in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class Extended(ctypes.Structure):
            _fields_=[('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR]; self.kernel.CreateJobObjectW.restype=w.HANDLE
        self.kernel.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]
        self.kernel.CloseHandle.argtypes=[w.HANDLE]
        self.handle=self.kernel.CreateJobObjectW(None,None)
        info=Extended(); info.basic.flags=0x2000  # KILL_ON_JOB_CLOSE includes DataLoader descendants.
        if not self.handle or not self.kernel.SetInformationJobObject(self.handle,9,ctypes.byref(info),ctypes.sizeof(info)) or not self.kernel.AssignProcessToJobObject(self.handle,int(process._handle)):
            self.close(); process.kill(); raise OSError(ctypes.get_last_error(),'Cannot isolate worker process tree')
    def close(self):
        if self.handle: self.kernel.CloseHandle(self.handle); self.handle=None


def keep_awake(enable):
    if os.name=='nt':
        result=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if enable else 0x80000000)
        if enable and not result: raise OSError('Windows sleep-prevention request failed')


def run_controller(command,run,cfg,poll=5,retry_delay=30,use_jobs=True,use_power=True,progress_callback=None):
    run=Path(run); run.mkdir(parents=True,exist_ok=True); (run/'worker_logs').mkdir(exist_ok=True)
    max_attempts=cfg['max_restarts']+1; started=time.monotonic(); micro=cfg.get('micro_batch',4)
    prior=run/'supervisor_status.json'
    if prior.exists():
        old=json.loads(prior.read_text()); micro=min(micro,old.get('micro_batch',micro))
    if use_power: keep_awake(True)
    try:
        for attempt in range(1,max_attempts+1):
            if time.monotonic()-started>cfg['supervisor_hours']*3600: raise RuntimeError('Overnight supervisor wall-time budget exhausted; checkpoints preserved')
            while shutil.disk_usage(run).free<cfg.get('min_free_bytes',0):
                atomic_json(prior,{'utc':now(),'state':'waiting_for_disk_space','pid':os.getpid(),'micro_batch':micro})
                if time.monotonic()-started>cfg['supervisor_hours']*3600: raise RuntimeError('Disk space remained below reserve')
                time.sleep(min(60,max(poll,1)))
            stdout_path=run/'worker_logs'/f'attempt_{time.time_ns()}.stdout.log'; stderr_path=stdout_path.with_name(stdout_path.name.replace('stdout','stderr'))
            cmd=list(command)+(['--micro-batch',str(micro)] if cfg.get('append_micro',True) else [])
            native,env=native_python_environment()
            if Path(cmd[0]).resolve()==Path(sys.executable).resolve(): cmd[0]=native
            flags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
            with stdout_path.open('w',encoding='utf-8') as out,stderr_path.open('w',encoding='utf-8') as err:
                child=subprocess.Popen(cmd,cwd=str(HERE),stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=flags,env=env)
                job=WorkerJob(child,use_jobs); last_progress=time.monotonic(); token=None; timed_out=False
                try:
                    while child.poll() is None:
                        if time.monotonic()-started>cfg['supervisor_hours']*3600 and not (run/'STOP.request').exists():
                            atomic_json(run/'STOP.request',{'utc':now(),'reason':'Overnight wall-time budget reached; checkpoint and pause'})
                        status_path=run/'status.json'
                        if status_path.exists():
                            try:
                                state=json.loads(status_path.read_text())
                                if state.get('pid')==child.pid:
                                    current=(state.get('updated_utc'),state.get('phase'),state.get('stage'),state.get('epoch'),state.get('step'))
                                    if current!=token: token=current; last_progress=time.monotonic()
                            except (OSError,ValueError): pass
                        elapsed=time.monotonic()-last_progress
                        atomic_json(prior,{'utc':now(),'state':'running','pid':os.getpid(),'worker_pid':child.pid,'attempt':attempt,
                            'micro_batch':micro,'seconds_without_progress':elapsed,'stdout':str(stdout_path),'stderr':str(stderr_path),'sleep_prevention':use_power})
                        if progress_callback is not None:progress_callback()
                        if elapsed>cfg['watchdog_seconds']:
                            timed_out=True; job.close()
                            if child.poll() is None: child.kill()
                            event(run/'supervisor_events.jsonl',event='watchdog_restart',worker_pid=child.pid,seconds_without_progress=elapsed)
                            break
                        time.sleep(poll)
                    code=child.wait(timeout=30)
                finally: job.close()
            state=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
            event(run/'supervisor_events.jsonl',event='worker_exit',attempt=attempt,exit_code=code,watchdog=timed_out,worker_pid=child.pid)
            if code==0 and state.get('phase') in ('complete','paused'):
                atomic_json(prior,{'utc':now(),'state':state['phase'],'pid':os.getpid(),'attempts':attempt,'micro_batch':micro,'worker_results':state})
                return
            if code in (65,78): raise RuntimeError('Worker reported a scientific/integrity failure; refusing blind retries: '+str(state))
            if code==42:
                options=[x for x in cfg['allowed_micro_batches'] if x<micro]
                if not options: raise RuntimeError('CUDA memory insufficient even at minimum microbatch')
                micro=max(options)
            if attempt==max_attempts: raise RuntimeError('Worker restart budget exhausted; checkpoints preserved')
            time.sleep(min(retry_delay*attempt,300))
    finally:
        if use_power: keep_awake(False)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default=str(HERE/'config.json')); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); run=HERE/'runs'/cfg['run_name']; run.mkdir(parents=True,exist_ok=True)
    # A separate controller lock prevents two supervisors from restarting each other.
    import msvcrt
    with (run/'supervisor.lock').open('a+b') as lock:
        if lock.seek(0,2)==0: lock.write(b'0'); lock.flush()
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            from protocol import load_config,prepare,stamp
            from evaluation import finalize_study
            resolved=load_config(args.config);p=prepare(resolved,run);completed=[];last_snapshot=0.
            def publish(phase,arm=None,force=False,**details):
                nonlocal last_snapshot
                child_status={}
                path=run/arm/'status.json' if arm else None
                if path and path.exists():
                    try:child_status=json.loads(path.read_text())
                    except (OSError,ValueError):pass
                value={'updated_utc':stamp(),'phase':phase,'pid':os.getpid(),'current_arm':arm,
                    'completed_arms':list(completed),'total_arms':len(cfg['arms']),'worker_status':child_status,**details}
                atomic_json(run/'status.json',value);atomic_json(run/'supervisor_status.json',value)
                if force or time.monotonic()-last_snapshot>=cfg['progress_interval_seconds']:
                    atomic_json(run/'progress_15min.json',value)
                    with (run/'progress_15min.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')
                    last_snapshot=time.monotonic()
            for arm in cfg['arms']:
                arm_run=run/arm
                if (arm_run/'status.json').exists() and (arm_run/'verification.json').exists():
                    prior_state=json.loads((arm_run/'status.json').read_text());verification=json.loads((arm_run/'verification.json').read_text())
                    if prior_state.get('phase')=='complete' and verification.get('passed'):
                        completed.append(arm);continue
                publish('running',arm,force=True)
                run_controller([sys.executable,'-u',str(HERE/'train.py'),'--config',str(Path(args.config).resolve()),'--arm',arm],
                    arm_run,cfg,progress_callback=lambda:publish('running',arm))
                worker=json.loads((arm_run/'status.json').read_text())
                if worker.get('phase')=='paused':publish('paused',arm,force=True);return
                if worker.get('phase')!='complete':raise RuntimeError('Arm did not finish cleanly: '+arm)
                completed.append(arm);publish('arm_complete',arm,force=True)
            checks=finalize_study(p,resolved,run);publish('complete',force=True,verification_checks=checks,results=str(run/'results.md'))
        except BaseException as error:
            atomic_json(run/'supervisor_status.json',{'utc':now(),'state':'needs_attention','pid':os.getpid(),'error':repr(error),'checkpoints_preserved':True})
            atomic_json(run/'status.json',{'updated_utc':now(),'phase':'needs_attention','pid':os.getpid(),'error':repr(error),'checkpoints_preserved':True})
            raise


if __name__=='__main__': main()
