"""Wait for the existing experiment's Windows process before using the GPU."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys

if __name__=='__main__':
    here = Path(__file__).resolve().parent
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE,wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    pid = int(sys.argv[1])
    process = kernel.OpenProcess(0x100000,False,pid)
    if process:
        print(f'Waiting for close-view trainer PID {pid}',flush=True)
        try:
            if kernel.WaitForSingleObject(process,0xffffffff)!=0:
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(process)
    elif ctypes.get_last_error()!=87:
        raise ctypes.WinError(ctypes.get_last_error())
    print('Close-view trainer exited. Starting multi-split gate study.',flush=True)
    kernel.SetThreadExecutionState(0x80000001)
    try:
        subprocess.run([sys.executable,'-u',str(here/'multisplit_gate.py')],check=True)
    finally:
        kernel.SetThreadExecutionState(0x80000000)
