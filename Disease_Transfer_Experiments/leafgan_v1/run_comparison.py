import subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
for args in [('export_multi.py',),('classifier_multi.py','--arm','leafgan'),('classifier_multi.py','--arm','real_only')]:
    subprocess.run([sys.executable,'-u',str(p/args[0]),*args[1:]],check=True,cwd=p)
