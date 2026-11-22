import subprocess,sys
from pathlib import Path
p=Path(__file__).parent
for name in ('generate_pilot.py','classifier_pilot.py'):
 subprocess.run([sys.executable,'-u',str(p/name)],check=True)
