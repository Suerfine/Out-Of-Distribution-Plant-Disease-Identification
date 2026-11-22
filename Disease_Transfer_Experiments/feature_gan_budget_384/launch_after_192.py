"""Start the development comparison after the official results are saved."""
import json
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    prior = here.parent/'feature_gan_official_192/summary.json'
    assert set(json.loads(prior.read_text())) == {'42','43','44'}
    subprocess.run([sys.executable,'-u',str(here/'run_study.py')],check=True)
