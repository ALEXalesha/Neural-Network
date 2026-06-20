import subprocess, sys, os
from pathlib import Path

BASE = Path(r"c:\iCloudDrive\Алексей\Neural network")
os.chdir(str(BASE))
python = sys.executable if 'python' in sys.executable.lower() else 'python'
subprocess.run([python, str(BASE / 'coder_team.py')])
