import subprocess, os, sys, shutil
from pathlib import Path
BASE = Path(r"c:\iCloudDrive\Алексей\Neural network")
os.chdir(str(BASE))
python = shutil.which('python') or shutil.which('python3') or 'python'
subprocess.run([python, str(BASE / 'desktop_app.py')])
