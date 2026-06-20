import subprocess, os, shutil
from pathlib import Path
BASE = Path(r"c:\iCloudDrive\Алексей\Neural network")
os.chdir(str(BASE))
python = shutil.which('python') or shutil.which('python3') or 'python'
subprocess.run([python, str(BASE / 'menu.py')])
input("Нажми Enter для выхода...")
