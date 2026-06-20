import subprocess, os, sys, webbrowser, threading, time, shutil
from pathlib import Path
BASE = Path(r"c:\iCloudDrive\Алексей\Neural network")
os.chdir(str(BASE))
python = shutil.which('python') or shutil.which('python3') or 'python'
threading.Thread(target=lambda: (time.sleep(3), webbrowser.open('http://localhost:5050')), daemon=True).start()
subprocess.run([python, str(BASE / 'alexgpt_build' / 'main.py')])
input("Нажми Enter для выхода...")
