import os
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
EXE_DIR = Path(sys.executable).parent if FROZEN else None
APP_DIR = Path(sys._MEIPASS) if FROZEN else Path(__file__).resolve().parent


def _default_models_dir():
    return APP_DIR / "models" if FROZEN else APP_DIR.parent / "models"


def _default_data_dir():
    if not FROZEN:
        return APP_DIR.parent / ".appdata"
    if (EXE_DIR / "portable.txt").exists():
        return EXE_DIR / "data"
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AlexGPT"


MODELS_DIR = Path(os.environ.get("ALEXGPT_MODELS") or _default_models_dir())
DATA_DIR = Path(os.environ.get("ALEXGPT_DATA") or _default_data_dir())
DATA_DIR.mkdir(parents=True, exist_ok=True)


def lm_config_path():
    user_cfg = DATA_DIR / "lm_config.json"
    return user_cfg if user_cfg.exists() else APP_DIR / "lm_config.json"


def script_cmd(name, *args):
    if FROZEN:
        return [sys.executable, "--run", name, *map(str, args)]
    return [sys.executable, "-u", str(APP_DIR / f"{name}.py"), *map(str, args)]


def pyfile_cmd(path):
    if FROZEN:
        return [sys.executable, "--pyfile", str(path)]
    return [sys.executable, str(path)]
