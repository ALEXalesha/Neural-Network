"""
AlexGPT — единая точка входа (и для исходников, и для PyInstaller-сборки).

  AlexGPT.exe                         → окно приложения
  AlexGPT.exe --server                → Flask-сервер (его запускает окно)
  AlexGPT.exe --run <модуль> [args]   → coder_team / coder_team_v2 (их запускает сервер)
  AlexGPT.exe --pyfile <файл.py>      → выполнить Python-файл (тесты Coder Team v2)

В сборке нет python.exe, поэтому все дочерние процессы — это тот же exe с ключом.
"""
import os
import runpy
import sys
import traceback
from pathlib import Path

from paths import DATA_DIR

CHILD_MODULES = {"coder_team", "coder_team_v2"}


def _open_stdio(log_path=None):
    # В windowed-сборке sys.stdout бывает None, даже когда вывод перенаправлен в pipe
    for name, fd in (("stdout", 1), ("stderr", 2)):
        if log_path:
            stream = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
        elif getattr(sys, name) is None:
            try:
                stream = open(fd, "w", encoding="utf-8", errors="replace", buffering=1, closefd=False)
            except OSError:
                stream = open(os.devnull, "w")
        else:
            stream = getattr(sys, name)
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        setattr(sys, name, stream)


def run_child(argv):
    """Код выхода дочернего режима или None, если нужно открыть окно."""
    if not argv or argv[0] not in ("--server", "--run", "--pyfile"):
        return None
    mode, rest = argv[0], argv[1:]
    try:
        if mode == "--server":
            log_path = DATA_DIR / "server.log"
            log_path.write_text("", encoding="utf-8")
            _open_stdio(log_path)
            import app_server
            app_server.run()
        elif mode == "--run":
            _open_stdio()
            if not rest or rest[0] not in CHILD_MODULES:
                print(f"Неизвестный модуль: {rest[:1]}", file=sys.stderr)
                return 2
            sys.argv = [rest[0], *rest[1:]]
            runpy.run_module(rest[0], run_name="__main__", alter_sys=True)
        else:
            _open_stdio()
            path = Path(rest[0]).resolve()
            sys.argv = [str(path), *rest[1:]]
            sys.path.insert(0, str(path.parent))
            runpy.run_path(str(path), run_name="__main__")
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else int(e.code is not None)
    except BaseException:
        # Без этого windowed-exe показал бы окно с трейсбеком вместо кода выхода
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    code = run_child(sys.argv[1:])
    if code is not None:
        sys.exit(code)
    from gui import main
    try:
        main()
    except Exception:
        (DATA_DIR / "alexgpt_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
