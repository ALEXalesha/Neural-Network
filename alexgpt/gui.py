"""AlexGPT — окно на PyQt6 + QtWebEngine поверх локального Flask-сервера."""
import getpass
import json
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

from PyQt6.QtCore import QTimer, QUrl, Qt
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtNetwork import QLocalServer, QLocalSocket
from PyQt6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication, QMainWindow, QMenu, QSystemTrayIcon

from paths import DATA_DIR, FROZEN

WIN_CMD = DATA_DIR / ".win_cmd"
NO_WIN  = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

LOADING_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;width:100%;height:100%;background:transparent;overflow:hidden}
#root{position:absolute;inset:8px;border-radius:12px;background:#0d1117;
      box-shadow:0 12px 48px rgba(0,0,0,.85),0 0 0 1px rgba(255,255,255,.07);
      display:flex;flex-direction:column;align-items:center;justify-content:center;gap:20px}
.icon{font-size:72px;animation:pulse 2s ease-in-out infinite}
h1{font-size:28px;font-weight:700;background:linear-gradient(135deg,#7c3aed,#3b82f6);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
p{color:#8b949e;font-size:14px;text-align:center;padding:0 20px}
.spin{width:40px;height:40px;border:3px solid #21262d;border-top-color:#7c3aed;
      border-radius:50%;animation:spin .8s linear infinite}
.err{color:#f85149;font-size:13px;max-width:500px;white-space:pre-wrap;text-align:left;
     background:#161b22;padding:12px;border-radius:6px;font-family:monospace;display:none;
     max-height:300px;overflow:auto}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.1)}}
</style></head>
<body><div id="root">
<div class="icon">🧠</div>
<h1>AlexGPT</h1>
<div class="spin" id="spin"></div>
<p id="msg">Запуск сервера…</p>
<div class="err" id="err"></div>
</div></body></html>"""


def make_icon(size: int = 48) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(QColor("transparent"))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor("#7c3aed"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.setPen(QColor("white"))
    f = QFont(); f.setPixelSize(size // 2); f.setBold(True)
    p.setFont(f)
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "A")
    p.end()
    return QIcon(pix)


class MainWindow(QMainWindow):
    SERVER_TIMEOUT_MS = 120_000  # 2 мин — первый запуск грузит модели

    def __init__(self, port, server):
        super().__init__()
        self.port = port
        self.server = server
        self._really_close = False
        self._server_wait_elapsed = 0
        self.setWindowTitle("AlexGPT")
        self.setMinimumSize(960, 640)
        self.resize(1400, 870)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        icon = make_icon()
        self.setWindowIcon(icon)

        profile = QWebEngineProfile.defaultProfile()
        profile.setPersistentStoragePath(str(DATA_DIR / "webdata"))
        profile.setCachePath(str(DATA_DIR / "webcache"))
        ws = profile.settings()
        ws.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        for attr in ("JavascriptCanAccessClipboard", "JavascriptCanPaste"):
            a = getattr(QWebEngineSettings.WebAttribute, attr, None)
            if a is not None:
                ws.setAttribute(a, True)

        self.view = QWebEngineView()
        self.view.page().setBackgroundColor(QColor("#0d1117"))
        self.view.setHtml(LOADING_HTML)
        self.setCentralWidget(self.view)

        self._setup_tray(icon)

        self._server_timer = QTimer(self)
        self._server_timer.setInterval(250)
        self._server_timer.timeout.connect(self._check_server)
        self._server_timer.start()

        WIN_CMD.unlink(missing_ok=True)
        self._cmd_timer = QTimer(self)
        self._cmd_timer.setInterval(50)
        self._cmd_timer.timeout.connect(self._check_win_cmd)
        self._cmd_timer.start()

    # Перемещение и ресайз окна без рамки делает сам UI: JS шлёт startmove/startresize_*
    def _check_win_cmd(self):
        if not WIN_CMD.exists():
            return
        try:
            cmd = WIN_CMD.read_text().strip()
            WIN_CMD.unlink(missing_ok=True)
        except OSError:
            return
        if cmd == "minimize":
            self.showMinimized()
        elif cmd == "maximize":
            if self.isMaximized():
                self.showNormal()
                js = ("document.documentElement.setAttribute('data-max','0');"
                      "var ic=document.getElementById('max-icon');if(ic)ic.textContent='☐';")
            else:
                self.showMaximized()
                js = ("document.documentElement.setAttribute('data-max','1');"
                      "var ic=document.getElementById('max-icon');if(ic)ic.textContent='❐';")
            self.view.page().runJavaScript(js)
        elif cmd == "hide":
            self.hide()
            self.tray.showMessage("AlexGPT", "Приложение работает в трее",
                                  QSystemTrayIcon.MessageIcon.Information, 2500)
        elif cmd == "quit":
            self.quit_app()
        elif cmd == "startmove":
            self.windowHandle().startSystemMove()
        elif cmd.startswith("startresize_"):
            E = Qt.Edge
            emap = {"n": E.TopEdge, "s": E.BottomEdge,
                    "w": E.LeftEdge, "e": E.RightEdge,
                    "nw": E.TopEdge | E.LeftEdge, "ne": E.TopEdge | E.RightEdge,
                    "sw": E.BottomEdge | E.LeftEdge, "se": E.BottomEdge | E.RightEdge}
            d = cmd.split("_", 1)[1]
            if d in emap:
                self.windowHandle().startSystemResize(emap[d])

    def _setup_tray(self, icon: QIcon):
        self.tray = QSystemTrayIcon(icon, self)
        self.tray.setToolTip("AlexGPT")
        menu = QMenu()
        menu.addAction(QAction("📂  Открыть", self, triggered=self.show_from_tray))
        menu.addSeparator()
        menu.addAction(QAction("✕  Выйти", self, triggered=self.quit_app))
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda r: self.show_from_tray()
            if r == QSystemTrayIcon.ActivationReason.Trigger else None
        )
        self.tray.show()

    def show_from_tray(self):
        if self.isMinimized():
            self.showNormal()
        self.show(); self.activateWindow(); self.raise_()

    def quit_app(self):
        self._really_close = True
        QApplication.instance().quit()

    def closeEvent(self, event):
        if self._really_close:
            event.accept()
        else:
            event.ignore()
            self.quit_app()

    def _check_server(self):
        if self.server.poll() is not None:
            self._server_timer.stop()
            self._show_server_error("Сервер завершился с ошибкой. Лог:")
            return
        try:
            socket.create_connection(("127.0.0.1", self.port), timeout=0.15).close()
        except OSError:
            self._server_wait_elapsed += self._server_timer.interval()
            if self._server_wait_elapsed >= self.SERVER_TIMEOUT_MS:
                self._server_timer.stop()
                self._show_server_error("Сервер не отвечает за 2 минуты. Лог:")
            return
        self._server_timer.stop()
        self.view.load(QUrl(f"http://127.0.0.1:{self.port}"))

    def _show_server_error(self, title):
        log_path = DATA_DIR / "server.log"
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2500:] if log_path.exists() else ""
        self.view.page().runJavaScript(
            "document.getElementById('spin').style.display='none';"
            f"document.getElementById('msg').textContent={json.dumps(title)};"
            "var e=document.getElementById('err');e.style.display='block';"
            f"e.textContent={json.dumps(tail)};"
        )


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn_server(port):
    env = {**os.environ,
           "ALEXGPT_PORT":       str(port),
           "ALEXGPT_DATA":       str(DATA_DIR),
           "ALEXGPT_WIN_CMD":    str(WIN_CMD),
           "ALEXGPT_PARENT_PID": str(os.getpid())}
    cmd = [sys.executable, "--server"] if FROZEN else [sys.executable, str(Path(__file__).with_name("main.py")), "--server"]
    return subprocess.Popen(cmd, env=env, cwd=str(DATA_DIR), stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WIN)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AlexGPT")
    app.setQuitOnLastWindowClosed(False)

    # Второй запуск просто показывает уже открытое окно
    instance_name = f"AlexGPT-{getpass.getuser()}"
    probe = QLocalSocket()
    probe.connectToServer(instance_name)
    if probe.waitForConnected(300):
        probe.write(b"show")
        probe.waitForBytesWritten(300)
        return
    QLocalServer.removeServer(instance_name)
    instance = QLocalServer()
    instance.listen(instance_name)

    port = free_port()
    server = spawn_server(port)
    win = MainWindow(port, server)
    instance.newConnection.connect(lambda: (instance.nextPendingConnection(), win.show_from_tray()))

    def on_quit():
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/api/unload", method="POST", data=b"{}",
                headers={"Content-Type": "application/json"}), timeout=3)
        except OSError:
            pass
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
        WIN_CMD.unlink(missing_ok=True)

    app.aboutToQuit.connect(on_quit)

    scr = app.primaryScreen()
    if scr:
        geo = scr.availableGeometry()
        win.move(geo.x() + (geo.width() - win.width()) // 2, geo.y() + (geo.height() - win.height()) // 2)
    win.show(); win.activateWindow(); win.raise_()
    app.exec()
