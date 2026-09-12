"""AlexGPT — окно на PyQt6 + QtWebEngine поверх локального Flask-сервера."""
import getpass
import json
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from PyQt6.QtCore import QTimer, QUrl, Qt
from PyQt6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtNetwork import QLocalServer, QLocalSocket
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication, QMainWindow, QMenu, QSystemTrayIcon

from paths import APP_DIR, DATA_DIR, FROZEN

WIN_CMD = DATA_DIR / ".win_cmd"
NO_WIN  = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
STATIC  = APP_DIR / "static"
LOADER  = STATIC / "loading.html"
MAX_ICON = '<svg class="ic"><use href="/static/icons.svg#i-{}"/></svg>'


def loader_query(data_dir: Path = DATA_DIR) -> dict:
    """Тема и акцент для экрана загрузки. localStorage программы до запуска сервера недоступен,
    поэтому программа сохраняет их в ui.json (POST /api/ui)."""
    try:
        ui = json.loads((data_dir / "ui.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(ui, dict):
        return {}
    query = {}
    for key in ("theme", "accent"):
        v = ui.get(key)
        if isinstance(v, str) and v.isascii() and v.isalpha() and len(v) <= 20:
            query[key] = v
    return query


def make_icon(size: int = 48) -> QIcon:
    """Запасной значок, если logo.png не нашёлся."""
    pix = QPixmap(size, size)
    pix.fill(QColor("transparent"))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor("#7c5cff"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(1, 1, size - 2, size - 2, size * .28, size * .28)
    p.setPen(QColor("white"))
    f = QFont(); f.setPixelSize(size // 2); f.setBold(True)
    p.setFont(f)
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "A")
    p.end()
    return QIcon(pix)


def app_icon() -> QIcon:
    logo = STATIC / "logo.png"
    return QIcon(str(logo)) if logo.exists() else make_icon()


class WindowPage(QWebEnginePage):
    """Экран загрузки не может звать сервер (его ещё нет), поэтому команды окну пишет в консоль:
    console.log('alexgpt:startmove'). Слушаем только локальный файл экрана загрузки."""

    def __init__(self, profile, parent, on_cmd):
        super().__init__(profile, parent)
        self._on_cmd = on_cmd

    def javaScriptConsoleMessage(self, level, message, line, source):
        if message.startswith("alexgpt:") and self.url().isLocalFile():
            self._on_cmd(message.split(":", 1)[1])


class MainWindow(QMainWindow):
    SERVER_TIMEOUT_MS = 300_000  # первый запуск после установки: антивирус сканирует ~1 ГБ файлов

    def __init__(self, port, server):
        super().__init__()
        self.port = port
        self.server = server
        self._really_close = False
        self._server_wait_elapsed = 0
        self._loader_ready = False
        self._pending_js = None
        self.setWindowTitle("AlexGPT")
        self.setMinimumSize(960, 640)
        self.resize(1400, 870)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        icon = app_icon()
        self.setWindowIcon(icon)

        # defaultProfile() в Qt 6 — "инкогнито": localStorage с историей чатов терялся бы при выходе
        self.profile = QWebEngineProfile("AlexGPT", QApplication.instance())
        self.profile.setPersistentStoragePath(str(DATA_DIR / "webdata"))
        self.profile.setCachePath(str(DATA_DIR / "webcache"))
        ws = self.profile.settings()
        ws.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        for attr in ("JavascriptCanAccessClipboard", "JavascriptCanPaste"):
            a = getattr(QWebEngineSettings.WebAttribute, attr, None)
            if a is not None:
                ws.setAttribute(a, True)

        self.view = QWebEngineView()
        self.view.setPage(WindowPage(self.profile, self.view, self._loader_cmd))
        # Прозрачный фон: скруглённые углы и тень рисует сама страница
        self.view.page().setBackgroundColor(QColor(0, 0, 0, 0))
        self.view.loadFinished.connect(self._on_load_finished)
        self._show_loader()
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

    # ── Экран загрузки ──
    def _show_loader(self):
        self._loader_ready = False
        url = QUrl.fromLocalFile(str(LOADER))
        url.setQuery(urlencode(loader_query()))
        self.view.load(url)

    def _on_load_finished(self, ok):
        if not self.view.url().isLocalFile():
            return
        self._loader_ready = True
        if self._pending_js:
            self.view.page().runJavaScript(self._pending_js)
            self._pending_js = None

    def _loader_js(self, js):
        if self._loader_ready:
            self.view.page().runJavaScript(js)
        else:
            self._pending_js = js

    def _loader_cmd(self, cmd):
        if cmd == "startmove":
            self.windowHandle().startSystemMove()
        elif cmd == "minimize":
            self.showMinimized()
        elif cmd == "quit":
            self.quit_app()
        elif cmd == "openlog":
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(DATA_DIR)))
        elif cmd == "restart":
            self._restart_server()

    def _restart_server(self):
        if self.server.poll() is None:
            self.server.kill()
        self.server = spawn_server(self.port)
        self._server_wait_elapsed = 0
        self._show_loader()
        self._server_timer.start()

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
            maximized = not self.isMaximized()
            self.showMaximized() if maximized else self.showNormal()
            icon = json.dumps(MAX_ICON.format("copy" if maximized else "square"))
            title = json.dumps("Восстановить" if maximized else "Развернуть")
            self.view.page().runJavaScript(
                f"document.documentElement.setAttribute('data-max','{int(maximized)}');"
                f"var ic=document.getElementById('max-icon');if(ic)ic.innerHTML={icon};"
                f"var b=document.getElementById('btn-max');if(b)b.title={title};")
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
        menu.addAction(QAction("Открыть", self, triggered=self.show_from_tray))
        menu.addSeparator()
        menu.addAction(QAction("Выйти", self, triggered=self.quit_app))
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
            self._show_server_error("Сервер завершился с ошибкой")
            return
        try:
            socket.create_connection(("127.0.0.1", self.port), timeout=0.15).close()
        except OSError:
            self._server_wait_elapsed += self._server_timer.interval()
            if self._server_wait_elapsed >= self.SERVER_TIMEOUT_MS:
                self._server_timer.stop()
                self._show_server_error(f"Сервер не ответил за {self.SERVER_TIMEOUT_MS // 60000} минут")
            return
        self._server_timer.stop()
        # Короткое затухание экрана загрузки, потом сама программа
        self._loader_js("typeof finish === 'function' && finish()")
        QTimer.singleShot(260, lambda: self.view.load(QUrl(f"http://127.0.0.1:{self.port}")))

    def _show_server_error(self, title):
        log_path = DATA_DIR / "server.log"
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2500:] if log_path.exists() else ""
        self._loader_js(f"showError({json.dumps(title)}, {json.dumps(tail)})")


def pick_port():
    # localStorage (история чатов) привязан к origin с портом, поэтому порт по возможности постоянный
    saved = DATA_DIR / "port.txt"
    text = saved.read_text().strip() if saved.exists() else ""
    for port in (int(text) if text.isdigit() else 5050, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            port = s.getsockname()[1]
        saved.write_text(str(port))
        return port


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

    port = pick_port()
    win = MainWindow(port, spawn_server(port))
    instance.newConnection.connect(lambda: (instance.nextPendingConnection(), win.show_from_tray()))

    def on_quit():
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/api/unload", method="POST", data=b"{}",
                headers={"Content-Type": "application/json"}), timeout=3)
        except OSError:
            pass
        win.server.terminate()          # после «Перезапустить» это уже новый процесс сервера
        try:
            win.server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            win.server.kill()
        WIN_CMD.unlink(missing_ok=True)

    app.aboutToQuit.connect(on_quit)

    scr = app.primaryScreen()
    if scr:
        geo = scr.availableGeometry()
        win.move(geo.x() + (geo.width() - win.width()) // 2, geo.y() + (geo.height() - win.height()) // 2)
    win.show(); win.activateWindow(); win.raise_()
    app.exec()
