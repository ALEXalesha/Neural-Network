"""
AlexGPT — Десктопное приложение (Frameless, как Telegram)
Запуск: python desktop_app.py
"""
import sys, io, threading, socket
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PORT      = 5050
WIN_CMD   = Path(__file__).parent / ".win_cmd"   # файл-команда для управления окном

LOADING_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;width:100%;height:100%;background:transparent;overflow:hidden}
#root{position:absolute;inset:8px;border-radius:12px;background:#0d1117;
      box-shadow:0 12px 48px rgba(0,0,0,.85),0 0 0 1px rgba(255,255,255,.07);
      display:flex;flex-direction:column;align-items:center;justify-content:center;gap:20px}
.icon{font-size:72px;animation:pulse 2s ease-in-out infinite}
h1{font-size:28px;font-weight:700;background:linear-gradient(135deg,#7c3aed,#3b82f6);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
p{color:#8b949e;font-size:14px}
.spin{width:40px;height:40px;border:3px solid #21262d;border-top-color:#7c3aed;
      border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.1)}}
</style></head>
<body><div id="root">
<div class="icon">🧠</div>
<h1>AlexGPT</h1>
<div class="spin"></div>
<p>Запуск сервера и загрузка компонентов...</p>
</div></body></html>"""


def _install_deps():
    import subprocess
    for pkg in ["PyQt6", "PyQt6-WebEngine"]:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=False)


try:
    from PyQt6.QtWidgets import QApplication, QMainWindow, QSystemTrayIcon, QMenu
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
    from PyQt6.QtCore import QUrl, QTimer, Qt, QRect, QFile, QIODevice, pyqtSlot
    from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QAction
except ImportError:
    print("Устанавливаю PyQt6...")
    _install_deps()
    from PyQt6.QtWidgets import QApplication, QMainWindow, QSystemTrayIcon, QMenu
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
    from PyQt6.QtCore import QUrl, QTimer, Qt, QRect, QFile, QIODevice, pyqtSlot
    from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QAction


# ─── Иконка ───────────────────────────────────────────────────────────────────
def _make_icon(size: int = 48) -> QIcon:
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


# ─── Главное окно ─────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    BORDER    = 10   # px — зона ресайза по краям
    TITLEBAR  = 42   # px — высота тайтлбара (drag-зона)
    BTN_WIDTH = 130  # px — правая зона с кнопками (не drag)

    def __init__(self):
        super().__init__()
        self._really_close = False
        self.setWindowTitle("AlexGPT")
        self.setMinimumSize(960, 640)
        self.resize(1400, 870)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        icon = _make_icon()
        self.setWindowIcon(icon)

        # WebEngine
        profile = QWebEngineProfile.defaultProfile()
        ws = profile.settings()
        ws.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        ws.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        for attr in ("AllowRunningInsecureContent", "JavascriptCanAccessClipboard",
                     "JavascriptCanPaste", "ClipboardReadWriteEnabled"):
            try:
                ws.setAttribute(getattr(QWebEngineSettings.WebAttribute, attr), True)
            except Exception:
                pass

        self.view = QWebEngineView()
        self.view.page().setBackgroundColor(QColor("#0d1117"))  # не прозрачный — точно видимый
        self.view.setHtml(LOADING_HTML)
        self.setCentralWidget(self.view)

        self._setup_tray(icon)

        # Таймер: ждём сервер
        self._server_timer = QTimer(self)
        self._server_timer.setInterval(250)
        self._server_timer.timeout.connect(self._check_server)
        self._server_timer.start()

        # Таймер: читаем команды от JS (кнопки окна)
        WIN_CMD.unlink(missing_ok=True)
        self._cmd_timer = QTimer(self)
        self._cmd_timer.setInterval(30)
        self._cmd_timer.timeout.connect(self._check_win_cmd)
        self._cmd_timer.start()

    # ── Тайтлбар / ресайз через Windows WM_NCHITTEST ──────────────────────────
    def nativeEvent(self, eventType, message):
        # Используем ctypes для чтения MSG структуры
        # ВАЖНО: читаем через ctypes.cast, не from_address чтобы избежать hang
        try:
            import ctypes
            WM_NCHITTEST = 0x0084
            # Читаем message как (UINT message, ...) — первые 4 байта это message type
            msg_type = ctypes.c_uint.from_address(ctypes.c_size_t(int(message)).value).value
            if msg_type == WM_NCHITTEST:
                # Читаем lParam (смещение 12 байт в MSG: hwnd=8, message=4, wParam=8/4, lParam)
                # MSG: HWND(8) + UINT(4) + pad(4) + WPARAM(8) + LPARAM(8) на x64
                lp_offset = 8 + 4 + 4 + 8   # = 24 на x64
                lParam = ctypes.c_longlong.from_address(
                    ctypes.c_size_t(int(message)).value + lp_offset).value
                x = ctypes.c_int16(lParam & 0xFFFF).value
                y = ctypes.c_int16((lParam >> 16) & 0xFFFF).value
                geo  = self.frameGeometry()
                lx   = x - geo.x()
                ly   = y - geo.y()
                w, h = geo.width(), geo.height()
                B    = self.BORDER

                top   = ly < B;        bot   = ly > h - B
                left  = lx < B;        right = lx > w - B
                if top  and left:  return True, 13
                if top  and right: return True, 14
                if bot  and left:  return True, 16
                if bot  and right: return True, 17
                if top:            return True, 12
                if bot:            return True, 15
                if left:           return True, 10
                if right:          return True, 11
                if ly < self.TITLEBAR and lx < w - self.BTN_WIDTH:
                    return True, 2    # HTCAPTION — перетаскивание
        except Exception:
            pass
        return False, 0  # не обрабатываем — Qt обработает сам

    # ── Команды от JS (кнопки) ────────────────────────────────────────────────
    def _check_win_cmd(self):
        try:
            if WIN_CMD.exists():
                cmd = WIN_CMD.read_text().strip()
                WIN_CMD.unlink(missing_ok=True)
                if   cmd == "minimize":
                    self.showMinimized()
                elif cmd == "maximize":
                    if self.isMaximized(): self.showNormal()
                    else: self.showMaximized()
                    self.view.page().runJavaScript(
                        "var ic=document.getElementById('max-icon');"
                        "if(ic)ic.textContent=document.documentElement.getAttribute('data-max')==='1'?'☐':'❐';"
                    )
                elif cmd == "hide":
                    self.hide()
                    self.tray.showMessage("AlexGPT", "Приложение работает в трее",
                                         QSystemTrayIcon.MessageIcon.Information, 2500)
                elif cmd == "quit":
                    self._quit_app()
                elif cmd == "startmove":
                    self.windowHandle().startSystemMove()
                elif cmd.startswith("startresize_"):
                    from PyQt6.QtCore import Qt as _Qt
                    d = cmd.split("_", 1)[1]
                    E = _Qt.Edge
                    emap = {
                        "n":  E.TopEdge,
                        "s":  E.BottomEdge,
                        "w":  E.LeftEdge,
                        "e":  E.RightEdge,
                        "nw": E.TopEdge  | E.LeftEdge,
                        "ne": E.TopEdge  | E.RightEdge,
                        "sw": E.BottomEdge | E.LeftEdge,
                        "se": E.BottomEdge | E.RightEdge,
                    }
                    if d in emap:
                        self.windowHandle().startSystemResize(emap[d])
        except Exception:
            pass

    # ── Трей ─────────────────────────────────────────────────────────────────
    def _setup_tray(self, icon: QIcon):
        self.tray = QSystemTrayIcon(icon, self)
        self.tray.setToolTip("AlexGPT")
        menu = QMenu()
        menu.addAction(QAction("📂  Открыть", self, triggered=self.show_from_tray))
        menu.addSeparator()
        menu.addAction(QAction("✕  Выйти",   self, triggered=self._quit_app))
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda r: self.show_from_tray()
            if r == QSystemTrayIcon.ActivationReason.Trigger else None
        )
        self.tray.show()

    def show_from_tray(self):
        self.show(); self.activateWindow(); self.raise_()

    def _quit_app(self):
        self._really_close = True
        QApplication.instance().quit()

    def closeEvent(self, event):
        if self._really_close:
            event.accept()
        else:
            event.ignore()
            self.hide()
            self.tray.showMessage("AlexGPT", "Приложение работает в трее",
                                  QSystemTrayIcon.MessageIcon.Information, 2500)

    # ── Сервер ───────────────────────────────────────────────────────────────
    def _check_server(self):
        try:
            s = socket.create_connection(("127.0.0.1", PORT), timeout=0.15)
            s.close()
            self._server_timer.stop()
            self.view.load(QUrl(f"http://127.0.0.1:{PORT}"))
        except OSError:
            pass


# ─── Точка входа ──────────────────────────────────────────────────────────────
def main():
    import subprocess

    app = QApplication(sys.argv)
    app.setApplicationName("AlexGPT")
    app.setQuitOnLastWindowClosed(False)

    # Flask в отдельном процессе (DLL-изоляция)
    server = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "app_server.py")],
        cwd=str(Path(__file__).parent)
    )
    app.aboutToQuit.connect(server.terminate)
    app.aboutToQuit.connect(lambda: WIN_CMD.unlink(missing_ok=True))

    win = MainWindow()

    scr = app.primaryScreen()
    if scr:
        geo = scr.availableGeometry()
        win.move((geo.width() - win.width()) // 2, (geo.height() - win.height()) // 2)

    win.show()
    win.activateWindow()
    win.raise_()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        err = traceback.format_exc()
        Path(__file__).parent.joinpath("desktop_error.log").write_text(err, encoding="utf-8")
        raise
