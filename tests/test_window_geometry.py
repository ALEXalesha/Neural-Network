"""Место и размер окна между запусками (1.4.0): window.json рядом с ui.json."""
import json

import pytest
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow

import gui
import window_geometry


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window():
    # Без рамки, как настоящее окно AlexGPT: у окна с рамкой, ни разу не показанного, Qt
    # угадывает толщину рамки, и место при восстановлении съезжает на пару пикселей.
    win = QMainWindow()
    win.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
    win.setMinimumSize(960, 640)
    win.resize(1400, 870)
    return win


def test_a_window_comes_back_where_and_how_large_it_was(qapp, tmp_path):
    win = _window()
    # Экран самого окна: у скрытого окна он не следит за перемещением на другой монитор.
    area = win.screen().availableGeometry()
    # Влезает и в 1024x768 раннера. Не у самого верха: restoreGeometry() Qt оставляет над
    # окном место под заголовок (32 px) и сдвинул бы окно с y=30 на 32 - один раз, дальше
    # это место уже сохраняется как есть.
    want = QRect(area.x() + 40, area.y() + 60, 980, 640)
    win.setGeometry(want)
    gui.save_window(win, tmp_path)
    assert isinstance(gui.load_window(tmp_path), str)
    assert not (tmp_path / "window.tmp").exists()
    again = _window()
    assert gui.place_window(again, qapp.primaryScreen(), tmp_path)
    assert (again.x(), again.y(), again.width(), again.height()) == (want.x(), want.y(), 980, 640)
    win.deleteLater()
    again.deleteLater()


@pytest.mark.parametrize("text", ["", "{", "[]", "null", '{"window": 42}', '{"window": "@@@"}',
                                  '{"window": "aGVsbG8="}', '{"window": "AAAA"}'])
def test_a_broken_file_centres_the_default_window(qapp, tmp_path, text):
    (tmp_path / "window.json").write_text(text, encoding="utf-8")
    win = _window()
    size = win.size()
    screen = qapp.primaryScreen()
    assert not gui.place_window(win, screen, tmp_path)
    geo = screen.availableGeometry()
    assert win.size() == size
    assert (win.x(), win.y()) == (geo.x() + (geo.width() - size.width()) // 2,
                                  geo.y() + (geo.height() - size.height()) // 2)
    win.deleteLater()


def test_no_file_centres_the_window(qapp, tmp_path):
    assert gui.load_window(tmp_path) is None
    assert not gui.place_window(_window(), qapp.primaryScreen(), tmp_path)


def test_a_window_saved_far_off_screen_is_not_restored_there(qapp, tmp_path):
    far = QMainWindow()
    far.resize(1000, 700)
    far.move(-30000, -30000)
    (tmp_path / "window.json").write_text(json.dumps({"window": window_geometry.encode(far)}), encoding="utf-8")
    win = _window()
    gui.place_window(win, qapp.primaryScreen(), tmp_path)
    # Либо Qt сам перенёс окно на экран, либо проверка отказалась и окно по центру.
    assert window_geometry.title_on_screen(win)
    far.deleteLater()
    win.deleteLater()


def test_the_quit_handler_saves_the_window():
    src = (gui.Path(gui.__file__)).read_text(encoding="utf-8")
    on_quit = src.split("def on_quit():", 1)[1].split("\n    app.aboutToQuit", 1)[0]
    assert "save_window(win)" in on_quit
    assert "place_window(win, app.primaryScreen())" in src
