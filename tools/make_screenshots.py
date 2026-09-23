"""Снимки для README: собираются программой, а не руками.

    .venv\\Scripts\\python tools\\make_screenshots.py

Скрипт сам поднимает сервер приложения на свободном порту с пустой временной папкой
данных, открывает интерфейс в Chromium через Playwright, запускает настоящие модели
(переводчик, NER, тональность, «Угадай рисунок») и снимает страницу.

Почему не снимок экрана по прямоугольнику окна: только что открытое окно может
оказаться позади других, и снимок тогда захватывает чужие окна пользователя. Однажды так
в кадр попал личный чат. Здесь снимается сама страница в безоконном Chromium - в кадр
физически не может попасть ничего, кроме интерфейса.

Зависимость только для этого скрипта: playwright (есть в requirements-dev.txt) и один
раз `python -m playwright install chromium`.
"""

import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
VIEWPORT = {"width": 1600, "height": 1000}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int, data_dir: str) -> subprocess.Popen:
    env = {**os.environ, "ALEXGPT_PORT": str(port), "ALEXGPT_DATA": data_dir, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "alexgpt" / "app_server.py")],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"сервер завершился с кодом {proc.returncode}")
        try:
            urllib.request.urlopen(url, timeout=2)
            return proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("сервер не ответил за 90 с")


def shot(page, name: str) -> None:
    path = OUT / name
    # Снимаем так, как окно выглядит развёрнутым (html[data-max='1']): без скругления.
    # В настоящем прозрачном окне углы скругления прозрачны, а в кадре Chromium они
    # залились бы светлым фоном страницы. До 1.3.1 вокруг окна было ещё и поле 8 px
    # под тень - в кадрах оно выходило светлой рамкой, на рабочем столе прозрачной.
    page.evaluate("document.documentElement.dataset.max = '1'")
    page.wait_for_timeout(150)
    page.screenshot(path=str(path))
    print(f"  {name} ({path.stat().st_size // 1024} КБ)")


def open_section(page, section: str) -> None:
    page.evaluate("s => go(s)", section)
    page.wait_for_timeout(400)


def translator(page) -> None:
    open_section(page, "translator")
    page.fill("#trans-src", "Artificial intelligence is changing the world. I love learning new things every day.")
    page.click("button[onclick='doTranslate()']")
    # Пока идёт перевод, в поле стоит заглушка «…переводю…» - ждать надо её исчезновения,
    # а не просто непустого поля, иначе в кадр попадает загрузка.
    page.wait_for_function(
        "(v => v.length > 0 && !v.startsWith('…'))(document.getElementById('trans-tgt').value.trim())",
        timeout=60_000)
    page.wait_for_timeout(300)
    shot(page, "translator.png")


def ner(page) -> None:
    open_section(page, "ner")
    page.fill("#ner-input", "Юрий Гагарин полетел в космос 12 апреля 1961 года с космодрома Байконур. "
                            "Позже он побывал в Лондоне и Токио.")
    page.click("button[onclick='analyzeNER()']")
    page.wait_for_function("document.getElementById('ner-list').children.length > 0", timeout=60_000)
    page.wait_for_timeout(300)
    shot(page, "ner.png")


def sentiment(page) -> None:
    open_section(page, "sentiment")
    page.fill("#sent-input", "Пришло быстро, упаковано аккуратно, но размер маломерит на целый номер.")
    page.click("button[onclick='analyzeSentiment()']")
    page.wait_for_function("document.getElementById('sent-bars').children.length > 0", timeout=60_000)
    page.wait_for_timeout(300)
    shot(page, "sentiment.png")


def sketch(page) -> None:
    """Рисуем «101» настоящей мышью: три отдельных символа, сервер режет их сам."""
    open_section(page, "sketch")
    page.evaluate("sketchClear()")
    box = page.locator("#sketch-canvas").bounding_box()

    def at(fx, fy):
        return box["x"] + fx * box["width"], box["y"] + fy * box["height"]

    def stroke(points):
        page.mouse.move(*at(*points[0]))
        page.mouse.down()
        for p in points[1:]:
            page.mouse.move(*at(*p), steps=4)
        page.mouse.up()
        page.wait_for_timeout(250)

    stroke([(0.18, 0.28), (0.18, 0.72)])
    ring = [(0.50 + 0.12 * math.sin(t / 24 * 2 * math.pi), 0.50 - 0.22 * math.cos(t / 24 * 2 * math.pi))
            for t in range(25)]
    stroke(ring)
    stroke([(0.82, 0.28), (0.82, 0.72)])

    page.wait_for_function("document.getElementById('sketch-result').innerText.includes('%')", timeout=60_000)
    page.wait_for_timeout(400)
    shot(page, "sketch.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    port = free_port()
    with tempfile.TemporaryDirectory(prefix="alexgpt-shots-") as data_dir:
        server = start_server(port, data_dir)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
                page.goto(f"http://127.0.0.1:{port}/")
                page.wait_for_function("typeof go === 'function'")
                page.wait_for_timeout(1500)

                open_section(page, "dashboard")
                page.wait_for_timeout(1500)
                shot(page, "hero.png")

                translator(page)
                ner(page)
                sentiment(page)
                sketch(page)

                # Две светлые темы из пяти: по ним видно, что оформление не прибито к тёмной.
                for theme in ("light", "sepia"):
                    page.evaluate("t => applyUi(Object.assign(uiSettings(), {theme: t}))", theme)
                    open_section(page, "dashboard")
                    page.wait_for_timeout(600)
                    shot(page, f"theme-{theme}.png")

                browser.close()
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
    print(f"Готово: {OUT}")


if __name__ == "__main__":
    main()
