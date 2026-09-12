"""Модуль рисования: сервер ↔ отдельный процесс draw_worker. Вместо SDXL — поддельный процесс
с тем же протоколом, поэтому тест быстрый и не нужна видеокарта."""
import sys
import textwrap

import pytest

from conftest import check_response

FAKE_WORKER = textwrap.dedent('''
    import json, os, sys, time
    from pathlib import Path
    stop = Path(os.environ["ALEXGPT_DRAW_STOP"])
    print(json.dumps({"ready": True, "device": "cpu"}), flush=True)
    for line in sys.stdin:
        p = json.loads(line)["prompt"]
        if p == "crash":
            sys.exit(3)
        if p == "slow":
            while not stop.exists():
                time.sleep(0.05)
            print(json.dumps({"error": "Остановлено"}), flush=True)
            continue
        print(json.dumps({"image": "data:image/png;base64,AAAA", "prompt": p}, ensure_ascii=False), flush=True)
''')


@pytest.fixture
def fake_worker(server, tmp_path, monkeypatch):
    (tmp_path / "draw_worker.py").write_text(FAKE_WORKER, encoding="utf-8")
    monkeypatch.setattr(server, "APP_DIR", tmp_path)
    monkeypatch.setattr(server, "draw_python", lambda: sys.executable)
    yield
    server.stop_draw_worker()


def test_draw_via_worker_reuses_process(client, server, fake_worker):
    d = check_response(client.post("/api/draw", json={"prompt": "кот в космосе"}))
    assert d["image"].startswith("data:image/png") and d["prompt"] == "кот в космосе"
    pid = server._draw.proc.pid
    check_response(client.post("/api/draw", json={"prompt": "ещё"}))
    assert server._draw.proc.pid == pid, "модель должна оставаться загруженной между картинками"


def test_addon_python_ignores_script_dir(client, server, fake_worker, monkeypatch):
    # В сборке draw_worker.py лежит среди библиотек программы: Python модуля не должен их импортировать
    monkeypatch.setattr(server, "draw_addon_python", lambda: sys.executable)
    check_response(client.post("/api/draw", json={"prompt": "кот"}))
    assert "-P" in server._draw.proc.args


def test_draw_stop(client, server, fake_worker):
    import threading
    threading.Timer(0.5, lambda: client.post("/api/draw/stop")).start()
    assert check_response(client.post("/api/draw", json={"prompt": "slow"})) == {"error": "Остановлено"}
    # Старый «Стоп» не должен обрывать следующую картинку
    assert "image" in check_response(client.post("/api/draw", json={"prompt": "после стопа"}))


def test_worker_crash_is_502_and_recovers(client, server, fake_worker):
    resp = client.post("/api/draw", json={"prompt": "crash"})
    assert resp.status_code == 502 and "завершился" in resp.get_json()["error"]
    assert "image" in check_response(client.post("/api/draw", json={"prompt": "снова"}))


def test_draw_without_addon_is_501(client, server, monkeypatch):
    monkeypatch.setattr(server, "draw_python", lambda: None)
    resp = client.post("/api/draw", json={"prompt": "кот"})
    assert resp.status_code == 501 and "Установить модуль" in resp.get_json()["error"]


def test_addon_status_and_remove_stay_inside_addon_dir(client, server, tmp_path, monkeypatch):
    addon, model, outside = tmp_path / "addons" / "draw", tmp_path / "hf" / "models--x", tmp_path / "keep.txt"
    (addon / "venv" / "Scripts").mkdir(parents=True)
    (addon / "venv" / "Scripts" / "python.exe").write_bytes(b"x" * 1000)
    (addon / "installed.json").write_text('{"torch": "cuda", "size_gb": 4.9}')
    model.mkdir(parents=True)
    (model / "unet.safetensors").write_bytes(b"y" * 1000)
    outside.write_text("keep")
    monkeypatch.setattr(server, "DRAW_ADDON", addon)
    monkeypatch.setattr(server, "hf_model_dir", lambda: model)
    s = check_response(client.get("/api/draw/addon"))
    assert s["addon"] and s["ready"] and s["torch"] == "cuda" and s["addon_gb"] == 4.9
    check_response(client.post("/api/draw/addon/remove", json={"model": False}))
    assert not addon.exists() and model.exists()
    check_response(client.post("/api/draw/addon/remove", json={"model": True}))
    assert not model.exists() and outside.read_text() == "keep"
    assert not check_response(client.get("/api/draw/addon"))["addon"]


def test_dir_size_ignores_symlinks(server, tmp_path):
    blob = tmp_path / "blobs" / "abc"
    blob.parent.mkdir()
    blob.write_bytes(b"z" * 300_000_000)
    (tmp_path / "snapshots").mkdir()
    try:
        (tmp_path / "snapshots" / "unet.safetensors").symlink_to(blob)
    except OSError:
        pytest.skip("нет прав на симлинки в этой системе")
    assert server.dir_size_gb(tmp_path) == 0.3


def test_install_reports_failed_step(client, server, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DRAW_ADDON", tmp_path / "addon")
    monkeypatch.setattr(server.shutil, "which", lambda name: sys.executable if name == "uv" else None)

    def fake_stream(cmd, env):
        yield server.sse({"line": "нет сети"})
        return 1
    monkeypatch.setattr(server, "stream_process", fake_stream)
    events = check_response(client.post("/api/draw/addon/install", json={}))
    assert events[-1]["done"] and not events[-1]["ok"] and "Python 3.12" in events[-1]["error"]
    assert not server._draw.installing and not (tmp_path / "addon" / "installed.json").exists()
