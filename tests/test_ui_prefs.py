"""Экран загрузки и оформление: тема и акцент сохраняются только известными значениями, окно читает их
из ui.json, а файлы экрана загрузки ссылаются только на то, что лежит в static."""
import json
import re
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from conftest import check_response

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "alexgpt" / "static"

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=20),
    lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=8), c, max_size=3), max_leaves=8)
ui_values = json_values | st.sampled_from(["", "light", "sepia", "blue", "teal", "Light", "../x"])


@given(body=st.dictionaries(st.sampled_from(["theme", "accent", "sidebar"]), ui_values, max_size=3))
def test_ui_saves_only_known_values(client, server, body):
    d = check_response(client.post("/api/ui", json=body))
    assert set(d) <= {"theme", "accent"}
    assert d.get("theme", "") in server.UI_THEMES and d.get("accent", "") in server.UI_ACCENTS
    assert json.loads((server.DATA_DIR / "ui.json").read_text(encoding="utf-8")) == d


def test_ui_ignores_non_object_body(client):
    for raw in ("[1, 2]", "null", "not json", '"light"'):
        assert check_response(client.post("/api/ui", data=raw, content_type="application/json")) == {}


def test_whitelist_matches_theme_css(server):
    css = (STATIC / "theme.css").read_text(encoding="utf-8")
    assert set(re.findall(r"html\[data-theme=(\w+)\]", css)) == server.UI_THEMES - {""}
    assert set(re.findall(r"html\[data-accent=(\w+)\]", css)) == server.UI_ACCENTS - {""}


def test_theme_css_fonts_exist():
    css = (STATIC / "theme.css").read_text(encoding="utf-8")
    fonts = re.findall(r"url\(([^)]+)\)", css)
    assert fonts and all((STATIC / f).exists() for f in fonts)


@pytest.mark.parametrize("page", [STATIC / "loading.html", ROOT / "alexgpt" / "templates" / "app.html"])
def test_pages_use_existing_icons(page):
    ids = set(re.findall(r'id="(i-[\w-]+)"', (STATIC / "icons.svg").read_text(encoding="utf-8")))
    used = set(re.findall(r"icons\.svg#(i-[\w-]+)", page.read_text(encoding="utf-8")))
    assert used and used <= ids, used - ids


def test_loader_page_contract():
    # gui.py вызывает finish() и showError(), а команды ловит по console.log('alexgpt:…')
    html = (STATIC / "loading.html").read_text(encoding="utf-8")
    for part in ("function finish(", "function showError(", "console.log('alexgpt:' + c)", 'href="theme.css"'):
        assert part in html


gui = pytest.importorskip("gui", reason="нужен PyQt6")


@given(text=st.text(max_size=60) | json_values.map(json.dumps))
def test_loader_query_never_fails(tmp_path_factory, text):
    d = tmp_path_factory.mktemp("ui")
    (d / "ui.json").write_text(text, encoding="utf-8")
    q = gui.loader_query(d)
    assert set(q) <= {"theme", "accent"} and all(v.isascii() and v.isalpha() for v in q.values())


def test_loader_query_reads_saved_theme(tmp_path):
    assert gui.loader_query(tmp_path) == {}
    (tmp_path / "ui.json").write_text('{"theme": "light", "accent": "a<b", "x": 1}', encoding="utf-8")
    assert gui.loader_query(tmp_path) == {"theme": "light"}
