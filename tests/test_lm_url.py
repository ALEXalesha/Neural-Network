"""Адрес LM Studio: localhost заменяется на 127.0.0.1 (иначе Windows ждёт ~2 с на попытку по IPv6
перед каждым запросом к LM Studio), всё остальное в адресе не меняется."""
from pathlib import Path
from urllib.parse import urlsplit

from hypothesis import given, strategies as st

from paths import lm_studio_url

ROOT = Path(__file__).resolve().parents[1]

hosts = st.sampled_from(["localhost", "LocalHost", "LOCALHOST", "127.0.0.1", "gitea.local", "lmstudio.local",
                         "localhost.example", "mylocalhost"])


@given(scheme=st.sampled_from(["http", "https"]), host=hosts, port=st.none() | st.integers(1, 65535),
       path=st.sampled_from(["", "/v1", "/v1/", "/api/v1"]))
def test_localhost_becomes_ipv4_rest_unchanged(scheme, host, port, path):
    src = f"{scheme}://{host}{f':{port}' if port else ''}{path}"
    out, s = urlsplit(lm_studio_url({"lm_studio_url": src})), urlsplit(src)
    assert out.hostname == ("127.0.0.1" if host.lower() == "localhost" else host.lower())
    assert (out.scheme, out.port, out.path) == (s.scheme, s.port, s.path)


@given(value=st.none() | st.integers() | st.text(max_size=40))
def test_never_fails(value):
    assert isinstance(lm_studio_url({"lm_studio_url": value}), str)


def test_default_and_credentials():
    assert lm_studio_url({}) == "http://127.0.0.1:1234/v1"
    assert lm_studio_url({"lm_studio_url": "http://u:p@localhost:1234/v1"}) == "http://u:p@127.0.0.1:1234/v1"


def test_every_module_uses_helper():
    # Если где-то снова прочитать адрес напрямую из конфига, вернётся задержка в 2 с на каждый запрос
    for name in ("app_server.py", "coder_team.py", "coder_team_v2.py"):
        src = (ROOT / "alexgpt" / name).read_text(encoding="utf-8")
        assert "lm_studio_url(" in src, name
        assert ".get('lm_studio_url'" not in src and '.get("lm_studio_url"' not in src, name
