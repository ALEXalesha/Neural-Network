import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import requests
from hypothesis import HealthCheck, settings

ROOT = Path(__file__).resolve().parents[1]
os.environ["ALEXGPT_DATA"] = tempfile.mkdtemp(prefix="alexgpt_test_")
sys.path.insert(0, str(ROOT / "alexgpt"))

settings.register_profile(
    "default", max_examples=100, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
settings.register_profile(
    "thorough", max_examples=1000, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture(scope="session")
def server():
    import app_server
    return app_server


@pytest.fixture(scope="session")
def client(server):
    return server.app.test_client()


def _offline(*a, **kw):
    raise requests.exceptions.ConnectionError("LM Studio offline (test)")


@pytest.fixture(autouse=True)
def lm_offline(monkeypatch):
    for name in ("get", "post", "delete"):
        monkeypatch.setattr(requests, name, _offline)


def strict_json(body):
    def bad_constant(c):
        raise AssertionError(f"non-standard JSON constant {c!r} in {body[:200]!r}")
    return json.loads(body, parse_constant=bad_constant)


def sse_events(body):
    assert body.endswith("\n\n"), f"stream not terminated: {body[-200:]!r}"
    events = []
    for chunk in body.split("\n\n")[:-1]:
        assert chunk.startswith("data: "), f"bad SSE frame {chunk[:200]!r}"
        events.append(strict_json(chunk[6:]))
    return events


def check_response(resp):
    body = resp.get_data(as_text=True)
    assert resp.status_code != 500, (resp.status_code, body[:500])
    ctype = resp.headers.get("Content-Type", "")
    if ctype.startswith("application/json"):
        return strict_json(body)
    if ctype.startswith("text/event-stream"):
        events = sse_events(body)
        assert events, "empty event stream"
        last = events[-1]
        assert last.get("done") or last.get("type") in ("done", "error") or "error" in last, last
        return events
    return body
