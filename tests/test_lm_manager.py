import base64
import io
import json

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from PIL import Image

from conftest import check_response


@pytest.fixture
def lms_home(tmp_path, monkeypatch, server):
    models, hub = tmp_path / "models", tmp_path / "hub" / "models"
    (models / "pub" / "Repo-GGUF").mkdir(parents=True)
    (models / "pub" / "Repo-GGUF" / "Repo-Q4_K_M.gguf").write_bytes(b"x" * 10)
    (models / "pub" / "Repo-GGUF" / "mmproj-f16.gguf").write_bytes(b"y" * 5)
    (models / "pub" / "Two-GGUF").mkdir(parents=True)
    for q in ("Q4_K_M", "Q8_0"):
        (models / "pub" / "Two-GGUF" / f"Two-{q}.gguf").write_bytes(b"z")
    (models / "lmstudio-community" / "Hub-GGUF").mkdir(parents=True)
    (models / "lmstudio-community" / "Hub-GGUF" / "hub.gguf").write_bytes(b"h")
    (hub / "owner" / "hubmodel").mkdir(parents=True)
    (hub / "owner" / "hubmodel" / "manifest.json").write_text(json.dumps({"dependencies": [
        {"sources": [{"type": "huggingface", "user": "lmstudio-community", "repo": "Hub-GGUF"}]},
        {"sources": [{"type": "huggingface", "user": "..", "repo": ".."}]}]}))
    (tmp_path / "outside.txt").write_text("keep me")
    monkeypatch.setattr(server, "lms_dirs", lambda: (models.resolve(), hub.resolve()))
    return tmp_path


def test_model_files_targets(server, lms_home):
    models = lms_home / "models"
    assert server.model_files({"path": "pub/Repo-GGUF/Repo-Q4_K_M.gguf"}) == [(models / "pub" / "Repo-GGUF").resolve()]
    assert server.model_files({"path": "pub/Two-GGUF/Two-Q8_0.gguf"}) == [(models / "pub" / "Two-GGUF" / "Two-Q8_0.gguf").resolve()]
    hub = server.model_files({"path": "owner/hubmodel"})
    assert (models / "lmstudio-community" / "Hub-GGUF").resolve() in hub and len(hub) == 2


path_part = st.sampled_from(["..", ".", "pub", "Repo-GGUF", "Repo-Q4_K_M.gguf", "owner", "hubmodel", "C:", "", "~",
                             "outside.txt", "models", "hub"]) | st.text(max_size=6)


@given(st.lists(path_part, max_size=6).map("/".join) | st.text(max_size=40))
@settings(max_examples=400)
def test_model_files_never_leave_lmstudio_dirs(server, lms_home, path):
    models, hub = server.lms_dirs()
    for p in server.model_files({"path": path}):
        assert models in p.parents or hub in p.parents, (path, p)
        assert p != models and p != hub


def test_delete_removes_only_model(server, client, lms_home, monkeypatch):
    local = [{"modelKey": "k", "path": "pub/Two-GGUF/Two-Q8_0.gguf", "sizeBytes": 1}]
    monkeypatch.setattr(server, "lms_local_models", lambda: local)
    monkeypatch.setattr(server, "lms_path", lambda: "lms")
    monkeypatch.setattr(server, "lms_run", lambda *a, **kw: (0, ""))
    d = check_response(client.post("/api/lm/delete", json={"key": "k"}))
    assert d["ok"]
    assert not (lms_home / "models" / "pub" / "Two-GGUF" / "Two-Q8_0.gguf").exists()
    assert (lms_home / "models" / "pub" / "Two-GGUF" / "Two-Q4_K_M.gguf").exists()
    assert (lms_home / "outside.txt").read_text() == "keep me"
    assert client.post("/api/lm/delete", json={"key": "nope"}).status_code == 404


@given(st.from_regex(r"\A[A-Za-z0-9_-]{1,12}/[A-Za-z0-9._-]{1,20}/[A-Za-z0-9._-]{1,20}-(Q4_K_M|Q8_0|Q4_K_S|IQ3_XS)\.gguf\Z"))
def test_hf_spec(server, model_id):
    owner, repo, fname = model_id.split("/")
    spec = server.hf_spec(model_id)
    assert spec.startswith(f"https://huggingface.co/{owner}/{repo}@")
    assert server.SPEC_RE.fullmatch(spec)


@given(st.text(max_size=60))
@settings(max_examples=300)
def test_spec_validation_rejects_shell_chars(server, spec):
    if server.SPEC_RE.fullmatch(spec):
        assert not any(c in spec for c in " \t\n;&|<>`$\"'()"), spec


def test_lm_get_without_lms_is_501(client):
    for body in ({"role": "coder"}, {"spec": "https://huggingface.co/a/b@Q4_K_M"}):
        assert client.post("/api/lm/get", json=body).status_code == 501


def test_catalog_offline(client):
    d = check_response(client.get("/api/lm/catalog"))
    assert not d["online"] and not d["lms"]
    assert {r["role"] for r in d["roles"]} == {"coder", "reasoner", "writer", "vision"}
    assert all(r["downloaded"] is None and r["size_gb"] for r in d["roles"])


# ---- Угадай рисунок
def png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@given(arrays(np.uint8, st.tuples(st.integers(1, 200), st.integers(1, 200))), st.integers(-3, 30))
@settings(max_examples=150)
def test_sketch_guesses(client, server, arr, top):
    resp = client.post("/api/sketch/predict", json={"image": png_b64(arr), "top": top})
    d = check_response(resp)
    if not (arr > 25).any():
        assert resp.status_code == 400
        return
    g = d["guesses"]
    assert len(g) == min(max(top, 1), 20)
    probs = [x["prob"] for x in g]
    assert probs == sorted(probs, reverse=True) and all(0 <= p <= 100 for p in probs)
    labels = server.load_sketch().labels
    assert all(x["label"] in labels for x in g)
