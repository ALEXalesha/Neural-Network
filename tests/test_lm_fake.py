import json

import pytest
import requests
from hypothesis import given, settings, strategies as st

from conftest import check_response


class FakeResp:
    def __init__(self, payload=None, lines=(), status=200):
        self.status_code = status
        self.ok = status < 400
        self._payload = payload
        self._lines = list(lines)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise requests.exceptions.HTTPError(response=self)

    def iter_lines(self):
        yield from self._lines

    def close(self):
        pass


@pytest.fixture
def fake_lm(monkeypatch):
    state = {"reply": "", "chunks": [], "status": 200, "sent": []}

    def fake_get(url, **kw):
        return FakeResp({"data": [{"id": "m1"}]})

    def fake_post(url, **kw):
        state["sent"].append(kw.get("json"))
        if state["status"] != 200:
            return FakeResp(status=state["status"])
        if kw.get("stream"):
            lines = [b"data: " + json.dumps({"choices": [{"delta": {"content": c}}]}).encode()
                     for c in state["chunks"]]
            return FakeResp(lines=lines + [b"", b"data: [DONE]"])
        return FakeResp({"choices": [{"message": {"content": state["reply"]}}]})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    return state


@given(st.lists(st.text(min_size=1, max_size=20), max_size=30))
@settings(max_examples=150)
def test_chat_stream_reassembles_tokens(client, fake_lm, chunks):
    fake_lm["chunks"] = chunks
    events = check_response(client.post("/api/chat", json={"message": "привет"}))
    assert events[0]["type"] == "model"
    assert "".join(e["text"] for e in events if e["type"] == "token") == "".join(chunks)
    assert events[-1]["type"] == "done"


@pytest.mark.parametrize("status", [200, 500])
def test_chat_image_leaves_no_temp_file(client, server, fake_lm, status):
    fake_lm["chunks"], fake_lm["status"] = ["ok"], status
    img = "data:image/png;base64,iVBORw0KGgo="
    check_response(client.post("/api/chat", json={"message": "что тут?", "image": img}))
    assert not list(server.DATA_DIR.glob("_upload_tmp*"))
    sent = fake_lm["sent"][-1]
    assert sent["messages"][-1]["content"][0]["image_url"]["url"].endswith("iVBORw0KGgo=")


answer = st.text(alphabet=st.characters(blacklist_categories=["Cc", "Cs", "Zl", "Zp"]), min_size=1, max_size=30) \
    .map(str.strip).filter(lambda s: s and not any(w in s.upper() for w in ("ЗАДАНИЕ", "ОТВЕТ", "ПОЯСНЕНИЕ")))


@given(st.lists(st.tuples(answer, st.none() | answer), min_size=2, max_size=6))
def test_homework_parses_every_task(client, fake_lm, tasks):
    blocks = []
    for i, (ans, exp) in enumerate(tasks, 1):
        blocks.append(f"ЗАДАНИЕ {i}:\nОТВЕТ: {ans}" + (f"\nПОЯСНЕНИЕ: {exp}" if exp else ""))
    fake_lm["reply"] = "<think>hmm</think>" + "\n\n".join(blocks)
    d = check_response(client.post("/api/homework/solve", json={"text": "реши"}))
    assert d["answer"].split("\n") == [f"Задание {i}: {a}" for i, (a, _) in enumerate(tasks, 1)]
    assert "<think>" not in d["raw"]


@given(answer, st.none() | answer)
def test_homework_single_answer(client, fake_lm, ans, exp):
    fake_lm["reply"] = f"ОТВЕТ: {ans}" + (f"\nПОЯСНЕНИЕ: {exp}" if exp else "")
    d = check_response(client.post("/api/homework/solve", json={"text": "2+2"}))
    assert d["answer"] == ans
    assert d["explanation"] == (exp or "")
