"""Логика вокруг LM Studio, которую раньше проверяли только вживую: история чата, рассуждения R1,
повтор и запасная модель в «Решить тест», понятный текст ошибки."""
import json

import requests
from hypothesis import given, strategies as st

from conftest import check_response

msg = st.fixed_dictionaries({"role": st.sampled_from(["user", "assistant", "system", "tool", 1]),
                             "content": st.one_of(st.text(max_size=3000), st.none(), st.integers())})


@given(st.one_of(st.lists(st.one_of(msg, st.text(), st.none()), max_size=40), st.text(), st.none(), st.integers()))
def test_chat_history_invariants(server, raw):
    out = server.chat_history(raw)
    assert all(m["role"] in ("user", "assistant") and m["content"] and m["content"] == m["content"].strip() for m in out)
    assert sum(len(m["content"]) for m in out) <= server.CHAT_HISTORY_CHARS
    assert len(out) <= server.CHAT_HISTORY_MSGS
    assert not out or out[0]["role"] == "user"
    assert all("<think>" not in m["content"] for m in out if m["role"] == "assistant")


def test_chat_history_keeps_latest(server):
    raw = [{"role": "user", "content": f"вопрос {i}"} if i % 2 == 0 else {"role": "assistant", "content": f"ответ {i}"}
           for i in range(40)]
    out = server.chat_history(raw)
    assert out[-1]["content"] == "ответ 39" and out[0]["role"] == "user"
    think = server.chat_history([{"role": "user", "content": "2+2?"},
                                 {"role": "assistant", "content": "<think>считаю...</think>4"}])
    assert think[1]["content"] == "4"


class FakeStream:
    ok, status_code = True, 200

    def __init__(self, deltas):
        self.lines = [f"data: {json.dumps({'choices': [{'delta': d}]})}".encode() for d in deltas] + [b"data: [DONE]"]

    def iter_lines(self):
        return iter(self.lines)

    def close(self):
        pass


def fake_lm_online(monkeypatch, server):
    monkeypatch.setattr(server, "lm_loaded", lambda: ["any-model"])
    monkeypatch.setattr(server, "lm_error", lambda loaded: None)


def test_chat_streams_reasoning_and_history(client, server, monkeypatch):
    fake_lm_online(monkeypatch, server)
    sent = {}

    def post(url, json=None, **kw):
        sent.update(json)
        return FakeStream([{"reasoning_content": "Думаю"}, {"reasoning_content": "..."}, {"content": "Ответ"}])
    monkeypatch.setattr(requests, "post", post)
    history = [{"role": "user", "content": "Меня зовут Алексей"}, {"role": "assistant", "content": "Привет!"}]
    events = check_response(client.post("/api/chat", json={"message": "Как меня зовут?", "model": "reasoner",
                                                           "history": history}))
    assert [e["text"] for e in events if e.get("type") == "think"] == ["Думаю", "..."]
    assert "".join(e["text"] for e in events if e.get("type") == "token") == "Ответ"
    roles = [m["role"] for m in sent["messages"]]
    assert roles[-3:] == ["user", "assistant", "user"] and sent["messages"][-1]["content"] == "Как меня зовут?"


class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self.ok, self._p = status, status < 400, payload
        self.reason, self.text = "Bad Request", json.dumps(payload)

    def json(self):
        return self._p


def answer(content):
    return FakeResp(200, {"choices": [{"message": {"content": content, "reasoning_content": "..."}}]})


def test_homework_retries_channel_error(client, server, monkeypatch):
    fake_lm_online(monkeypatch, server)
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    replies = [FakeResp(400, {"error": "Channel Error"}), answer("ОТВЕТ: 4\nПОЯСНЕНИЕ: 2+2")]
    monkeypatch.setattr(requests, "post", lambda *a, **kw: replies.pop(0))
    d = check_response(client.post("/api/homework/solve", json={"text": "2+2?"}))
    assert "4" in d["answer"] and not replies


def test_homework_error_text_is_readable(client, server, monkeypatch):
    fake_lm_online(monkeypatch, server)
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    monkeypatch.setattr(requests, "post", lambda *a, **kw: FakeResp(400, {"error": {"message": "Model unloaded"}}))
    resp = client.post("/api/homework/solve", json={"text": "2+2?"})
    assert resp.status_code == 502 and "Model unloaded" in resp.get_json()["error"]


def test_parse_homework_several_answers_without_headers(server):
    raw = "ОТВЕТ: 4\nПОЯСНЕНИЕ: 2x = 8\n\nОТВЕТ: б) Париж\nПОЯСНЕНИЕ: столица Франции"
    answer, explanation = server.parse_homework(raw)
    assert answer == "Задание 1: 4\nЗадание 2: б) Париж"
    assert explanation == "2x = 8\nстолица Франции"
    assert server.parse_homework("ОТВЕТ: 42\nПОЯСНЕНИЕ: так") == ("42", "так")


def test_parse_homework_markdown_list_from_r1(server):
    # Дословный ответ DeepSeek R1 из живой проверки
    raw = "**ОТВЕТ:**\n1) x = 4  \n2) б) Париж  \n\n**ПОЯСНЕНИЕ:**  \n1) Вычитаем 6 и делим на 2.\n2) Столица Франции — Париж."
    answer, explanation = server.parse_homework(raw)
    assert answer == "1) x = 4  \n2) б) Париж"
    assert explanation.startswith("1) Вычитаем 6")
    assert server.parse_homework("**ЗАДАНИЕ 1:**\n**ОТВЕТ:** 4\n\n**ЗАДАНИЕ 2:**\n**ОТВЕТ:** в)")[0] == \
        "Задание 1: 4\nЗадание 2: в)"


@given(st.text(max_size=400))
def test_parse_homework_never_crashes(server, raw):
    answer, explanation = server.parse_homework(raw)
    assert isinstance(answer, str) and isinstance(explanation, str)


def test_homework_falls_back_to_writer_when_reasoning_ate_tokens(client, server, monkeypatch):
    fake_lm_online(monkeypatch, server)
    used = []

    def post(url, json=None, **kw):
        used.append(json["model"])
        return answer("" if len(used) == 1 else "ОТВЕТ: Париж")
    monkeypatch.setattr(requests, "post", post)
    d = check_response(client.post("/api/homework/solve", json={"text": "Столица Франции?"}))
    assert "Париж" in d["answer"]
    assert used == [server.LM_MDLS["reasoner"]["model_id"], server.LM_MDLS["writer"]["model_id"]]
