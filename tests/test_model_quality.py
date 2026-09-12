"""Проверки качества обученных моделей на очевидных примерах: ловят «сломанные» веса и рассинхрон
токенизации между lab/ и сервером (так раньше и выродился переводчик)."""
import pytest

from conftest import check_response


@pytest.mark.parametrize("text,label", [
    ("Отличный товар, очень доволен покупкой!", "позитивный"),
    ("Ужасное качество, полный брак, деньги на ветер.", "негативный"),
    ("Не рекомендую, сломался через день", "негативный"),
    ("Прекрасная вещь, всем советую", "позитивный"),
])
def test_sentiment_obvious(client, text, label):
    assert check_response(client.post("/api/sentiment/analyze", json={"text": text}))["label"] == label


@pytest.mark.parametrize("text,label", [
    ("Congratulations! You won a prize of $1000. Click here to claim now!", "SPAM"),
    ("Hi, can we meet tomorrow at 3pm to discuss the report?", "HAM"),
    ("Поздравляем! Вы выиграли iPhone, перейдите по ссылке, чтобы получить приз!", "SPAM"),
    ("Привет, купи хлеба по дороге домой", "HAM"),
])
def test_spam_obvious(client, text, label):
    assert check_response(client.post("/api/spam/analyze", json={"text": text}))["label"] == label


@pytest.mark.parametrize("text,expected", [
    ("Александр Пушкин родился в Москве 6 июня 1799 года.",
     {("Александр Пушкин", "PER"), ("Москве", "LOC"), ("6 июня 1799 года", "DATE")}),
    ("Elon Musk founded SpaceX in California in 2002.",
     {("Elon Musk", "PER"), ("SpaceX", "ORG"), ("California", "LOC"), ("2002", "DATE")}),
    ("Barack Obama was born in Hawaii on August 4, 1961.",
     {("Barack Obama", "PER"), ("Hawaii", "LOC"), ("August 4, 1961", "DATE")}),
])
def test_ner_known_sentences(client, text, expected):
    d = check_response(client.post("/api/ner/analyze", json={"text": text}))
    assert expected <= {(e["text"], e["type"]) for e in d["entities"]}


@pytest.mark.parametrize("text,direction,words", [
    ("Tom is my friend.", "en2ru", ["Том", "друг"]),
    ("I am very tired.", "en2ru", ["устал"]),
    ("Where is the station?", "en2ru", ["станц"]),
    ("Я люблю читать книги.", "ru2en", ["read", "book"]),
    ("Сегодня хорошая погода.", "ru2en", ["weather"]),
])
def test_translator_uses_source(client, text, direction, words):
    out = check_response(client.post("/api/translate", json={"text": text, "direction": direction}))["translation"]
    assert all(w.lower() in out.lower() for w in words), out


def test_translator_splits_sentences(client):
    out = check_response(client.post("/api/translate", json={
        "text": "Tom is my friend. Where is the station?", "direction": "en2ru"}))["translation"]
    assert "Том" in out and "станц" in out.lower(), out
