import base64
import io
import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from PIL import Image

from conftest import check_response

json_leaf = (st.none() | st.booleans() | st.integers(-10**20, 10**20)
             | st.floats(allow_nan=False, allow_infinity=False) | st.text(max_size=40))
json_any = st.recursive(json_leaf, lambda c: st.lists(c, max_size=4)
                        | st.dictionaries(st.text(max_size=8), c, max_size=4), max_leaves=12)
num_like = (st.integers(-10**6, 10**6) | st.floats(allow_nan=False, allow_infinity=False, width=32)
            | st.integers(-5, 5).map(str) | st.sampled_from(["", "abc", "1e400", "nan", "inf", None]))

ENDPOINT_KEYS = {
    "/api/math/solve": ["expression"],
    "/api/mnist/predict": ["image"],
    "/api/sentiment/analyze": ["text"],
    "/api/translate": ["text", "direction"],
    "/api/gan/generate": ["digit", "count"],
    "/api/price/predict": ["area", "rooms", "floor", "district", "age"],
    "/api/timeseries/forecast": ["values"],
    "/api/temperature/predict": ["temp_today", "pressure", "humidity", "wind", "cloud", "month"],
    "/api/spam/analyze": ["text"],
    "/api/cluster/customer": ["age", "orders", "avg", "total", "days", "visits"],
    "/api/anomaly/check": ["amount", "hour", "freq", "foreign", "online", "balance", "distance"],
    "/api/defect/check": ["thickness", "mass", "hardness", "roughness", "length", "width", "temp", "time"],
    "/api/ner/analyze": ["text"],
    "/api/summarize": ["text", "n_sentences"],
    "/api/cluster/documents": ["documents", "k"],
    "/api/recommender/recommend": ["liked", "top_n"],
    "/api/homework/solve": ["text", "image", "subject"],
    "/api/chat": ["message", "model", "image"],
    "/api/lm/download": ["model_id"],
    "/api/coder_v2/control": ["action", "text"],
}


def body_for(keys):
    return (st.fixed_dictionaries({}, optional={k: json_any | num_like for k in keys})
            | json_any)


@pytest.mark.parametrize("path", sorted(ENDPOINT_KEYS))
@given(data=st.data())
def test_fuzz_never_500(client, path, data):
    body = data.draw(body_for(ENDPOINT_KEYS[path]))
    check_response(client.post(path, json=body))


@pytest.mark.parametrize("path", sorted(ENDPOINT_KEYS) + ["/api/gpt/stream", "/api/draw", "/api/coder_team/run"])
@pytest.mark.parametrize("payload,ctype", [
    ("not json", "text/plain"), ("{broken", "application/json"), ("", "application/json"),
])
def test_non_json_body_is_client_error(client, path, payload, ctype):
    resp = client.post(path, data=payload, content_type=ctype)
    check_response(resp)
    if resp.headers.get("Content-Type", "").startswith("application/json"):
        assert 400 <= resp.status_code < 500 or "error" in resp.get_json(), resp.get_json()


@pytest.mark.parametrize("path,method,code", [
    ("/favicon.ico", "get", 404), ("/no/such/page", "get", 404),
    ("/api/status", "post", 405), ("/api/chat", "get", 405),
])
def test_http_errors_keep_their_status(client, path, method, code):
    assert getattr(client, method)(path).status_code == code


def test_foreign_origin_is_rejected(client):
    resp = client.post("/api/win/quit", json={}, headers={"Origin": "http://evil.example"})
    assert resp.status_code == 403
    ok = client.post("/api/status", headers={"Origin": "http://127.0.0.1:5050"})
    assert ok.status_code != 403


def test_index_and_status(client):
    resp = client.get("/")
    assert resp.status_code == 200 and b"<html" in resp.data.lower()
    st_ = check_response(client.get("/api/status"))
    assert all(st_["models"].values()), st_["models"]


def test_marked_is_served_locally(client):
    html = client.get("/").get_data(as_text=True)
    assert "cdn.jsdelivr" not in html
    assert client.get("/static/marked.min.js").status_code == 200


# ---- math: result must equal Python's own arithmetic
def expr_tree():
    num = st.integers(0, 10**6).map(str) | st.floats(0, 1e6, allow_nan=False).map(lambda f: f"{f:.3f}")
    return st.recursive(num, lambda e: st.tuples(e, st.sampled_from("+-*/"), e).map(lambda t: f"({t[0]}{t[1]}{t[2]})")
                        | e.map(lambda x: f"-{x}"), max_leaves=12)


@given(expr_tree())
def test_math_matches_python(client, expr):
    d = check_response(client.post("/api/math/solve", json={"expression": expr}))
    try:
        want = eval(expr)
    except ZeroDivisionError:
        assert "error" in d
        return
    assert "error" not in d, d
    assert math.isclose(d["result"], round(float(want), 8), rel_tol=1e-9, abs_tol=1e-8)


@given(st.text(min_size=1, max_size=20).filter(lambda s: any(c.isalpha() for c in s)))
def test_math_rejects_letters_instead_of_dropping_them(client, junk):
    d = check_response(client.post("/api/math/solve", json={"expression": f"2{junk}+3"}))
    assert "error" in d


# ---- mnist
@given(arrays(np.uint8, st.tuples(st.integers(1, 120), st.integers(1, 120))), st.booleans())
@settings(max_examples=150)
def test_mnist_probabilities(client, arr, prefix):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    d = check_response(client.post("/api/mnist/predict", json={"image": ("data:image/png;base64," if prefix else "") + b64}))
    assert 0 <= d["digit"] <= 9 and len(d["probs"]) == 10
    assert abs(sum(d["probs"]) - 100) < 1.0
    assert d["probs"][d["digit"]] == max(d["probs"])
    assert math.isclose(d["confidence"], max(d["probs"]), abs_tol=0.11)


@pytest.mark.parametrize("junk", ["", "!!!", "data:image/png;base64,AAAA", "aGVsbG8="])
def test_mnist_bad_image_is_400(client, junk):
    assert client.post("/api/mnist/predict", json={"image": junk}).status_code == 400


# ---- text classifiers
@given(st.text(max_size=300))
def test_sentiment(client, text):
    resp = client.post("/api/sentiment/analyze", json={"text": text})
    d = check_response(resp)
    if not text.strip():
        assert resp.status_code == 400
        return
    assert d["label"] in d["scores"] and abs(sum(d["scores"].values()) - 100) < 1
    assert d["scores"][d["label"]] == max(d["scores"].values())


@given(st.text(max_size=300))
def test_spam(client, text):
    resp = client.post("/api/spam/analyze", json={"text": text})
    d = check_response(resp)
    if not text.strip():
        assert resp.status_code == 400
        return
    assert abs(d["spam_prob"] + d["ham_prob"] - 100) < 0.2
    assert d["label"] == ("SPAM" if d["spam_prob"] > 50 else "HAM") or d["spam_prob"] == 50.0


@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=["Zs", "Cc", "Zl", "Zp"]), min_size=1, max_size=12), max_size=40))
def test_ner_tokens_and_entities(client, server, words):
    text = " ".join(words)
    resp = client.post("/api/ner/analyze", json={"text": text})
    d = check_response(resp)
    if not text.strip():
        assert resp.status_code == 400
        return
    tokens = [t["token"] for t in d["tokens"]]
    assert tokens == text.split()[:server.load_ner().max_len]
    joined = " ".join(tokens)
    for ent in d["entities"]:
        assert ent["text"] in joined and ent["type"]


# ---- translator
@given(st.text(max_size=400), st.sampled_from(["en2ru", "ru2en"]))
@settings(max_examples=60)
def test_translate_any_text(client, text, direction):
    resp = client.post("/api/translate", json={"text": text, "direction": direction})
    d = check_response(resp)
    if not text.strip():
        assert resp.status_code == 400
    else:
        assert isinstance(d["translation"], str)


def test_translate_long_output_does_not_overflow_positions(client, server):
    model = server.load_translator("en2ru").model
    eos = 2
    orig = model.fc.forward

    def never_eos(x):
        out = orig(x)
        out[..., eos] = -1e9
        return out
    model.fc.forward = never_eos
    try:
        d = check_response(client.post("/api/translate", json={"text": "hello world", "direction": "en2ru"}))
        assert "translation" in d
    finally:
        model.fc.forward = orig


def test_translate_unknown_direction(client):
    assert client.post("/api/translate", json={"text": "hi", "direction": "xx"}).status_code == 400


# ---- GPT stream
@given(st.text(max_size=60), st.integers(-5, 25) | json_any,
       st.floats(-2, 5, allow_nan=False) | st.sampled_from([0, "0", "abc", None, 1e308]))
@settings(max_examples=60)
def test_gpt_stream_always_terminates(client, seed, length, temp):
    resp = client.post("/api/gpt/stream", json={"seed": seed, "length": length, "temperature": temp})
    out = check_response(resp)
    if isinstance(out, list):
        assert len(out) <= 402


# ---- GAN
@given(st.integers(-3, 12) | json_any, st.integers(-5, 30) | json_any)
@settings(max_examples=80)
def test_gan(client, digit, count):
    resp = client.post("/api/gan/generate", json={"digit": digit, "count": count})
    d = check_response(resp)
    if resp.status_code == 200:
        assert 1 <= len(d["images"]) <= 16
        assert all(i.startswith("data:image/png;base64,") for i in d["images"])
        if isinstance(count, int) and not isinstance(count, bool):
            assert len(d["images"]) == min(max(count, 1), 16)


# ---- tabular models: output must be finite, labels consistent
tab_num = st.floats(-1e4, 1e4, allow_nan=False)


@given(st.fixed_dictionaries({k: tab_num for k in ["area", "rooms", "floor", "district", "age"]}))
def test_price_finite(client, feats):
    d = check_response(client.post("/api/price/predict", json=feats))
    assert math.isfinite(d["price"]) and d["price_str"].startswith(f"{d['price']:.1f}")


@pytest.mark.parametrize("field", ["area", "rooms"])
@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "1e999", "abc"])
def test_price_rejects_non_finite(client, field, bad):
    assert client.post("/api/price/predict", json={field: bad}).status_code == 400


@given(st.lists(st.floats(0, 1000, allow_nan=False), min_size=0, max_size=40))
def test_timeseries(client, vals):
    resp = client.post("/api/timeseries/forecast", json={"values": vals})
    d = check_response(resp)
    if 0 < len(vals) < 30:
        assert resp.status_code == 400
        return
    assert len(d["forecast"]) == 7 and all(math.isfinite(v) for v in d["forecast"])


@given(st.fixed_dictionaries({k: tab_num for k in ["temp_today", "pressure", "humidity", "wind", "cloud", "month"]}))
def test_temperature(client, feats):
    d = check_response(client.post("/api/temperature/predict", json=feats))
    assert math.isfinite(d["temp_tomorrow"])


@given(st.fixed_dictionaries({k: tab_num for k in ["age", "orders", "avg", "total", "days", "visits"]}))
def test_cluster_customer(client, feats):
    d = check_response(client.post("/api/cluster/customer", json=feats))
    assert 0 <= d["cluster"] < len(d["all_dists"])
    assert 0 <= d["confidence"] <= 1
    assert d["segment"] in ("VIP", "Активные", "Молодые", "Пассивные")


@given(st.fixed_dictionaries({k: tab_num for k in ["amount", "hour", "freq", "foreign", "online", "balance", "distance"]}))
def test_anomaly(client, feats):
    d = check_response(client.post("/api/anomaly/check", json=feats))
    assert d["risk"] >= 0 and math.isfinite(d["recon_err"])
    want = "ВЫСОКИЙ РИСК" if d["risk"] > 1.5 else "Подозрительно" if d["risk"] > 1.0 else "Норма"
    assert d["label"] == want
    assert 0 <= d["risk_pct"] <= 100


@given(st.fixed_dictionaries({k: tab_num for k in ["thickness", "mass", "hardness", "roughness", "length", "width", "temp", "time"]}))
def test_defect(client, feats):
    d = check_response(client.post("/api/defect/check", json=feats))
    assert abs(d["defect_prob"] + d["ok_prob"] - 100) < 0.2
    assert d["label"] == ("БРАК" if d["defect_prob"] > 50 else "НОРМА") or d["defect_prob"] == 50.0


# ---- summarize / document clustering
sentence = st.text(alphabet="абвгдеёжзийклмнопрстуфхцчшщыэюяabcdefghij ", min_size=12, max_size=60).map(lambda s: s.strip() + ".")


@given(st.lists(sentence, min_size=1, max_size=15), st.integers(-3, 20) | json_any)
def test_summarize(client, sents, n):
    text = " ".join(sents)
    resp = client.post("/api/summarize", json={"text": text, "n_sentences": n})
    d = check_response(resp)
    if resp.status_code != 200:
        return
    sel = d["selected"]
    assert sel == sorted(set(sel)) and all(0 <= i < len(d["sentences"]) for i in sel)
    if len(d["sentences"]) > len(sel):
        assert d["summary"] == " ".join(d["sentences"][i] for i in sel)
        assert len(sel) >= 1


@given(st.lists(st.text(max_size=80) | st.integers() | st.none(), max_size=12), st.integers(-3, 15) | json_any)
def test_cluster_documents(client, docs, k):
    resp = client.post("/api/cluster/documents", json={"documents": docs, "k": k})
    d = check_response(resp)
    if resp.status_code != 200:
        return
    assert 1 <= d["k"] <= len(d["clusters"]) and len(d["topics"]) == d["k"]
    assert all(0 <= c["cluster"] < d["k"] for c in d["clusters"])


# ---- recommender
@pytest.fixture(scope="module")
def catalog(client):
    return [m["name"] for m in check_response(client.get("/api/recommender/catalog"))["movies"]]


@given(data=st.data())
def test_recommender(client, catalog, data):
    liked = data.draw(st.lists(st.sampled_from(catalog), min_size=1, max_size=5, unique=True))
    top_n = data.draw(st.integers(-3, 30))
    d = check_response(client.post("/api/recommender/recommend", json={"liked": liked, "top_n": top_n}))
    recs = d["recommendations"]
    assert not {r["name"] for r in recs} & set(liked)
    assert len(recs) == min(max(top_n, 1), len(catalog) - len(liked))
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


# ---- window commands
@pytest.mark.parametrize("action", ["minimize", "maximize", "hide", "quit", "startmove", "startresize_se"])
def test_win_command_written(client, server, action):
    check_response(client.post(f"/api/win/{action}"))
    assert server._WIN_CMD.read_text() == action


@pytest.mark.parametrize("action", ["reboot", "startresize_", "startresize_xx", "..%2f..%2fx"])
def test_win_command_unknown(client, action):
    assert client.post(f"/api/win/{action}").status_code in (400, 404)


# ---- LM Studio offline paths
@pytest.mark.parametrize("path", ["/api/chat", "/api/homework/solve"])
def test_lm_offline_is_reported(client, path):
    out = check_response(client.post(path, json={"message": "hi", "text": "hi"}))
    blob = str(out)
    assert "офлайн" in blob or "не запущ" in blob or "LM Studio" in blob


def test_unload_all_models(client, server):
    server.load_price()
    d = check_response(client.post("/api/unload/all"))
    assert "price" in d["removed"] and "price" not in server._cache


@given(st.sampled_from(["gpt", "mnist", "translator", "price", "nothing", "all"]))
@settings(max_examples=12)
def test_unload_any(client, name):
    check_response(client.post(f"/api/unload/{name}"))
