"""«Угадай рисунок»: несколько символов на одном холсте (sketch_segment.py и API)."""
import base64
import io
import random

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from PIL import Image, ImageDraw

import sketch_segment as seg
from conftest import check_response

SIZE, PEN = 400, 24          # как холст на странице


def canvas(*shapes, pen=PEN):
    """Рисует как страница: белые линии с круглыми концами на чёрном."""
    img = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(img)
    for kind, *a in shapes:
        if kind == "line":
            (x0, y0, x1, y1) = a
            d.line([(x0, y0), (x1, y1)], fill=255, width=pen)
            for x, y in ((x0, y0), (x1, y1)):
                d.ellipse([x - pen // 2, y - pen // 2, x + pen // 2, y + pen // 2], fill=255)
        elif kind == "oval":
            (cx, cy, rx, ry) = a
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=255, width=pen)
        elif kind == "dot":
            (x, y) = a
            d.ellipse([x - pen // 2, y - pen // 2, x + pen // 2, y + pen // 2], fill=255)
        elif kind == "arc":                       # «C»: овал с разрывом справа
            (cx, cy, rx, ry) = a
            d.arc([cx - rx, cy - ry, cx + rx, cy + ry], 40, 320, fill=255, width=pen)
    return np.asarray(img, dtype=np.float32) / 255.0


def png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ink_x(part):
    xs = np.nonzero((part > seg.INK).any(axis=0))[0]
    return xs.min(), xs.max()


# ---- разрезание ------------------------------------------------------------------

def test_sixty_five_is_two_symbols_with_a_detached_bar():
    arr = canvas(("oval", 100, 230, 50, 60), ("line", 70, 120, 100, 170),     # «6»
                 ("line", 250, 100, 350, 100),                               # черта «5» отдельно
                 ("line", 250, 130, 245, 210), ("oval", 290, 260, 50, 55))    # крючок «5»
    parts = seg.split(arr)
    assert len(parts) == 2
    assert ink_x(parts[0])[1] < ink_x(parts[1])[0], "символы не слева направо"
    assert (parts[1][100, 300] > 0.5), "черта «5» оторвалась от своего символа"


@pytest.mark.parametrize("name, shapes", [
    ("=", [("line", 100, 170, 300, 170), ("line", 110, 240, 290, 240)]),
    ("i", [("line", 200, 160, 200, 330), ("dot", 200, 100)]),
    ("÷", [("line", 100, 200, 300, 200), ("dot", 200, 130), ("dot", 200, 270)]),
    ("+", [("line", 100, 200, 300, 200), ("line", 200, 100, 200, 300)]),
    ("круг", [("oval", 200, 200, 120, 120)]),
])
def test_one_symbol_is_not_split(name, shapes):
    assert seg.split(canvas(*shapes)) == [], name


def test_thin_diagonal_stays_one_stroke():
    """Тонкая наклонная линия держится только углами пикселей: при поиске пятен
    по четырём соседям она рассыпалась бы на десятки «символов»."""
    arr = canvas(("line", 40, 360, 200, 40), ("line", 320, 60, 320, 340), pen=2)
    assert len(seg.split(arr)) == 2


def test_one_pixel_diagonal_at_full_resolution():
    """На маленькой картинке уменьшения нет, и линия в один пиксель держится
    только углами: соседи по диагонали обязаны считаться соседями."""
    img = Image.new("L", (120, 120), 0)
    d = ImageDraw.Draw(img)
    d.line([(5, 110), (50, 10)], fill=255, width=1)
    d.line([(90, 10), (90, 110)], fill=255, width=1)
    assert len(seg.split(np.asarray(img, np.float32) / 255.0)) == 2


def test_faint_halo_is_not_ink():
    """Браузер сглаживает края линий: вокруг краски бледный ореол. В кусок
    символа идёт только краска, ореол соседних символов не липнет."""
    arr = canvas(("line", 110, 100, 110, 300), ("oval", 260, 200, 60, 100))
    halo = np.zeros_like(arr)
    halo[1:, :] = np.maximum(halo[1:, :], arr[:-1, :])
    halo[:, 1:] = np.maximum(halo[:, 1:], arr[:, :-1])
    arr = np.where(arr > 0, arr, halo * 0.05).astype(np.float32)
    parts = seg.split(arr)
    assert len(parts) == 2
    assert all(((p == 0) | (p > seg.INK)).all() for p in parts)


def test_dust_does_not_become_a_symbol():
    arr = canvas(("line", 110, 100, 110, 300), ("oval", 260, 200, 60, 100))
    arr[380:382, 390:392] = 1.0           # пылинка в углу
    assert len(seg.split(arr)) == 2


def test_overlap_chains_through_the_growing_symbol():
    """Три черты лесенкой: вторая перекрывает первую, третья - только вторую.
    Всё это один символ: границы символа растут вместе с ним."""
    arr = canvas(("line", 100, 100, 200, 100), ("line", 150, 200, 300, 200), ("line", 260, 300, 320, 300))
    assert seg.split(arr) == []


def test_big_picture_is_fast():
    import time
    big = np.zeros((2000, 2000), np.float32)
    big[200:1800, 200:700] = 1.0
    big[200:1800, 1200:1700] = 1.0
    big[::97, ::89] = np.maximum(big[::97, ::89], 0.5)      # редкий сор по всему полю
    t = time.perf_counter()
    assert len(seg.split(big)) == 2
    assert time.perf_counter() - t < 1.0, "разрезание большой картинки слишком медленное"


def test_empty_canvas_and_many_spots():
    assert seg.split(np.zeros((50, 50), np.float32)) == []
    dots = canvas(*[("dot", 20 + 25 * k, 200) for k in range(15)], pen=10)
    assert seg.split(dots) == [], "15 точек в ряд - не надпись"


@given(st.lists(st.tuples(st.integers(0, 399), st.integers(0, 399), st.integers(0, 399), st.integers(0, 399)),
                min_size=1, max_size=6), st.integers(6, 40))
@settings(max_examples=120)
def test_split_invariants(lines, pen):
    """Любой рисунок: кусков 0 или от 2 до MAX_SYMBOLS; каждый кусок - часть
    исходной краски без выдумок; куски не делят краску между собой и идут
    слева направо; всё, что не сор, попадает в какой-то кусок."""
    arr = canvas(*[("line", *l) for l in lines], pen=pen)
    parts = seg.split(arr)
    assert parts == [] or 2 <= len(parts) <= seg.MAX_SYMBOLS
    if not parts:
        return
    stack = np.stack(parts)
    assert ((stack == 0) | (stack == arr)).all(), "в куске появилась краска, которой не было"
    assert ((stack == 0) | (stack > seg.INK)).all(), "в кусок попал фон вокруг краски"
    assert ((stack > 0).sum(axis=0) <= 1).all(), "одна и та же краска в двух символах"
    assert all((p > seg.INK).any() for p in parts)
    starts = [ink_x(p)[0] for p in parts]
    assert starts == sorted(starts)
    lost = ((arr > seg.INK) & ~(stack > 0).any(axis=0)).sum()
    assert lost <= seg.MIN_SHARE * seg.MAX_SYMBOLS * (arr > seg.INK).sum() + 1


# ---- чтение ----------------------------------------------------------------------

LABELS = ["0", "1", "5", "6", "b", "S", "O", "A", "|", "°"]


def probs(**kw):
    alias = {"six": "6", "five": "5", "zero": "0", "bar": "|", "deg": "°"}
    p = np.full(len(LABELS), 1e-4)
    for k, v in kw.items():
        p[LABELS.index(alias.get(k, k))] = v
    return p / p.sum()


def read(*ps):
    chosen, conf, sure = seg.read_sequence(list(ps), LABELS)
    return "".join(LABELS[c] for c in chosen), conf, sure


def test_reading_rules():
    assert read(probs(b=0.55, six=0.05), probs(five=0.9))[0] == "65"
    assert read(probs(bar=0.99), probs(deg=0.9))[0] == "10"
    assert read(probs(A=0.9, six=0.05), probs(five=0.9))[0] == "A5"
    assert read(probs(S=0.8), probs(O=0.8), probs(S=0.8))[0] == "SOS"
    assert read(probs(b=0.6))[0] == "b"
    assert read(probs(bar=0.99))[0] == "|", "одиночный символ не читается как число"
    text, _, sure = read(probs(A=0.6, six=0.3), probs(five=0.9))
    assert text == "65" and sure[0] == pytest.approx(probs(A=0.6, six=0.3)[3])


WORD_LABELS = ["2", "5", "a", "b", "c", "C", "d", "x", "X", "+", "=", "-", "рыба"]


def wprobs(**kw):
    alias = {"two": "2", "five": "5", "plus": "+", "eq": "=", "minus": "-", "fish": "рыба"}
    p = np.full(len(WORD_LABELS), 1e-4)
    for k, v in kw.items():
        p[WORD_LABELS.index(alias.get(k, k))] = v
    return p / p.sum()


def wread(*ps, heights=None):
    chosen, _, _ = seg.read_sequence(list(ps), WORD_LABELS, heights)
    return "".join(WORD_LABELS[c] for c in chosen)


def test_case_follows_height_next_to_tall_symbols():
    assert wread(wprobs(b=0.9), wprobs(C=0.9), heights=[200, 110]) == "bc"
    assert wread(wprobs(b=0.9), wprobs(c=0.9), heights=[200, 190]) == "bC"
    assert wread(wprobs(C=0.9), wprobs(x=0.9), heights=[100, 200]) == "Cx", "без высокого символа регистр не трогаем"
    assert wread(wprobs(b=0.9), wprobs(C=0.9)) == "bC", "без высот регистр не трогаем"
    assert wread(wprobs(C=0.9), heights=[50]) == "C", "один символ - сравнить не с чем"
    assert wread(wprobs(b=0.9), wprobs(d=0.9), wprobs(C=0.9), heights=[200, 120, 110]) == "bdc", \
        "рост строки - по самому высокому символу, а не по самому низкому"


def test_word_context_pulls_a_doubtful_symbol_to_a_letter():
    """Рукописную «a» сеть часто зовёт «2», а «a» ставит второй."""
    assert wread(wprobs(two=0.6, a=0.25), wprobs(b=0.9), wprobs(c=0.9), heights=[110, 200, 110]) == "abc"
    assert wread(wprobs(two=0.6, a=0.05), wprobs(b=0.9), wprobs(c=0.9)) == "2bc", "буква с 5 % не перебивает"
    assert wread(wprobs(two=0.6, a=0.25), wprobs(five=0.9)) == "25", "число важнее слова"
    assert wread(wprobs(two=0.6, a=0.25), wprobs(plus=0.9), wprobs(eq=0.9)) == "2+=", "букв меньшинство - не слово"
    assert wread(wprobs(two=0.6, a=0.25), wprobs(b=0.9), wprobs(plus=0.9)) == "2b+", "одна буква из трёх - не слово"


def test_picture_inside_an_inscription_becomes_a_sign():
    assert wread(wprobs(fish=0.5, plus=0.2), wprobs(eq=0.9), wprobs(minus=0.9)) == "+=-"
    assert wread(wprobs(fish=0.5, plus=0.05), wprobs(eq=0.9), wprobs(minus=0.9)) == "рыба=-"
    assert wread(wprobs(fish=0.5, plus=0.2)) == "рыба", "одна картинка остаётся картинкой"


@pytest.mark.parametrize("labels", [LABELS, WORD_LABELS])
def test_random_reads(labels):
    """Любые вероятности и высоты: уверенность - произведение и в [0, 1];
    в числе (цифр не меньше половины) цифра остаётся цифрой; символ меняется
    только на класс, у которого не меньше 10 %, или на свою пару (похожая
    цифра, другой регистр)."""
    rng = np.random.default_rng(1)
    for _ in range(1000):
        n = int(rng.integers(1, 6))
        ps = [rng.dirichlet(np.full(len(labels), 0.3)) for _ in range(n)]
        heights = list(rng.integers(20, 300, n)) if rng.random() < 0.5 else None
        chosen, conf, sure = seg.read_sequence(ps, labels, heights)
        assert conf == pytest.approx(float(np.prod(sure))) and 0 <= conf <= 1 + 1e-9
        assert all(0 <= s <= 1 + 1e-9 for s in sure)
        tops = [int(np.argmax(p)) for p in ps]
        numeric = sum(labels[t] in seg.DIGITS for t in tops) * 2 >= n
        for p, c in zip(ps, chosen):
            t = int(np.argmax(p))
            if numeric and labels[t] in seg.DIGITS:
                assert c == t
            if c != t:
                pair = seg.LOOKALIKES.get(labels[t]) == labels[c] or labels[t].lower() == labels[c].lower()
                # цепочка «спорный -> буква с >= 10 % -> та же буква в другом регистре»
                other_case = labels[c].swapcase()
                via_case = other_case in labels and p[labels.index(other_case)] >= 0.10 - 1e-9
                assert pair or via_case or p[c] >= 0.10 - 1e-9, (labels[t], labels[c], p[c])
        if n == 1:
            assert chosen == [int(np.argmax(ps[0]))]


def test_alternative_skips_the_source_of_a_lookalike():
    p = probs(bar=0.9, A=0.05)
    assert LABELS[seg.alternative(p, LABELS.index("1"), LABELS)] == "A"


# ---- API ------------------------------------------------------------------------

def test_api_single_symbol_keeps_the_old_answer(client):
    d = check_response(client.post("/api/sketch/predict", json={"image": png_b64(canvas(("oval", 200, 200, 90, 150))), "top": 5}))
    assert "symbols" not in d and "text" not in d
    assert len(d["guesses"]) == 5


@pytest.mark.parametrize("expected, shapes", [
    ("10", [("line", 110, 100, 110, 300), ("oval", 280, 200, 60, 100)]),
    ("11", [("line", 120, 100, 120, 300), ("line", 290, 100, 290, 300)]),
    ("70", [("line", 30, 110, 160, 110), ("line", 160, 110, 80, 310), ("oval", 290, 210, 60, 100)]),
    ("101", [("line", 60, 100, 60, 300), ("oval", 200, 200, 55, 100), ("line", 340, 100, 340, 300)]),
    ("ABC", [("line", 30, 300, 80, 100), ("line", 80, 100, 130, 300), ("line", 50, 220, 110, 220),
             ("line", 170, 100, 170, 300), ("oval", 200, 150, 40, 50), ("oval", 205, 250, 45, 50),
             ("arc", 330, 200, 55, 100)]),
    ("bc", [("line", 120, 90, 120, 300), ("oval", 165, 245, 45, 55), ("arc", 310, 245, 45, 55)]),
])
def test_api_reads_numbers_with_the_real_model(client, server, expected, shapes):
    d = check_response(client.post("/api/sketch/predict", json={"image": png_b64(canvas(*shapes)), "top": 3}))
    assert d["text"] == expected, d
    assert len(d["symbols"]) == len(expected)
    assert 0 <= d["confidence"] <= 100
    labels = server.load_sketch().labels
    for s in d["symbols"]:
        assert s["label"] in labels and s["alt"]["label"] in labels and s["alt"]["label"] != s["label"]
        assert 0 <= s["prob"] <= 100.1 and 0 <= s["alt"]["prob"] <= 100


def test_api_random_scribbles_are_consistent(client):
    rng = random.Random(4)
    for _ in range(25):
        shapes = [("line", *(rng.randint(0, 399) for _ in range(4))) for _ in range(rng.randint(1, 5))]
        d = check_response(client.post("/api/sketch/predict", json={"image": png_b64(canvas(*shapes))}))
        if "symbols" in d:
            assert 2 <= len(d["symbols"]) <= seg.MAX_SYMBOLS
            assert d["text"] == seg.join_labels([s["label"] for s in d["symbols"]])
            conf = np.prod([s["prob"] / 100 for s in d["symbols"]]) * 100
            assert abs(conf - d["confidence"]) <= 0.5 + 0.05 * len(d["symbols"])
