"""«Угадай рисунок»: разрезание картинки на символы и чтение числа или строки.

SketchNet обучена на одиночных символах (проект IntelegienceDrawer): весь
холст она ужимает в один кадр 28x28, и «65» превращается в «W». Браузер
присылает только картинку, без мазков, поэтому режем по ней: связные пятна
краски, а пятна, перекрывающиеся по горизонтали, - один символ («=», «i», «÷»,
«5» с отдельной чертой сверху). Стоящие рядом - разные символы, слева направо.

Правила те же, что в DrawGuess (segment.py): порог перекрытия, цифры и
похожие на них знаки. Модуль не знает ни про Flask, ни про сеть.
"""
import numpy as np

OVERLAP = 0.3        # доля более узкого символа, которую должно перекрыть соседнее пятно
GRID = 128           # пятна ищем на копии не больше GRID x GRID: быстро и без дыр в линиях
MIN_SHARE = 0.01     # пятна меньше 1 % всей краски - сор, символ из них не делаем
MAX_SYMBOLS = 12     # больше символов - это уже не надпись, а шум; читаем картинку целиком
INK = 0.1            # тот же порог краски, что в нормализации

DIGIT_SWITCH = 0.10
DIGITS = frozenset("0123456789")
# Одиночную палочку сеть называет «|», круглый ноль - «°»: для подсчёта «тут
# число?» они идут как цифры.
DIGIT_SHAPES = {"|": "1", "°": "0"}
# Похожие на цифры знаки меняются на цифры только рядом с цифрами:
# «S5» - это «55», а «SOS» остаётся «SOS».
LOOKALIKES = {**DIGIT_SHAPES, "l": "1", "I": "1", "i": "1", "/": "1", "O": "0", "o": "0",
              "D": "0", "Q": "0", "Z": "2", "z": "2", "S": "5", "s": "5", "b": "6",
              "G": "6", "T": "7", "B": "8", "g": "9", "q": "9"}
# Буквы, у которых строчная и заглавная - одна и та же форма. Нормализация
# растягивает каждый символ до одного размера, и сеть их не различает. В строке
# размер виден: такую букву сравниваем по высоте с символами, чей рост известен
# (цифры, заглавные, строчные с верхним выносом), и выбираем регистр.
SAME_SHAPE = frozenset("cosuvwxz")
TALL = frozenset("0123456789ABDEFGHIJKLMNPQRTYbdfhklt")
SMALL_RATIO = 0.75   # ниже 3/4 высоты высоких символов - строчная
# Внутри надписи «кошка» или «рыба» почти наверняка ошибка: если хотя бы
# половина символов - знаки, картинка меняется на лучший знак, если у него
# хотя бы столько.
CHAR_SWITCH = 0.10


def _label(mask):
    """Связные области (8 соседей) на маленькой маске: номер области для каждой клетки, 0 - пусто."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    n = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if labels[y0, x0]:
            continue
        n += 1
        labels[y0, x0] = n
        stack = [(y0, x0)]
        while stack:
            y, x = stack.pop()
            for yy in (y - 1, y, y + 1):
                for xx in (x - 1, x, x + 1):
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not labels[yy, xx]:
                        labels[yy, xx] = n
                        stack.append((yy, xx))
    return labels, n


def split(arr):
    """Картинка (float 0..1, белая краска на чёрном) -> картинки символов слева направо.

    Один символ или не больше чем MAX_SYMBOLS - список из стольких картинок той
    же формы, в каждой только краска своего символа. Если символ один или пятен
    слишком много (шум) - пустой список: читать картинку целиком, как раньше.
    """
    ink = arr > INK
    if not ink.any():
        return []
    h, w = arr.shape
    f = max(1, -(-max(h, w) // GRID))
    ph, pw = -(-h // f) * f, -(-w // f) * f
    padded = np.zeros((ph, pw), dtype=bool)
    padded[:h, :w] = ink
    small = padded.reshape(ph // f, f, pw // f, f).any(axis=(1, 3))
    labels, n = _label(small)
    if n < 2:
        return []

    full = np.repeat(np.repeat(labels, f, axis=0), f, axis=1)[:h, :w]
    full[~ink] = 0
    areas = np.bincount(full.ravel(), minlength=n + 1)
    total = areas[1:].sum()
    spots = []
    for k in range(1, n + 1):
        if areas[k] < MIN_SHARE * total:
            continue
        xs = np.nonzero((full == k).any(axis=0))[0]
        spots.append((int(xs.min()), int(xs.max()) + 1, k))
    if len(spots) < 2:
        return []

    groups = []                               # [x0, x1, [номера пятен]]
    for x0, x1, k in sorted(spots):
        if groups:
            g = groups[-1]
            inter = min(g[1], x1) - max(g[0], x0)
            if inter >= OVERLAP * min(x1 - x0, g[1] - g[0]):
                g[1] = max(g[1], x1)
                g[2].append(k)
                continue
        groups.append([x0, x1, [k]])
    if not 2 <= len(groups) <= MAX_SYMBOLS:
        return []
    return [np.where(np.isin(full, g[2]), arr, 0).astype(arr.dtype) for g in groups]


def ink_heights(parts):
    """Высота краски каждого куска в пикселях (для выбора регистра)."""
    out = []
    for part in parts:
        ys = np.nonzero((part > INK).any(axis=1))[0]
        out.append(int(ys.max() - ys.min() + 1) if len(ys) else 0)
    return out


def read_sequence(probs_list, labels, heights=None):
    """Вероятности по символам -> (выбранные классы, уверенность строки, уверенность символов).

    Правила по очереди:
    1. Число: если хотя бы половина символов - цифры (или «|», «°»), похожие
       знаки меняются на цифры, спорный символ - на лучшую цифру, если у неё
       не меньше DIGIT_SWITCH.
    2. Надпись: если хотя бы половина символов - знаки (не картинки вроде
       «кошка»), картинка меняется на лучший знак, если у него не меньше
       CHAR_SWITCH.
    3. Слово: если хотя бы половина - латинские буквы, спорный символ меняется
       на лучшую букву, если у неё не меньше CHAR_SWITCH.
    4. Регистр (если известны высоты): c, o, s, u, v, w, x, z ниже 3/4 роста
       высоких символов строки - строчные, иначе заглавные.
    Уверенность строки - произведение уверенностей символов.
    """
    chosen, sure = _read_numbers(probs_list, labels)
    if len(chosen) > 1:
        _prefer_characters(probs_list, labels, chosen, sure)
        _read_words(probs_list, labels, chosen, sure)
        if heights is not None:
            _fix_case(probs_list, labels, heights, chosen, sure)
    return chosen, float(np.prod(sure)), sure


def is_letter(name):
    return len(name) == 1 and name.isascii() and name.isalpha()


def _read_words(probs_list, labels, chosen, sure):
    """Слово: если хотя бы половина символов - латинские буквы (и строка не
    число), спорный символ меняется на лучшую букву, если у неё не меньше
    CHAR_SWITCH. Рукописную «a» сеть часто зовёт «2» или «d», а «a» ставит
    второй с 20 %: рядом с буквами это «a»."""
    letters = [i for i, name in enumerate(labels) if is_letter(name)]
    names = [labels[c] for c in chosen]
    numeric = sum(n in DIGITS for n in names) * 2 >= len(names)
    if not letters or numeric or sum(map(is_letter, names)) * 2 < len(names):
        return
    for n, p in enumerate(probs_list):
        if is_letter(labels[chosen[n]]):
            continue
        best = max(letters, key=lambda i: p[i])
        if p[best] >= CHAR_SWITCH:
            chosen[n] = best
            sure[n] = float(p[best])


def _prefer_characters(probs_list, labels, chosen, sure):
    chars = [i for i, name in enumerate(labels) if len(name) == 1]
    if not chars or sum(len(labels[c]) == 1 for c in chosen) * 2 < len(chosen):
        return
    for n, p in enumerate(probs_list):
        if len(labels[chosen[n]]) == 1:
            continue
        best = max(chars, key=lambda i: p[i])
        if p[best] >= CHAR_SWITCH:
            chosen[n] = best
            sure[n] = float(p[best])


def _fix_case(probs_list, labels, heights, chosen, sure):
    index = {name: i for i, name in enumerate(labels)}
    tall = [h for c, h in zip(chosen, heights) if labels[c] in TALL]
    if not tall:
        return
    ref = max(tall)
    for n, p in enumerate(probs_list):
        name = labels[chosen[n]]
        if name.lower() not in SAME_SHAPE:
            continue
        want = name.lower() if heights[n] < SMALL_RATIO * ref else name.upper()
        if want != name and want in index:
            twin = index[want]
            sure[n] = float(p[chosen[n]] + p[twin])
            chosen[n] = twin


def _read_numbers(probs_list, labels):
    index = {name: i for i, name in enumerate(labels)}
    tops = [int(np.argmax(p)) for p in probs_list]
    chosen = list(tops)
    sure = [float(p[t]) for p, t in zip(probs_list, tops)]
    digit_idx = [i for i, name in enumerate(labels) if name in DIGITS]
    digitish = sum(labels[t] in DIGITS or labels[t] in DIGIT_SHAPES for t in tops)
    if digit_idx and len(tops) > 1 and digitish * 2 >= len(tops):
        for n, p in enumerate(probs_list):
            name = labels[tops[n]]
            if name in DIGITS:
                continue
            twin = LOOKALIKES.get(name)
            if twin in index:
                chosen[n] = index[twin]
                sure[n] = float(p[tops[n]] + p[index[twin]])
                continue
            best = max(digit_idx, key=lambda i: p[i])
            if p[best] >= DIGIT_SWITCH:
                chosen[n] = best
                sure[n] = float(p[best])
    return chosen, sure


def join_labels(names):
    """«6», «5» -> «65»; «солнце», «дерево» -> «солнце дерево»."""
    return "".join(names) if all(len(n) == 1 for n in names) else " ".join(names)


def alternative(p, chosen, labels):
    """Лучший другой вариант символа, кроме выбранного и того, из которого он
    получился («|» у единицы, «C» у «c» показывать незачем)."""
    name = labels[chosen]
    skip = {chosen} | {i for i, other in enumerate(labels)
                       if LOOKALIKES.get(other) == name
                       or (other != name and other.lower() == name.lower() and name.lower() in SAME_SHAPE)}
    return max((i for i in range(len(p)) if i not in skip), key=lambda i: p[i])
