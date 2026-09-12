"""
NER (распознавание именованных сущностей), русский + английский.
Теги: O, B/I-PER (люди), B/I-ORG (организации), B/I-LOC (места), B/I-DATE (даты).

Данные:
  * WikiANN ru + en (unimelb-nlp/wikiann) — 40 тыс. размеченных фрагментов Википедии (PER/ORG/LOC);
  * шаблонные предложения: сущности из WikiANN подставляются в живые фразы с датами —
    так модель учит DATE и видит сущности внутри обычного текста, а не только в заголовках;
  * ручные примеры RAW (разметка прямо в тексте: [Илон Маск|PER]).
Модель: BiLSTM по словам + CNN по символам (регистр букв и суффиксы важны для имён).
Запуск из папки lab: python train_ner.py → ner_model.pth (скопировать в ../models).
"""
import random
import re
import sys
import time
from collections import Counter

import torch
import torch.nn as nn

from hf_data import load_parquet, tokenize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE", "I-DATE"]
T2I = {t: i for i, t in enumerate(TAGS)}      # номера 0..6 совпадают с WikiANN
MAX_CHARS, MAX_TOKENS = 20, 64

RAW = """
[Elon Musk|PER] founded [SpaceX|ORG] in [California|LOC] in [2002|DATE].
[Barack Obama|PER] was born in [Hawaii|LOC] on [August 4, 1961|DATE].
[Marie Curie|PER] received the Nobel Prize in [1903|DATE].
[Steve Jobs|PER] and [Steve Wozniak|PER] co-founded [Apple|ORG] in [Los Angeles|LOC].
[Albert Einstein|PER] published his theory in [1905|DATE] in [Bern|LOC].
[Jeff Bezos|PER] started [Amazon|ORG] in [Seattle|LOC] in [July 1994|DATE].
[Ada Lovelace|PER] is considered the first programmer.
[Leonardo da Vinci|PER] lived in [Florence|LOC] during the Renaissance.
[Mark Zuckerberg|PER] launched [Facebook|ORG] from [Harvard|ORG] in [2004|DATE].
[Isaac Newton|PER] was born on [January 4, 1643|DATE].
[Google|ORG] was founded in [1998|DATE] by [Larry Page|PER].
[Microsoft|ORG] released Windows 95 in [August 1995|DATE].
[Tesla|ORG] opened a factory in [Berlin|LOC] in [2021|DATE].
[OpenAI|ORG] was created in [San Francisco|LOC] in [December 2015|DATE].
[NASA|ORG] launched Apollo 11 on [July 16, 1969|DATE].
The [United Nations|ORG] was founded in [1945|DATE] in [New York|LOC].
[Amazon|ORG] acquired [Whole Foods|ORG] in [2017|DATE] for 13.7 billion dollars.
[Paris|LOC] is the capital of [France|LOC] and is located in [Europe|LOC].
The [Amazon River|LOC] flows through [Brazil|LOC] and [Peru|LOC].
[Mount Everest|LOC] is located in [Nepal|LOC] near [Tibet|LOC].
The [Great Wall of China|LOC] was built over many centuries.
[Tokyo|LOC] is the largest city in [Japan|LOC] with over 13 million people.
The meeting is scheduled for [Monday, March 10, 2025|DATE].
World War II ended on [September 2, 1945|DATE].
He was born on [April 15|DATE] and died in [November 2001|DATE].
In [2023|DATE], [OpenAI|ORG] released GPT-4 in [San Francisco|LOC].
[Bill Gates|PER] and [Paul Allen|PER] founded [Microsoft|ORG] in [Albuquerque|LOC] in [1975|DATE].
[Apple|ORG] announced the iPhone on [January 9, 2007|DATE] in [San Francisco|LOC].
The [Berlin Wall|LOC] fell on [November 9, 1989|DATE].
[SpaceX|ORG] launched its first Falcon rocket from [Cape Canaveral|LOC] in [March 2006|DATE].
Queen [Elizabeth II|PER] died on [September 8, 2022|DATE] at [Balmoral Castle|LOC].
[Nikola Tesla|PER] was born in [Serbia|LOC] on [July 10, 1856|DATE].
I bought 3 apples and 12 oranges for 5 dollars.
The price rose by 20 percent last year.
Please call me at 10 o'clock tomorrow.
[Александр Пушкин|PER] родился в [Москве|LOC] [6 июня 1799 года|DATE].
[Лев Толстой|PER] жил в [Ясной Поляне|LOC] в [Тульской области|LOC].
[Юрий Гагарин|PER] совершил первый полёт в космос [12 апреля 1961 года|DATE].
[Михаил Ломоносов|PER] основал [Московский университет|ORG] в [1755 году|DATE].
[Фёдор Достоевский|PER] написал роман в [Санкт-Петербурге|LOC] в [1866 году|DATE].
[Пётр Чайковский|PER] родился в [Воткинске|LOC] в [мае 1840 года|DATE].
[Дмитрий Менделеев|PER] создал периодическую таблицу элементов в [1869 году|DATE].
[Антон Чехов|PER] жил в [Ялте|LOC] в последние годы своей жизни.
[Сергей Брин|PER] и [Ларри Пейдж|PER] основали [Google|ORG] в [1998 году|DATE].
[Анна Ахматова|PER] написала стихи в блокадном [Ленинграде|LOC] в [1941 году|DATE].
[Яндекс|ORG] был основан в [Москве|LOC] в [1997 году|DATE].
[Сбербанк|ORG] открыл новый офис в [Москве|LOC] в [марте 2023 года|DATE].
[Газпром|ORG] добывает природный газ в [Сибири|LOC] и экспортирует в [Европу|LOC].
[МГУ|ORG] расположен в [Москве|LOC] на [Воробьёвых горах|LOC].
[Роснефть|ORG] подписала договор с китайскими компаниями в [декабре 2022 года|DATE].
[ВТБ|ORG] и [Сбербанк|ORG] являются крупнейшими банками [России|LOC].
[Аэрофлот|ORG] выполняет рейсы из [Москвы|LOC] в [Токио|LOC] и [Пекин|LOC].
[РЖД|ORG] запустила новый маршрут из [Петербурга|LOC] в [Хельсинки|LOC] в [январе 2024|DATE].
[Москва|LOC] является столицей [России|LOC] и расположена на реке [Москве|LOC].
[Байкал|LOC] находится в [Сибири|LOC] и является самым глубоким озером в мире.
[Волга|LOC] течёт через [Казань|LOC] и впадает в [Каспийское море|LOC].
[Санкт-Петербург|LOC] был основан [Петром Первым|PER] в [1703 году|DATE].
[Уральские горы|LOC] разделяют [Европу|LOC] и [Азию|LOC].
[Владивосток|LOC] находится на [Дальнем Востоке|LOC] у берегов [Японского моря|LOC].
Великая Отечественная война закончилась [9 мая 1945 года|DATE].
Революция произошла в [октябре 1917 года|DATE] в [России|LOC].
Олимпийские игры в [Сочи|LOC] прошли в [феврале 2014 года|DATE].
Чемпионат мира по футболу в [России|LOC] состоялся летом [2018 года|DATE].
[Владимир Путин|PER] посетил [Санкт-Петербург|LOC] в [январе 2024 года|DATE].
[Google|ORG] открыла офис в [Москве|LOC] в [2006 году|DATE].
[Apple|ORG] представила iPhone X в [Купертино|LOC] в [сентябре 2017 года|DATE].
[ООН|ORG] провела встречу в [Женеве|LOC] в [декабре 2023 года|DATE].
[Илон Маск|PER] посетил [Москву|LOC] в [марте 2022 года|DATE].
Я купил 3 яблока и 12 апельсинов за 500 рублей.
Цены выросли на 20 процентов за год.
Позвони мне завтра в 10 часов.
"""

SPAN_RE = re.compile(r"\[([^\]|]+)\|(PER|ORG|LOC|DATE)\]")


def parse_annotated(line):
    """'[Илон Маск|PER] посетил ...' → (токены, теги)."""
    tokens, tags, pos = [], [], 0
    for m in SPAN_RE.finditer(line):
        for t in tokenize(line[pos:m.start()]):
            tokens.append(t); tags.append("O")
        for i, t in enumerate(tokenize(m.group(1))):
            tokens.append(t); tags.append(("B-" if i == 0 else "I-") + m.group(2))
        pos = m.end()
    for t in tokenize(line[pos:]):
        tokens.append(t); tags.append("O")
    return tokens, [T2I[t] for t in tags]


# ── Шаблоны с датами ─────────────────────────────────────────────────────────
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
             "October", "November", "December"]
MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября",
              "октября", "ноября", "декабря"]
MONTHS_PREP = ["январе", "феврале", "марте", "апреле", "мае", "июне", "июле", "августе", "сентябре",
               "октябре", "ноябре", "декабре"]


def year():
    return str(random.choice([random.randint(1700, 2030), random.randint(1950, 2026)]))


def date_on(lang):   # полная дата: «12 апреля 1961 года», «July 16, 1969»
    d, m = random.randint(1, 28), random.randrange(12)
    if lang == "ru":
        return f"{d} {MONTHS_GEN[m]} {year()} года"
    return random.choice([f"{MONTHS_EN[m]} {d}, {year()}", f"{d} {MONTHS_EN[m]} {year()}"])


def date_in(lang):   # после «в»/«in»: «1998 году», «мае 1840 года», «July 1994», «2002»
    m = random.randrange(12)
    if lang == "ru":
        return random.choice([f"{year()} году", f"{MONTHS_PREP[m]} {year()} года", f"{MONTHS_PREP[m]} {year()}"])
    return random.choice([year(), f"{MONTHS_EN[m]} {year()}"])


TEMPLATES = {
    "ru": [
        "{PER} родился в {LOC} {ON}.", "{PER} переехал в {LOC} в {IN}.", "{ORG} была основана в {IN}.",
        "{PER} возглавил {ORG} в {IN}.", "Компания {ORG} открыла офис в {LOC}.",
        "{PER} выступил на конференции в {LOC} {ON}.", "В {IN} {PER} получил премию.",
        "{ORG} объявила о сделке {ON}.", "{PER} и {PER} встретились в {LOC}.",
        "Штаб-квартира {ORG} находится в {LOC}.", "{ON} {PER} посетил {LOC}.",
        "Соглашение между {ORG} и {ORG} подписано в {IN}.", "{PER} работает в {ORG} с {IN}.",
        "Мы поедем в {LOC} в {IN}.", "Встреча назначена на {ON}.", "По словам {PER}, {ORG} растёт.",
        "{LOC} — один из крупнейших городов региона.", "Туристы из {LOC} приехали в {LOC}.",
    ],
    "en": [
        "{PER} was born in {LOC} on {ON}.", "{PER} moved to {LOC} in {IN}.", "{ORG} was founded in {IN}.",
        "{PER} joined {ORG} in {IN}.", "{ORG} opened a new office in {LOC}.",
        "{PER} spoke at a conference in {LOC} on {ON}.", "In {IN}, {PER} won the award.",
        "{ORG} announced the deal on {ON}.", "{PER} met {PER} in {LOC}.",
        "The headquarters of {ORG} is in {LOC}.", "On {ON}, {PER} visited {LOC}.",
        "{ORG} and {ORG} signed an agreement in {IN}.", "{PER} has worked at {ORG} since {IN}.",
        "We will travel to {LOC} in {IN}.", "The meeting is scheduled for {ON}.",
        "According to {PER}, {ORG} is growing.", "{LOC} is one of the largest cities in the region.",
        "Tourists from {LOC} arrived in {LOC}.",
    ],
}
SLOT_RE = re.compile(r"\{(PER|ORG|LOC|ON|IN)\}")


def harvest_entities(rows):
    """Сущности из WikiANN: {'PER': [['Лев', 'Толстой'], ...], ...} — только 1-4 «чистых» слова."""
    out = {"PER": [], "ORG": [], "LOC": []}
    for r in rows:
        cur, typ = [], None
        for tok, tag in list(zip(r["tokens"], r["ner_tags"])) + [("", 0)]:
            name = TAGS[tag]
            if name.startswith("I-") and typ == name[2:]:
                cur.append(tok)
                continue
            if cur and len(cur) <= 4 and all(re.fullmatch(r"[\w.-]+", t) for t in cur):
                out[typ].append(cur)
            cur, typ = ([tok], name[2:]) if name.startswith("B-") else ([], None)
    return out


def from_template(lang, ents):
    tpl = random.choice(TEMPLATES[lang])
    line, pos = "", 0
    for m in SLOT_RE.finditer(tpl):
        slot = m.group(1)
        if slot == "ON":
            val, typ = date_on(lang), "DATE"
        elif slot == "IN":
            val, typ = date_in(lang), "DATE"
        else:
            val, typ = " ".join(random.choice(ents[slot])), slot
        line += tpl[pos:m.start()] + f"[{val}|{typ}]"
        pos = m.end()
    return parse_annotated(line + tpl[pos:])


# ── Модель ───────────────────────────────────────────────────────────────────
class NERTagger(nn.Module):
    """Слово = эмбеддинг слова (нижний регистр) + CNN по символам (с регистром) → BiLSTM → тег."""

    def __init__(self, n_words, n_chars, n_tags, word_dim=100, char_dim=32, char_filters=64, hidden=192):
        super().__init__()
        self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=0)
        self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.char_cnn = nn.Conv1d(char_dim, char_filters, 3, padding=1)
        self.lstm = nn.LSTM(word_dim + char_filters, hidden, num_layers=2, bidirectional=True,
                            batch_first=True, dropout=0.3)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden * 2, n_tags)

    def forward(self, words, chars):
        B, T, C = chars.shape
        c = self.char_emb(chars.view(B * T, C)).transpose(1, 2)
        c = torch.relu(self.char_cnn(c)).max(dim=2).values.view(B, T, -1)
        out, _ = self.lstm(self.drop(torch.cat([self.word_emb(words), c], dim=-1)))
        return self.fc(self.drop(out))


def encode_batch(sents, wvocab, cvocab):
    T = max(len(s) for s in sents)
    words = torch.zeros(len(sents), T, dtype=torch.long)
    chars = torch.zeros(len(sents), T, MAX_CHARS, dtype=torch.long)
    for i, s in enumerate(sents):
        for j, tok in enumerate(s):
            words[i, j] = wvocab.get(tok.lower(), 1)
            for k, ch in enumerate(tok[:MAX_CHARS]):
                chars[i, j, k] = cvocab.get(ch, 1)
    return words, chars


def spans(tags):
    """Сущности (тип, начало, конец) из BIO; I- без B- начинает новую сущность."""
    out, start, typ = set(), None, None
    for i, t in enumerate(list(tags) + [0]):
        name = TAGS[t]
        if name.startswith("I-") and typ == name[2:]:
            continue
        if typ:
            out.add((typ, start, i))
        start, typ = (i, name[2:]) if name != "O" else (None, None)
    return out


def predict(model, sents, wvocab, cvocab):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(sents), 256):
            chunk = sents[i:i + 256]
            w, c = encode_batch(chunk, wvocab, cvocab)
            out = model(w.to(DEVICE), c.to(DEVICE)).argmax(-1).cpu()
            preds += [out[k, :len(s)].tolist() for k, s in enumerate(chunk)]
    return preds


def f1(model, data, wvocab, cvocab):
    preds = predict(model, [s for s, _ in data], wvocab, cvocab)
    tp = fp = fn = 0
    for (_, gold), pred in zip(data, preds):
        g, p = spans(gold), spans(pred)
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def wikiann(lang, split):
    rows = load_parquet("unimelb-nlp/wikiann", f"{lang}/{split}-00000-of-00001.parquet")
    return rows, [(r["tokens"][:MAX_TOKENS], r["ner_tags"][:MAX_TOKENS]) for r in rows if r["tokens"]]


def main():
    train, val, test = [], [], {}
    for lang in ("ru", "en"):
        rows, tr = wikiann(lang, "train")
        ents = harvest_entities(rows)
        print(f"WikiANN {lang}: {len(tr)} | сущностей PER {len(ents['PER'])}, ORG {len(ents['ORG'])}, LOC {len(ents['LOC'])}")
        train += tr + [from_template(lang, ents) for _ in range(12000)]
        val += wikiann(lang, "validation")[1][:3000] + [from_template(lang, ents) for _ in range(1000)]
        test[lang] = wikiann(lang, "test")[1]
    raw = [parse_annotated(line) for line in RAW.strip().splitlines()]
    train += raw * 20
    random.shuffle(train)

    wcnt = Counter(t.lower() for s, _ in train for t in s)
    wvocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in wcnt.most_common(40000):
        if c < 2:
            break
        wvocab[w] = len(wvocab)
    ccnt = Counter(ch for s, _ in train for t in s for ch in t[:MAX_CHARS])
    cvocab = {"<PAD>": 0, "<UNK>": 1}
    for ch, c in ccnt.most_common():
        if c >= 5:
            cvocab[ch] = len(cvocab)

    cfg = {"word_dim": 100, "char_dim": 32, "char_filters": 64, "hidden": 192}
    model = NERTagger(len(wvocab), len(cvocab), len(TAGS), **cfg).to(DEVICE)
    print(f"Обучение {len(train)} | валидация {len(val)} | слов {len(wvocab)} | символов {len(cvocab)} | "
          f"{sum(p.numel() for p in model.parameters()):,} параметров")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    best, bad = -1.0, 0
    for epoch in range(1, 21):
        t0 = time.time()
        model.train()
        random.shuffle(train)
        for i in range(0, len(train), 64):
            batch = train[i:i + 64]
            w, c = encode_batch([s for s, _ in batch], wvocab, cvocab)
            y = torch.full(w.shape, -100, dtype=torch.long)
            for k, (_, tags) in enumerate(batch):
                y[k, :len(tags)] = torch.tensor(tags)
            # Иногда прячем слово — модель учится узнавать незнакомые имена по буквам
            w = w.masked_fill((torch.rand(w.shape) < 0.05) & (w > 1), 1)
            loss = loss_fn(model(w.to(DEVICE), c.to(DEVICE)).view(-1, len(TAGS)), y.view(-1).to(DEVICE))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        score = f1(model, val, wvocab, cvocab)
        mark = ""
        if score > best:
            best, bad, mark = score, 0, " *"
            torch.save({"model_state": model.state_dict(), "word_vocab": wvocab, "char_vocab": cvocab,
                        "tags": TAGS, "max_chars": MAX_CHARS, "config": cfg}, "ner_model.pth")
        else:
            bad += 1
        print(f"  эпоха {epoch:2d} | val F1 {score:.3f}{mark} | {time.time() - t0:.0f} с")
        if bad >= 3:
            break

    model.load_state_dict(torch.load("ner_model.pth", weights_only=True)["model_state"])
    for lang, data in test.items():
        print(f"WikiANN {lang} test F1: {f1(model, data, wvocab, cvocab):.3f}")
    print(f"Ручные примеры F1: {f1(model, raw, wvocab, cvocab):.3f}")
    for text in ["Александр Пушкин родился в Москве в июне 1799 года.",
                 "Яндекс открыл офис в Санкт-Петербурге в 2021 году.",
                 "Юрий Гагарин полетел в космос 12 апреля 1961 года.",
                 "Мария Иванова работает в Газпроме с 2015 года.",
                 "Elon Musk founded SpaceX in California in 2002.",
                 "Angela Merkel met Emmanuel Macron in Paris on March 3, 2019."]:
        toks = tokenize(text)
        pred = predict(model, [toks], wvocab, cvocab)[0]
        print(f"  {text}\n    → {[(' '.join(toks[s:e]), t) for t, s, e in sorted(spans(pred), key=lambda x: x[1])]}")


if __name__ == "__main__":
    main()
