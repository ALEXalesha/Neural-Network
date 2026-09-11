"""
Обучение NER-модели (Named Entity Recognition) — LSTM-тегер
Двуязычная: английский + русский
Теги: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-DATE, I-DATE
Сохраняет ner_model.pth
"""
import sys, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import Counter

torch.manual_seed(42)
np.random.seed(42)

BASE   = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {DEVICE}\n")

TAGS  = ["O", "B-PER","I-PER", "B-ORG","I-ORG", "B-LOC","I-LOC", "B-DATE","I-DATE"]
T2I   = {t: i for i, t in enumerate(TAGS)}
NUM_T = len(TAGS)

# ── Обучающие данные ──────────────────────────────────────────────────────────
RAW = [
    # ── English: PER ──
    ("Elon Musk founded SpaceX in California in 2002.",
     ["B-PER","I-PER","O","B-ORG","O","B-LOC","O","B-DATE","O"]),
    ("Barack Obama was born in Hawaii on August 4, 1961.",
     ["B-PER","I-PER","O","O","O","B-LOC","O","B-DATE","I-DATE","I-DATE","O"]),
    ("Marie Curie received the Nobel Prize in 1903.",
     ["B-PER","I-PER","O","O","B-ORG","I-ORG","O","B-DATE","O"]),
    ("Steve Jobs and Steve Wozniak co-founded Apple in Los Angeles.",
     ["B-PER","I-PER","O","B-PER","I-PER","O","B-ORG","O","B-LOC","I-LOC","O"]),
    ("Albert Einstein published his theory in 1905 in Bern.",
     ["B-PER","I-PER","O","O","O","O","B-DATE","O","B-LOC","O"]),
    ("Jeff Bezos started Amazon in Seattle in July 1994.",
     ["B-PER","I-PER","O","B-ORG","O","B-LOC","O","B-DATE","I-DATE","O"]),
    ("Ada Lovelace is considered the first programmer.",
     ["B-PER","I-PER","O","O","O","O","O","O"]),
    ("Leonardo da Vinci lived in Florence during the Renaissance.",
     ["B-PER","I-PER","I-PER","O","O","B-LOC","O","O","O","O"]),
    ("Mark Zuckerberg launched Facebook from Harvard in 2004.",
     ["B-PER","I-PER","O","B-ORG","O","B-ORG","O","B-DATE","O"]),
    ("Isaac Newton was born on January 4, 1643.",
     ["B-PER","I-PER","O","O","O","B-DATE","I-DATE","I-DATE","O"]),
    # ── English: ORG ──
    ("Google was founded in 1998 by Larry Page.",
     ["B-ORG","O","O","O","B-DATE","O","B-PER","I-PER","O"]),
    ("Microsoft released Windows 95 in August 1995.",
     ["B-ORG","O","O","O","O","B-DATE","I-DATE","O"]),
    ("Tesla opened a factory in Berlin in 2021.",
     ["B-ORG","O","O","O","B-LOC","O","B-DATE","O"]),
    ("OpenAI was created in San Francisco in December 2015.",
     ["B-ORG","O","O","O","B-LOC","I-LOC","O","B-DATE","I-DATE","O"]),
    ("NASA launched Apollo 11 on July 16, 1969.",
     ["B-ORG","O","O","O","O","B-DATE","I-DATE","I-DATE","O"]),
    ("The United Nations was founded in 1945 in New York.",
     ["O","B-ORG","I-ORG","O","O","O","B-DATE","O","B-LOC","I-LOC","O"]),
    ("Amazon acquired Whole Foods in 2017 for 13.7 billion dollars.",
     ["B-ORG","O","B-ORG","I-ORG","O","B-DATE","O","O","O","O","O"]),
    # ── English: LOC ──
    ("Paris is the capital of France and is located in Europe.",
     ["B-LOC","O","O","O","O","B-LOC","O","O","O","O","B-LOC","O"]),
    ("The Amazon River flows through Brazil and Peru.",
     ["O","B-LOC","I-LOC","O","O","B-LOC","O","B-LOC","O"]),
    ("Mount Everest is located in Nepal near Tibet.",
     ["B-LOC","I-LOC","O","O","O","B-LOC","O","B-LOC","O"]),
    ("The Great Wall of China was built over many centuries.",
     ["O","B-LOC","I-LOC","I-LOC","I-LOC","O","O","O","O","O","O"]),
    ("Tokyo is the largest city in Japan with over 13 million people.",
     ["B-LOC","O","O","O","O","B-LOC","O","O","O","O","O","O","O"]),
    # ── English: DATE ──
    ("The meeting is scheduled for Monday, March 10, 2025.",
     ["O","O","O","O","O","B-DATE","I-DATE","I-DATE","I-DATE","O"]),
    ("World War II ended on September 2, 1945.",
     ["B-LOC","I-LOC","I-LOC","O","O","B-DATE","I-DATE","I-DATE","O"]),
    ("He was born on April 15 and died in November 2001.",
     ["O","O","O","O","B-DATE","I-DATE","O","O","O","B-DATE","I-DATE","O"]),
    # ── English: Mixed ──
    ("In 2023, OpenAI released GPT-4 in San Francisco.",
     ["O","B-DATE","O","B-ORG","O","O","O","B-LOC","I-LOC","O"]),
    ("Bill Gates and Paul Allen founded Microsoft in Albuquerque in 1975.",
     ["B-PER","I-PER","O","B-PER","I-PER","O","B-ORG","O","B-LOC","O","B-DATE","O"]),
    ("Apple announced the iPhone on January 9, 2007 in San Francisco.",
     ["B-ORG","O","O","B-ORG","O","B-DATE","I-DATE","I-DATE","O","B-LOC","I-LOC","O"]),
    ("The Berlin Wall fell on November 9, 1989.",
     ["O","B-LOC","I-LOC","O","O","B-DATE","I-DATE","I-DATE","O"]),
    ("SpaceX launched its first Falcon rocket from Cape Canaveral in March 2006.",
     ["B-ORG","O","O","O","O","O","O","B-LOC","I-LOC","O","B-DATE","I-DATE","O"]),
    ("Queen Elizabeth II died on September 8, 2022 at Balmoral Castle.",
     ["B-PER","I-PER","I-PER","O","O","B-DATE","I-DATE","I-DATE","O","B-LOC","I-LOC","O"]),
    ("Nikola Tesla was born in Serbia on July 10, 1856.",
     ["B-PER","I-PER","O","O","O","B-LOC","O","B-DATE","I-DATE","I-DATE","O"]),

    # ── Russian: PER ──
    ("Александр Пушкин родился в Москве 6 июня 1799 года.",
     ["B-PER","I-PER","O","O","B-LOC","B-DATE","I-DATE","I-DATE","I-DATE","O"]),
    ("Лев Толстой жил в Ясной Поляне в Тульской области.",
     ["B-PER","I-PER","O","O","B-LOC","I-LOC","O","B-LOC","I-LOC","O"]),
    ("Юрий Гагарин совершил первый полёт в космос 12 апреля 1961 года.",
     ["B-PER","I-PER","O","O","O","O","O","B-DATE","I-DATE","I-DATE","I-DATE","O"]),
    ("Михаил Ломоносов основал Московский университет в 1755 году.",
     ["B-PER","I-PER","O","B-ORG","I-ORG","O","B-DATE","I-DATE","O"]),
    ("Фёдор Достоевский написал роман в Санкт-Петербурге в 1866 году.",
     ["B-PER","I-PER","O","O","O","B-LOC","O","B-DATE","I-DATE","O"]),
    ("Пётр Чайковский родился в Воткинске в мае 1840 года.",
     ["B-PER","I-PER","O","O","B-LOC","O","B-DATE","I-DATE","I-DATE","O"]),
    ("Дмитрий Менделеев создал периодическую таблицу элементов в 1869 году.",
     ["B-PER","I-PER","O","O","O","O","O","B-DATE","I-DATE","O"]),
    ("Антон Чехов жил в Ялте в последние годы своей жизни.",
     ["B-PER","I-PER","O","O","B-LOC","O","O","O","O","O","O","O"]),
    ("Сергей Брин и Ларри Пейдж основали Google в 1998 году.",
     ["B-PER","I-PER","O","B-PER","I-PER","O","B-ORG","O","B-DATE","I-DATE","O"]),
    ("Анна Ахматова написала стихи в блокадном Ленинграде в 1941 году.",
     ["B-PER","I-PER","O","O","O","B-LOC","O","B-DATE","I-DATE","O"]),
    # ── Russian: ORG ──
    ("Яндекс был основан в Москве в 1997 году.",
     ["B-ORG","O","O","O","B-LOC","O","B-DATE","I-DATE","O"]),
    ("Сбербанк открыл новый офис в Москве в марте 2023 года.",
     ["B-ORG","O","O","O","O","B-LOC","O","B-DATE","I-DATE","I-DATE","O"]),
    ("Газпром добывает природный газ в Сибири и экспортирует в Европу.",
     ["B-ORG","O","O","O","O","B-LOC","O","O","O","B-LOC","O"]),
    ("МГУ расположен в Москве на Воробьёвых горах.",
     ["B-ORG","O","O","B-LOC","O","B-LOC","I-LOC","O"]),
    ("Роснефть подписала договор с китайскими компаниями в декабре 2022 года.",
     ["B-ORG","O","O","O","O","O","O","B-DATE","I-DATE","I-DATE","O"]),
    ("ВТБ и Сбербанк являются крупнейшими банками России.",
     ["B-ORG","O","B-ORG","O","O","O","O","B-LOC","O"]),
    ("Аэрофлот выполняет рейсы из Москвы в Токио и Пекин.",
     ["B-ORG","O","O","O","B-LOC","O","B-LOC","O","B-LOC","O"]),
    ("РЖД запустила новый маршрут из Петербурга в Хельсинки в январе 2024.",
     ["B-ORG","O","O","O","O","B-LOC","O","B-LOC","O","B-DATE","I-DATE","O"]),
    # ── Russian: LOC ──
    ("Москва является столицей России и расположена на реке Москве.",
     ["B-LOC","O","O","B-LOC","O","O","O","O","O","B-LOC","O"]),
    ("Байкал находится в Сибири и является самым глубоким озером в мире.",
     ["B-LOC","O","O","B-LOC","O","O","O","O","O","O","O","O","O"]),
    ("Волга течёт через Казань и впадает в Каспийское море.",
     ["B-LOC","O","O","B-LOC","O","O","O","B-LOC","I-LOC","O"]),
    ("Санкт-Петербург был основан Петром Первым в 1703 году.",
     ["B-LOC","O","O","B-PER","I-PER","O","B-DATE","I-DATE","O"]),
    ("Уральские горы разделяют Европу и Азию.",
     ["B-LOC","I-LOC","O","B-LOC","O","B-LOC","O"]),
    ("Владивосток находится на Дальнем Востоке у берегов Японского моря.",
     ["B-LOC","O","O","B-LOC","I-LOC","O","O","B-LOC","I-LOC","O"]),
    # ── Russian: DATE ──
    ("Великая Отечественная война закончилась 9 мая 1945 года.",
     ["B-LOC","I-LOC","I-LOC","O","B-DATE","I-DATE","I-DATE","I-DATE","O"]),
    ("Революция произошла в октябре 1917 года в России.",
     ["O","O","O","B-DATE","I-DATE","I-DATE","O","B-LOC","O"]),
    ("Олимпийские игры в Сочи прошли в феврале 2014 года.",
     ["B-ORG","I-ORG","O","B-LOC","O","O","B-DATE","I-DATE","I-DATE","O"]),
    ("Чемпионат мира по футболу в России состоялся летом 2018 года.",
     ["B-ORG","I-ORG","O","O","O","B-LOC","O","O","B-DATE","I-DATE","I-DATE","O"]),
    # ── Russian: Mixed ──
    ("Владимир Путин посетил Санкт-Петербург в январе 2024 года.",
     ["B-PER","I-PER","O","B-LOC","O","B-DATE","I-DATE","I-DATE","O"]),
    ("Google открыла офис в Москве в 2006 году.",
     ["B-ORG","O","O","O","B-LOC","O","B-DATE","I-DATE","O"]),
    ("Apple представила iPhone X в Купертино в сентябре 2017 года.",
     ["B-ORG","O","B-ORG","I-ORG","O","B-LOC","O","B-DATE","I-DATE","I-DATE","O"]),
    ("ООН провела встречу в Женеве в декабре 2023 года.",
     ["B-ORG","O","O","O","B-LOC","O","B-DATE","I-DATE","I-DATE","O"]),
    ("Илон Маск посетил Москву в марте 2022 года.",
     ["B-PER","I-PER","O","B-LOC","O","B-DATE","I-DATE","I-DATE","O"]),
]

# ── Словарь ────────────────────────────────────────────────────────────────────
all_words = [w for sent, _ in RAW for w in sent.split()]
cnt = Counter(w.lower() for w in all_words)
VOCAB = {"<PAD>": 0, "<UNK>": 1}
for w, _ in cnt.most_common():
    VOCAB[w.lower()] = len(VOCAB)
for w in all_words:
    if w not in VOCAB:
        VOCAB[w] = len(VOCAB)

print(f"Примеров: {len(RAW)} (EN+RU) | Словарь: {len(VOCAB)} | Теги: {NUM_T}")

def encode_sent(tokens):
    return [VOCAB.get(t.lower(), 1) for t in tokens]

def encode_tags(tags):
    return [T2I.get(t, 0) for t in tags]

MAX_LEN = 20

def pad(lst, val, length):
    return lst[:length] + [val] * max(0, length - len(lst))

X_data, Y_data = [], []
for sent, tags in RAW:
    tokens = sent.split()
    tokens = tokens[:len(tags)]
    x = pad(encode_sent(tokens), 0, MAX_LEN)
    y = pad(encode_tags(tags), 0, MAX_LEN)
    X_data.append(x); Y_data.append(y)

# Augmentation
import random
random.seed(42)
aug_X, aug_Y = [], []
for x, y in zip(X_data, Y_data):
    for _ in range(12):
        ax = x[:]
        for i in range(len(ax)):
            if ax[i] != 0 and random.random() < 0.08:
                ax[i] = random.randint(1, len(VOCAB)-1)
        aug_X.append(ax); aug_Y.append(y)

X_data += aug_X; Y_data += aug_Y

X_t = torch.tensor(X_data, dtype=torch.long)
Y_t = torch.tensor(Y_data, dtype=torch.long)

split = int(0.85 * len(X_t))
X_train, Y_train = X_t[:split].to(DEVICE), Y_t[:split].to(DEVICE)
X_test,  Y_test  = X_t[split:].to(DEVICE), Y_t[split:].to(DEVICE)

# ── Модель: Bi-LSTM тегер ─────────────────────────────────────────────────────
class NERModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_tags=NUM_T):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                             bidirectional=True, num_layers=2, dropout=0.3)
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x):
        e   = self.embed(x)
        out, _ = self.lstm(e)
        return self.fc(self.drop(out))

VOCAB_SIZE = len(VOCAB)
model     = NERModel(VOCAB_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ── Обучение ──────────────────────────────────────────────────────────────────
EPOCHS = 150
print(f"Обучение: {EPOCHS} эпох...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = torch.randperm(split)
    for i in range(0, split, 32):
        xb = X_train[idx[i:i+32]]
        yb = Y_train[idx[i:i+32]]
        logits = model(xb)
        loss   = criterion(logits.view(-1, NUM_T), yb.view(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    if epoch % 30 == 0:
        model.eval()
        with torch.no_grad():
            logits_t = model(X_test)
            preds = logits_t.argmax(-1)
            mask  = Y_test != 0
            acc   = (preds[mask] == Y_test[mask]).float().mean().item()
        print(f"  Эпоха {epoch}/{EPOCHS} | Acc (non-O): {acc*100:.1f}%")

# ── Тест ──────────────────────────────────────────────────────────────────────
def predict_ner(text):
    tokens = text.split()[:MAX_LEN]
    ids    = pad(encode_sent(tokens), 0, MAX_LEN)
    inp    = torch.tensor([ids]).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(inp)[0][:len(tokens)]
        preds  = logits.argmax(-1).cpu().numpy()
    return [(t, TAGS[p]) for t, p in zip(tokens, preds)]

print("\nПримеры EN:")
for text in [
    "Elon Musk founded SpaceX in California in 2002.",
    "Marie Curie received the Nobel Prize in 1903.",
]:
    res = predict_ner(text)
    ents = [f"{t}({tag})" for t, tag in res if tag != "O"]
    print(f"  '{text[:55]}' → {ents}")

print("\nПримеры RU:")
for text in [
    "Александр Пушкин родился в Москве в июне 1799 года.",
    "Яндекс открыл офис в Санкт-Петербурге в 2021 году.",
    "Юрий Гагарин полетел в космос 12 апреля 1961 года.",
]:
    res = predict_ner(text)
    ents = [f"{t}({tag})" for t, tag in res if tag != "O"]
    print(f"  '{text[:55]}' → {ents}")

# ── Сохранение ────────────────────────────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "vocab":       VOCAB,
    "vocab_size":  VOCAB_SIZE,
    "tags":        TAGS,
    "max_len":     MAX_LEN,
    "embed_dim":   64,
    "hidden_dim":  128,
    "languages":   ["en", "ru"],
}, BASE / "ner_model.pth")

print("\nМодель сохранена: ner_model.pth")
