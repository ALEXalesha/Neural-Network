"""
Нейронный переводчик RU <-> EN
Архитектура: Transformer Encoder-Decoder с BPE токенизацией
Данные: параллельный корпус (скачивается автоматически)
"""
import sys, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import re
import os
import bz2
import urllib.request
from pathlib import Path
from collections import Counter

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}\n")

BASE = Path(__file__).parent

# ──────────────────────────────────────────────────────
# 1. ЗАГРУЗКА ДАННЫХ
# ──────────────────────────────────────────────────────

def download_tatoeba():
    """Скачивает параллельный корпус Tatoeba EN-RU"""
    links_file = BASE / 'tatoeba_links.tsv'
    eng_file   = BASE / 'tatoeba_eng.tsv'
    rus_file   = BASE / 'tatoeba_rus.tsv'

    if not (BASE / 'translator_data.txt').exists():
        print("Скачиваю корпус Tatoeba (EN+RU)...")

        def dl(url, path):
            if not Path(str(path) + '.bz2').exists() and not path.exists():
                print(f"  {url.split('/')[-1]}...")
                urllib.request.urlretrieve(url, str(path) + '.bz2')
            if not path.exists():
                print(f"  Распаковываю {path.name}...")
                with bz2.open(str(path) + '.bz2', 'rt', encoding='utf-8') as f:
                    data = f.read()
                path.write_text(data, encoding='utf-8')

        try:
            dl('https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2', eng_file)
            dl('https://downloads.tatoeba.org/exports/per_language/rus/rus_sentences.tsv.bz2', rus_file)
            dl('https://downloads.tatoeba.org/exports/links.tar.bz2', links_file)
        except Exception as e:
            print(f"  Ошибка скачивания: {e}")
            print("  Использую встроенные данные...")
            return None

        # Парсим
        print("  Строю индекс пар...")
        eng = {}
        with open(eng_file, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    eng[parts[0]] = parts[2]
        rus = {}
        with open(rus_file, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    rus[parts[0]] = parts[2]

        pairs = []
        with open(links_file, encoding='utf-8') as f:
            for line in f:
                a, b = line.strip().split('\t')
                if a in eng and b in rus:
                    pairs.append((eng[a], rus[b]))
                elif b in eng and a in rus:
                    pairs.append((eng[b], rus[a]))
                if len(pairs) >= 100000:
                    break

        print(f"  Найдено {len(pairs)} пар EN-RU")
        with open(BASE / 'translator_data.txt', 'w', encoding='utf-8') as f:
            for en, ru in pairs:
                f.write(f"{en}\t{ru}\n")
        return pairs

    else:
        pairs = []
        with open(BASE / 'translator_data.txt', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        return pairs

# Встроенные данные (если нет интернета)
BUILTIN_PAIRS = [
    ("Hello", "Привет"), ("Good morning", "Доброе утро"),
    ("How are you?", "Как дела?"), ("Thank you", "Спасибо"),
    ("Please", "Пожалуйста"), ("Yes", "Да"), ("No", "Нет"),
    ("I love you", "Я тебя люблю"), ("Goodbye", "До свидания"),
    ("What is your name?", "Как тебя зовут?"),
    ("My name is", "Меня зовут"), ("I am happy", "Я счастлив"),
    ("The weather is nice today", "Сегодня хорошая погода"),
    ("I want to eat", "Я хочу есть"), ("Where are you from?", "Откуда ты?"),
    ("I am from Russia", "Я из России"), ("Do you speak English?", "Ты говоришь по-английски?"),
    ("I don't understand", "Я не понимаю"), ("Can you help me?", "Ты можешь мне помочь?"),
    ("This is beautiful", "Это красиво"), ("I like cats", "Мне нравятся кошки"),
    ("The book is on the table", "Книга на столе"),
    ("I go to school", "Я иду в школу"), ("She is my friend", "Она моя подруга"),
    ("He works every day", "Он работает каждый день"),
    ("We live in Moscow", "Мы живём в Москве"),
    ("The sun is shining", "Солнце светит"),
    ("I read books", "Я читаю книги"), ("Music is my passion", "Музыка — моя страсть"),
    ("Time flies", "Время летит"), ("Life is beautiful", "Жизнь прекрасна"),
    ("Knowledge is power", "Знание — сила"),
    ("The cat sat on the mat", "Кот сидел на коврике"),
    ("I want to learn Russian", "Я хочу выучить русский язык"),
    ("Moscow is the capital of Russia", "Москва — столица России"),
    ("I drink coffee every morning", "Я пью кофе каждое утро"),
    ("The train arrives at noon", "Поезд прибывает в полдень"),
    ("She sings very well", "Она поёт очень хорошо"),
    ("Children play in the park", "Дети играют в парке"),
    ("The sky is blue", "Небо голубое"), ("Snow is white", "Снег белый"),
    ("I need your help", "Мне нужна твоя помощь"),
    ("Let us go", "Пойдём"), ("Good night", "Спокойной ночи"),
    ("See you tomorrow", "До завтра"), ("What time is it?", "Который час?"),
    ("It is three o'clock", "Сейчас три часа"),
    ("I work at a company", "Я работаю в компании"),
    ("She studies at university", "Она учится в университете"),
    ("We are friends", "Мы друзья"), ("This is my house", "Это мой дом"),
    ("Open the door", "Открой дверь"), ("Close the window", "Закрой окно"),
    ("The food is delicious", "Еда восхитительная"),
    ("I am tired", "Я устал"), ("He is sick", "Он болен"),
    ("Spring is my favorite season", "Весна — моё любимое время года"),
    ("I will call you later", "Я позвоню тебе позже"),
    ("Where is the nearest cafe?", "Где ближайшее кафе?"),
    ("How much does it cost?", "Сколько это стоит?"),
    ("I want to travel the world", "Я хочу путешествовать по миру"),
    ("The meeting starts at nine", "Встреча начинается в девять"),
    ("Please be quiet", "Пожалуйста, помолчи"),
    ("I forgot my keys", "Я забыл свои ключи"),
    ("The river is wide", "Река широкая"),
    ("Mountains are high", "Горы высокие"),
    ("I enjoy reading", "Мне нравится читать"),
    ("Technology changes the world", "Технологии меняют мир"),
    ("Artificial intelligence is fascinating", "Искусственный интеллект — это увлекательно"),
    ("Neural networks learn from data", "Нейронные сети учатся на данных"),
    ("The computer is powerful", "Компьютер мощный"),
    ("I play chess", "Я играю в шахматы"),
    ("She draws beautiful pictures", "Она рисует красивые картины"),
    ("We celebrate holidays together", "Мы вместе отмечаем праздники"),
    ("Happy birthday", "С днём рождения"),
    ("Congratulations", "Поздравляю"),
    ("I am sorry", "Мне жаль"), ("Excuse me", "Извините"),
    ("The exam was difficult", "Экзамен был сложным"),
    ("He passed the test", "Он сдал тест"),
    ("She has a beautiful voice", "У неё красивый голос"),
    ("The dog runs fast", "Собака бежит быстро"),
    ("Cats are independent animals", "Кошки — независимые животные"),
    ("I write letters", "Я пишу письма"),
    ("The city never sleeps", "Город никогда не спит"),
    ("Summer is hot", "Лето жаркое"), ("Winter is cold", "Зима холодная"),
    ("Autumn leaves are falling", "Осенние листья падают"),
    ("The flowers bloom in spring", "Цветы расцветают весной"),
]

# Аугментация: перефразируем предложения
def augment_pairs(pairs, n=5):
    import random
    result = list(pairs)
    for en, ru in pairs:
        words_en = en.split()
        words_ru = ru.split()
        if len(words_en) > 2:
            for _ in range(n):
                result.append((en.lower(), ru))
                result.append((en.upper(), ru))
    return result

# ──────────────────────────────────────────────────────
# 2. ТОКЕНИЗАЦИЯ
# ──────────────────────────────────────────────────────
PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

def build_vocab(texts, max_size=16000):
    counter = Counter()
    for t in texts:
        counter.update(t.lower().split())
    vocab = SPECIAL + [w for w, _ in counter.most_common(max_size - len(SPECIAL))]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    return w2i, i2w

def encode_seq(text, w2i, max_len=30):
    tokens = [SOS] + [w2i.get(t, UNK) for t in text.lower().split()][:max_len-2] + [EOS]
    return tokens + [PAD] * (max_len - len(tokens))

# ──────────────────────────────────────────────────────
# 3. TRANSFORMER
# ──────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TranslatorTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=8,
                 num_enc=3, num_dec=3, dim_ff=512, dropout=0.1, max_len=30):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        self.pos_enc   = PositionalEncoding(d_model, max_len)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_enc, num_decoder_layers=num_dec,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True,
        )
        self.fc = nn.Linear(d_model, tgt_vocab)
        self.d_model = d_model

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None):
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.fc(out)

    def translate(self, src_ids, tgt_w2i, tgt_i2w, max_len=30):
        self.eval()
        src = torch.tensor([src_ids], dtype=torch.long).to(device)
        memory = self.transformer.encoder(
            self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        )
        out_ids = [SOS]
        for _ in range(max_len):
            tgt = torch.tensor([out_ids], dtype=torch.long).to(device)
            tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
            sz = tgt.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz).to(device)
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            next_id = self.fc(out[:, -1]).argmax(-1).item()
            if next_id == EOS:
                break
            out_ids.append(next_id)
        return ' '.join(tgt_i2w.get(i, '<UNK>') for i in out_ids[1:])

# ──────────────────────────────────────────────────────
# 4. ПОДГОТОВКА ДАННЫХ
# ──────────────────────────────────────────────────────
print("Загрузка данных...")
pairs = download_tatoeba()
if not pairs or len(pairs) < 100:
    print("Используем встроенные данные + аугментация")
    pairs = BUILTIN_PAIRS

pairs = augment_pairs(pairs)
np.random.shuffle(pairs)
pairs = pairs[:50000]  # ограничиваем для скорости

print(f"Пар для обучения: {len(pairs)}")

en_texts = [p[0] for p in pairs]
ru_texts = [p[1] for p in pairs]

en_w2i, en_i2w = build_vocab(en_texts)
ru_w2i, ru_i2w = build_vocab(ru_texts)

MAX_LEN = 30
print(f"Словарь EN: {len(en_w2i)}, RU: {len(ru_w2i)}\n")

# Кодируем
X = np.array([encode_seq(t, en_w2i, MAX_LEN) for t in en_texts])
Y = np.array([encode_seq(t, ru_w2i, MAX_LEN) for t in ru_texts])

split = int(0.9 * len(X))
X_train = torch.tensor(X[:split], dtype=torch.long)
Y_train = torch.tensor(Y[:split], dtype=torch.long)
X_test  = torch.tensor(X[split:], dtype=torch.long)
Y_test  = torch.tensor(Y[split:], dtype=torch.long)

# ──────────────────────────────────────────────────────
# 5. ОБУЧЕНИЕ
# ──────────────────────────────────────────────────────
model = TranslatorTransformer(
    src_vocab=len(en_w2i), tgt_vocab=len(ru_w2i),
    d_model=256, nhead=8, num_enc=3, num_dec=3
).to(device)

params = sum(p.numel() for p in model.parameters())
print(f"Параметров: {params:,}")

criterion = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

EPOCHS = 100
BATCH  = 128

print("\nОбучение...\n")
best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = torch.randperm(len(X_train))
    epoch_loss = 0

    for i in range(0, len(X_train), BATCH):
        batch_idx = idx[i:i+BATCH]
        src = X_train[batch_idx].to(device)
        tgt = Y_train[batch_idx].to(device)

        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        sz = tgt_in.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz).to(device)
        src_pad  = (src == PAD)
        tgt_pad  = (tgt_in == PAD)

        out  = model(src, tgt_in, src_pad, tgt_mask, tgt_pad)
        loss = criterion(out.reshape(-1, len(ru_w2i)), tgt_out.reshape(-1))

        optimizer.zero_grad(); loss.backward();
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg = epoch_loss / (len(X_train) // BATCH + 1)
    scheduler.step(avg)

    if avg < best_loss:
        best_loss = avg
        torch.save({
            'model': model.state_dict(),
            'en_w2i': en_w2i, 'en_i2w': en_i2w,
            'ru_w2i': ru_w2i, 'ru_i2w': ru_i2w,
        }, BASE / 'translator_model.pth')

    if epoch % 10 == 0:
        print(f"Эпоха {epoch:3d}/{EPOCHS} | Loss: {avg:.4f}")

print("\nМодель сохранена!\n")

# ──────────────────────────────────────────────────────
# 6. ТЕСТ + ИНТЕРАКТИВ
# ──────────────────────────────────────────────────────
ckpt = torch.load(BASE / 'translator_model.pth', weights_only=False)
model.load_state_dict(ckpt['model'])
en_w2i = ckpt['en_w2i']; en_i2w = ckpt['en_i2w']
ru_w2i = ckpt['ru_w2i']; ru_i2w = ckpt['ru_i2w']

def translate_en_ru(text):
    src_ids = encode_seq(text, en_w2i, MAX_LEN)
    with torch.no_grad():
        return model.translate(src_ids, ru_w2i, ru_i2w, MAX_LEN)

print("=" * 50)
print("Тест на примерах:")
tests = [
    "Hello how are you",
    "I love Moscow",
    "The sun is shining today",
    "Neural networks are powerful",
    "I want to learn Russian",
    "Good morning my friend",
]
for t in tests:
    print(f"  EN: {t}")
    print(f"  RU: {translate_en_ru(t)}\n")

print("=" * 50)
print("Введите текст на английском для перевода:")
print("(Enter = выход)\n")
while True:
    try:
        text = input("EN: ").strip()
        if not text: break
        print(f"RU: {translate_en_ru(text)}\n")
    except (EOFError, KeyboardInterrupt):
        break
print("Готово!")
