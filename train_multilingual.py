"""
Дообучение GPT на английской Википедии (v2 — 80+ статей, языковые теги)
Модель учится различать EN и RU через теги [EN] / [RU]
Чтобы сгенерировать на нужном языке — начни seed с [EN] или [RU]
"""
import sys, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

BASE   = Path(__file__).parent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}\n")

# ──────────────────────────────────────────────────────
# 1. СКАЧАТЬ АНГЛИЙСКУЮ ВИКИ (80+ статей)
# ──────────────────────────────────────────────────────
def download_en_wiki(target_mb=15):
    cache = BASE / 'en_wiki_text_v2.txt'
    if cache.exists() and cache.stat().st_size > 1_000_000:
        print(f"Английский текст уже есть: {cache.stat().st_size // 1024} КБ")
        return cache.read_text(encoding='utf-8')

    print(f"Скачиваю английскую Википедию (~{target_mb} МБ, 80+ статей)...")
    try:
        import wikipedia
        wikipedia.set_lang('en')
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'wikipedia', '-q'])
        import wikipedia
        wikipedia.set_lang('en')

    topics = [
        # Science & Tech
        'Artificial intelligence', 'Neural network', 'Machine learning',
        'Deep learning', 'Python programming language', 'Computer science',
        'Algorithm', 'Data structure', 'Operating system', 'Database',
        'Quantum computing', 'Robotics', 'Cybersecurity', 'Blockchain',
        'Internet of things', 'Cloud computing', 'Software engineering',
        # Natural Sciences
        'Physics', 'Chemistry', 'Biology', 'Mathematics', 'Science',
        'Quantum mechanics', 'Theory of relativity', 'Thermodynamics',
        'Electromagnetism', 'Genetics', 'Cell biology', 'Ecology',
        'Organic chemistry', 'Astronomy', 'Cosmology', 'Geology',
        'Meteorology', 'Neuroscience', 'Biochemistry',
        # Space
        'Universe', 'Solar System', 'Space exploration', 'NASA',
        'Black hole', 'Galaxy', 'Star', 'Planet', 'Mars', 'Moon',
        # History & Society
        'Industrial Revolution', 'Renaissance', 'World War II',
        'Ancient Rome', 'Ancient Greece', 'Middle Ages',
        'Cold War', 'French Revolution', 'History of China',
        'Democracy', 'Capitalism', 'Communism', 'Globalization',
        # Arts & Culture
        'Literature', 'Philosophy', 'Music theory', 'Architecture',
        'Painting', 'Cinema', 'Theatre', 'Photography',
        'Classical music', 'Jazz', 'Opera',
        # Life & Society
        'Psychology', 'Sociology', 'Economics', 'Education',
        'Medicine', 'Public health', 'Nutrition', 'Sport',
        'Religion', 'Ethics', 'Linguistics',
        # Nature
        'Evolution', 'DNA', 'Climate change', 'Geography',
        'Ocean', 'Forest', 'Biodiversity', 'Animal',
        'Human', 'Brain', 'Heart', 'Language',
    ]

    texts = []
    total = 0
    target = target_mb * 1024 * 1024

    for topic in topics:
        if total >= target:
            break
        try:
            page = wikipedia.page(topic, auto_suggest=False)
            text = page.content
            # Добавляем языковой тег [EN] в начало каждой статьи
            texts.append('[EN] ' + text)
            total += len(text.encode('utf-8'))
            print(f"  [{total//1024} КБ] {topic} ({len(text)} символов)")
        except Exception as e:
            print(f"  Пропуск: {topic} — {e}")

    full_text = '\n\n'.join(texts)
    cache.write_text(full_text, encoding='utf-8')
    print(f"\nИтого: {len(full_text):,} символов\n")
    return full_text

# ──────────────────────────────────────────────────────
# 2. ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ МОДЕЛИ
# ──────────────────────────────────────────────────────

# Копируем архитектуру из gpt_bpe.py
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
    def forward(self, x, mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / self.head_dim**0.5
        if mask is not None: att = att.masked_fill(mask==0, -1e9)
        att = torch.softmax(att, -1)
        return self.proj((att @ v).transpose(1,2).contiguous().view(B, T, C))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn  = SelfAttention(embed_dim, num_heads)
        self.ff    = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.ff(self.norm2(x))

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8,
                 num_layers=6, ff_dim=1024, context_len=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb   = nn.Embedding(context_len, embed_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.norm      = nn.LayerNorm(embed_dim)
        self.head      = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.ctx       = context_len
    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        emb  = self.drop(self.token_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            emb = block(emb, mask)
        return self.head(self.norm(emb))

# ──────────────────────────────────────────────────────
# 3. ЗАГРУЗКА ТОКЕНИЗАТОРА И МОДЕЛИ
# ──────────────────────────────────────────────────────

tok_path  = BASE / 'bpe_tokenizer.json'
model_path = BASE / 'gpt_bpe_model.pth'

if not tok_path.exists():
    print("Токенизатор не найден — запустите gpt_bpe.py сначала")
    sys.exit(1)

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(str(tok_path))
VOCAB = tokenizer.get_vocab_size()
CTX   = 128

print(f"Токенизатор: {VOCAB} токенов")

model = MiniGPT(VOCAB, embed_dim=256, num_heads=8, num_layers=6, ff_dim=1024, context_len=128).to(device)
if model_path.exists():
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    print("Загружена предыдущая модель (RU Вики)")
else:
    print("Начинаем с нуля")

params = sum(p.numel() for p in model.parameters())
print(f"Параметров: {params:,}\n")

# ──────────────────────────────────────────────────────
# 4. ПОДГОТОВКА ДАННЫХ (EN + RU с языковыми тегами)
# ──────────────────────────────────────────────────────
en_text = download_en_wiki(target_mb=15)
print(f"Токенизирую английский текст...")
enc_en = tokenizer.encode(en_text)
ids_en = np.array(enc_en.ids, dtype=np.int32)
print(f"EN токенов: {len(ids_en):,}")

# Подгружаем русский текст (добавляем тег [RU] чтобы модель различала языки)
ru_cache = BASE / 'ru_wiki_text.txt'
ids_ru = None
if ru_cache.exists() and ru_cache.stat().st_size > 100_000:
    print(f"Загружаю русский текст для смешивания...")
    ru_raw = ru_cache.read_text(encoding='utf-8')
    # Добавляем тег [RU] к русскому тексту
    ru_tagged = '[RU] ' + ru_raw.replace('\n\n', '\n\n[RU] ')
    enc_ru = tokenizer.encode(ru_tagged)
    ids_ru = np.array(enc_ru.ids, dtype=np.int32)
    print(f"RU токенов: {len(ids_ru):,}")
else:
    print("Русский текст не найден — обучаем только на EN")

print()

# ──────────────────────────────────────────────────────
# 5. ДООБУЧЕНИЕ (EN + RU mix, 5000 шагов)
# ──────────────────────────────────────────────────────
def get_batch(ids, batch_size=32, ctx=CTX):
    ix = torch.randint(len(ids) - ctx, (batch_size,))
    x  = torch.stack([torch.tensor(ids[i:i+ctx],   dtype=torch.long) for i in ix])
    y  = torch.stack([torch.tensor(ids[i+1:i+ctx+1], dtype=torch.long) for i in ix])
    return x.to(device), y.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
criterion = nn.CrossEntropyLoss()

STEPS = 5000
print(f"Дообучение EN+RU ({STEPS} шагов)...\n")
best_loss = float('inf')

for step in range(1, STEPS + 1):
    model.train()
    # 70% EN батчи, 30% RU батчи (чтобы не забыть русский)
    if ids_ru is not None and np.random.rand() < 0.30:
        x, y = get_batch(ids_ru)
    else:
        x, y = get_batch(ids_en)

    logits = model(x)
    loss   = criterion(logits.view(-1, VOCAB), y.view(-1))
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); scheduler.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), BASE / 'gpt_multilingual.pth')

    if step % 500 == 0:
        print(f"  Шаг {step:4d}/{STEPS} | Loss: {loss.item():.4f}")

print("\nМодель сохранена: gpt_multilingual.pth\n")

# ──────────────────────────────────────────────────────
# 6. ГЕНЕРАЦИЯ — проверяем оба языка
# ──────────────────────────────────────────────────────
model.load_state_dict(torch.load(BASE / 'gpt_multilingual.pth', weights_only=True))
model.eval()

def generate(seed, length=200, temperature=0.8):
    enc_seed = tokenizer.encode(seed)
    ids_gen  = enc_seed.ids[-CTX:]
    generated = seed
    with torch.no_grad():
        for _ in range(length):
            x    = torch.tensor([ids_gen[-CTX:]], dtype=torch.long).to(device)
            logits = model(x)[0, -1] / temperature
            probs  = torch.softmax(logits, dim=0)
            next_id = torch.multinomial(probs, 1).item()
            ids_gen.append(next_id)
            generated += tokenizer.decode([next_id])
    return generated

print("=" * 55)
print("Генерация текста (двуязычная модель):")
print("  [EN] prefix → английский текст")
print("  [RU] prefix → русский текст\n")
for seed, lang in [
    ("[EN] Artificial intelligence", "EN"),
    ("[RU] Нейронные сети", "RU"),
    ("[EN] The history of", "EN"),
    ("[RU] Москва — столица", "RU"),
    ("[EN] Physics is the science", "EN"),
    ("[RU] Наука изучает", "RU"),
]:
    print(f"Seed: '{seed}'")
    print(generate(seed, 150))
    print()

# Интерактив
print("=" * 55)
print("Введите начало текста:")
print("  Начни с [EN] для английского или [RU] для русского")
print("  Пустая строка — выход\n")
while True:
    try:
        seed = input("Seed: ").strip()
        if not seed: break
        print(generate(seed, 200))
        print()
    except (EOFError, KeyboardInterrupt):
        break
print("Готово!")
