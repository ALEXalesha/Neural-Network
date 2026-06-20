import torch
import torch.nn as nn
import numpy as np
import wikipedia
import os

# ─────────────────────────────────────────
# 1. ЗАГРУЗКА ДАННЫХ — русская Википедия
# ─────────────────────────────────────────
cache_file = "books_cache.txt"

# Темы на английском — wikipedia пакет работает стабильнее с ними
# но язык ru, поэтому статьи будут на русском
TOPICS = [
    "Russia", "Moscow", "Saint Petersburg", "History of Russia",
    "Physics", "Chemistry", "Biology", "Mathematics", "Astronomy",
    "World War II", "Space", "Solar System", "Universe",
    "Literature", "Alexander Pushkin", "Leo Tolstoy", "Fyodor Dostoevsky",
    "Economics", "Technology", "Computer", "Internet", "Artificial intelligence",
    "Music", "Art", "Painting", "Architecture", "Cinema",
    "Animal", "Ocean", "Climate", "Evolution",
    "Philosophy", "Psychology", "Medicine", "Sport",
    "Football", "Olympic Games", "Theatre",
    "Europe", "Asia", "Africa", "Americas",
    "French Revolution", "Napoleon", "Peter the Great",
    "Quantum mechanics", "Theory of relativity", "DNA", "Genetics",
    "Ecology", "Geology", "Geography", "Sociology",
]

if os.path.exists(cache_file):
    print("Читаем кеш...")
    with open(cache_file, encoding='utf-8') as f:
        text = f.read()
else:
    wikipedia.set_lang("ru")
    text = ""
    ok, fail = 0, 0
    for topic in TOPICS:
        try:
            page = wikipedia.page(topic, auto_suggest=True)
            text += page.content + "\n\n"
            ok += 1
            print(f"  [{ok:2d}] OK: {len(page.content):,} символов")
        except Exception as e:
            fail += 1
            print(f"  SKIP: {topic} ({type(e).__name__})")

    print(f"\nУспешно: {ok} | Ошибок: {fail}")
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(text)

print(f"Итого символов: {len(text):,}")
print(f"Примерно {len(text)//1000} КБ текста\n")

# ─────────────────────────────────────────
# 2. ПОДГОТОВКА ДАННЫХ
# ─────────────────────────────────────────
# Оставляем только символы которые встречаются хотя бы 100 раз
# (убирает мусор — редкие спецсимволы)
from collections import Counter
counts = Counter(text)
chars = sorted(ch for ch, cnt in counts.items() if cnt >= 100)
vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
UNK = 0  # неизвестный символ → 0

print(f"Уникальных символов: {vocab_size}")

# Переводим текст в числа
data = torch.tensor(
    [char_to_idx.get(ch, UNK) for ch in text],
    dtype=torch.long
)

# Сохраняем словарь для генерации
torch.save({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, 'vocab.pth')

# ─────────────────────────────────────────
# 3. АРХИТЕКТУРА — большой LSTM
# ─────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}\n")

seq_len = 150  # длиннее контекст → лучше понимает структуру

def get_batch(batch_size=128):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[s : s + seq_len]     for s in starts])
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512, num_layers=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=0.4)
        self.norm  = nn.LayerNorm(hidden_dim)
        self.fc    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.norm(out)
        return self.fc(out), hidden


model = CharLSTM(vocab_size).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Параметров: {params:,}\n")

# ─────────────────────────────────────────
# 4. ОБУЧЕНИЕ
# ─────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)

epochs     = 500
print_every = 50
best_loss  = float('inf')

for epoch in range(1, epochs + 1):
    model.train()
    x, y = get_batch(128)
    output, _ = model(x)
    loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % print_every == 0:
        scheduler.step(loss)
        lr = optimizer.param_groups[0]['lr']
        print(f"Эпоха {epoch:4d}/{epochs} | Loss: {loss.item():.4f} | LR: {lr:.5f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'text_model.pth')


# ─────────────────────────────────────────
# 5. ГЕНЕРАЦИЯ
# ─────────────────────────────────────────
def generate(seed, length=600, temperature=0.8):
    model.eval()
    ids = torch.tensor(
        [[char_to_idx.get(ch, UNK) for ch in seed]],
        dtype=torch.long
    ).to(device)

    generated = seed
    hidden = None

    with torch.no_grad():
        _, hidden = model(ids, hidden)
        current = ids[:, -1:]
        for _ in range(length):
            out, hidden = model(current, hidden)
            probs = torch.softmax(out[0, -1] / temperature, dim=0)
            next_id = torch.multinomial(probs, 1).item()
            generated += idx_to_char.get(next_id, '?')
            current = torch.tensor([[next_id]], dtype=torch.long).to(device)

    return generated


print("\n" + "="*60)
print("ГЕНЕРАЦИЯ:")
print("="*60)

for seed in ["Россия", "Наука", "История"]:
    print(f"\n--- Тема: '{seed}' ---")
    print(generate(seed, length=500, temperature=0.8))
