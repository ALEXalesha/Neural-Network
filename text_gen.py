import torch
import torch.nn as nn
import numpy as np

# ─────────────────────────────────────────
# 1. ТЕКСТ ДЛЯ ОБУЧЕНИЯ
# ─────────────────────────────────────────
# Чем больше текст — тем лучше генерация
# Здесь короткий пример доклада про космос

text = """
Космос — это бесконечное пространство за пределами атмосферы Земли.
Изучение космоса началось в двадцатом веке с запуска первых спутников.
В 1957 году Советский Союз запустил первый искусственный спутник Земли — Спутник-1.
В 1961 году Юрий Гагарин стал первым человеком, побывавшим в космосе.
Космос состоит из планет, звёзд, галактик и межзвёздного пространства.
Наша галактика называется Млечный Путь и содержит миллиарды звёзд.
Солнечная система включает восемь планет: Меркурий, Венеру, Землю, Марс,
Юпитер, Сатурн, Уран и Нептун. Самая большая планета — Юпитер.
Учёные изучают космос с помощью телескопов и космических аппаратов.
Телескоп Хаббл позволил сделать снимки далёких галактик и туманностей.
Исследование космоса помогает понять происхождение Вселенной.
Большой взрыв произошёл около четырнадцати миллиардов лет назад.
Тёмная материя и тёмная энергия составляют большую часть Вселенной.
Астронавты живут на Международной космической станции и проводят эксперименты.
В будущем люди планируют полететь на Марс и основать там колонию.
Космические технологии дали нам GPS, спутниковое телевидение и интернет.
Изучение космоса вдохновляет новые поколения учёных и инженеров.
Космос хранит множество тайн, которые ещё предстоит разгадать человечеству.
""" * 20  # повторяем текст 20 раз чтобы сеть лучше выучила паттерны


# ─────────────────────────────────────────
# 2. ПОДГОТОВКА ДАННЫХ
# ─────────────────────────────────────────
# Составляем словарь: каждый уникальный символ → число
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Длина текста:    {len(text)} символов")
print(f"Уникальных букв: {vocab_size}")

# Переводим весь текст в числа
data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

# Нарезаем текст на последовательности длиной 100 символов
# Вход:  "Космос — это бесконечное простр"  (100 символов)
# Цель:  "osmос — это бесконечное простра"  (те же 100, сдвинутые на 1)
seq_len = 100

def get_batch(batch_size=64):
    # Случайные стартовые позиции
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[s : s + seq_len]     for s in starts])
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x, y


# ─────────────────────────────────────────
# 3. АРХИТЕКТУРА — символьная LSTM
# ─────────────────────────────────────────
#
# RNN (Recurrent Neural Network) — сеть с памятью
# В отличие от обычной сети, она обрабатывает последовательности:
# читает символ за символом и запоминает контекст
#
# LSTM (Long Short-Term Memory) — умная версия RNN
# Решает проблему "забывания" длинного контекста
# Внутри есть "ворота": что запомнить, что забыть, что выдать
#
# Путь данных:
#   символ → Embedding (число → вектор) → LSTM → Linear → следующий символ

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}\n")

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, num_layers=2):
        super().__init__()
        # Embedding: каждый символ → вектор из 64 чисел
        # (сеть сама учит, какие символы "похожи")
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # LSTM: 2 слоя, hidden_dim=256 — "рабочая память" сети
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.3)

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)                    # [batch, seq] → [batch, seq, 64]
        out, hidden = self.lstm(x, hidden)   # [batch, seq, 256]
        out = self.fc(out)                   # [batch, seq, vocab_size]
        return out, hidden


model = CharLSTM(vocab_size).to(device)
print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}\n")


# ─────────────────────────────────────────
# 4. ОБУЧЕНИЕ
# ─────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

epochs = 200
print_every = 20

for epoch in range(1, epochs + 1):
    model.train()
    x, y = get_batch(64)
    x, y = x.to(device), y.to(device)

    output, _ = model(x)
    # output: [batch, seq, vocab] → нужно [batch*seq, vocab]
    loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # стабилизация
    optimizer.step()

    if epoch % print_every == 0:
        print(f"Эпоха {epoch:3d}/{epochs} | Loss: {loss.item():.4f}")


# ─────────────────────────────────────────
# 5. ГЕНЕРАЦИЯ ТЕКСТА
# ─────────────────────────────────────────
def generate(seed_text, length=500, temperature=0.7):
    """
    temperature — "творческость" сети:
      0.3 = осторожная, повторяет знакомые фразы
      0.7 = баланс
      1.2 = творческая, но может нести бред
    """
    model.eval()
    input_ids = torch.tensor(
        [[char_to_idx.get(ch, 0) for ch in seed_text]],
        dtype=torch.long
    ).to(device)

    generated = seed_text
    hidden = None

    with torch.no_grad():
        # Прогреваем сеть на seed тексте
        _, hidden = model(input_ids, hidden)

        # Генерируем по одному символу
        current = input_ids[:, -1:]
        for _ in range(length):
            output, hidden = model(current, hidden)
            # Делим логиты на temperature перед softmax
            probs = torch.softmax(output[0, -1] / temperature, dim=0)
            # Сэмплируем — не берём максимум, а случайно по вероятностям
            next_id = torch.multinomial(probs, 1).item()
            generated += idx_to_char[next_id]
            current = torch.tensor([[next_id]], dtype=torch.long).to(device)

    return generated


print("\n" + "="*60)
print("ГЕНЕРАЦИЯ ТЕКСТА:")
print("="*60)

seeds = ["Космос", "Изучение", "В будущем"]
for seed in seeds:
    print(f"\n>>> Начало: '{seed}'")
    print(generate(seed, length=300, temperature=0.7))
    print()
