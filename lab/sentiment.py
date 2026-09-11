"""
Анализ тональности текста (Sentiment Analysis)
Задача: определить — отзыв позитивный, негативный или нейтральный

Метод: LSTM читает слова отзыва последовательно и выдаёт одну из 3 меток
  Позитивный: "отличный товар, очень доволен"
  Негативный: "ужасное качество, не рекомендую"
  Нейтральный: "товар пришёл вовремя, как описано"
"""
import torch
import torch.nn as nn
import numpy as np
import re
from collections import Counter

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}\n")

# ─────────────────────────────────────────
# 1. ДАННЫЕ — русские отзывы на товары
# ─────────────────────────────────────────
# 0=негатив, 1=нейтрал, 2=позитив

RAW_DATA = [
    # ПОЗИТИВНЫЕ
    ("отличный товар очень доволен покупкой рекомендую всем", 2),
    ("быстрая доставка качество супер спасибо продавцу", 2),
    ("великолепное качество все соответствует описанию", 2),
    ("шикарная вещь буду заказывать ещё очень нравится", 2),
    ("превосходно всё отлично работает счастлив покупкой", 2),
    ("замечательный товар упаковка хорошая пришло быстро", 2),
    ("восхитительно качество на высоте очень рад", 2),
    ("прекрасная покупка цена качество идеальное соотношение", 2),
    ("классный продукт работает отлично доволен", 2),
    ("топ товар лучшее что брал очень рекомендую", 2),
    ("потрясающе всё понравилось заказал ещё раз", 2),
    ("отличная вещь качество выше ожиданий спасибо", 2),
    ("супер товар доставили быстро всё целое", 2),
    ("превосходное качество очень доволен рекомендую", 2),
    ("отличный продавец быстрая доставка качество хорошее", 2),
    # НЕГАТИВНЫЕ
    ("ужасное качество полный брак не рекомендую никому", 0),
    ("разочарован товар не соответствует описанию обман", 0),
    ("сломалось через день деньги на ветер не берите", 0),
    ("отвратительное качество пластик дешёвый выброшу", 0),
    ("мусор а не товар зря потратил деньги", 0),
    ("ужас полный не работает сразу после распаковки", 0),
    ("плохое качество запах неприятный не советую", 0),
    ("разочарование полное не то что на картинке", 0),
    ("бракованный товар продавец не отвечает кошмар", 0),
    ("жуткое качество зря купил верните деньги", 0),
    ("отстой не работает как написано врут", 0),
    ("ужасная доставка товар пришёл сломанным разочарован", 0),
    ("не покупайте плохое качество маленький размер", 0),
    ("жалею что купил полное разочарование хлам", 0),
    ("плохо сделано кривое страшное не советую", 0),
    # НЕЙТРАЛЬНЫЕ
    ("товар пришёл вовремя всё как описано нормально", 1),
    ("среднее качество цена соответствует ожидал большего", 1),
    ("обычный товар ничего особенного но работает", 1),
    ("нормальная вещь без восторга без разочарования", 1),
    ("доставили в срок качество среднее как везде", 1),
    ("товар получен соответствует описанию вопросов нет", 1),
    ("среднее качество но за эту цену нормально", 1),
    ("ничего особенного обычный товар работает нормально", 1),
    ("пришло вовремя внешний вид нормальный буду использовать", 1),
    ("товар как товар не хуже и не лучше других", 1),
    ("нормальное качество цена приемлема посмотрим как долго", 1),
    ("получил без проблем пока работает нормально", 1),
    ("стандартное качество как у всех аналогов", 1),
    ("всё ок нет ни плюсов ни минусов обычный товар", 1),
    ("нейтрально товар есть функции выполняет", 1),
]

# Аугментация — увеличиваем датасет перестановками слов
def augment(text, n=5):
    words = text.split()
    results = [text]
    for _ in range(n):
        idx = list(range(len(words)))
        np.random.shuffle(idx)
        results.append(' '.join(words[i] for i in idx))
    return results

all_texts, all_labels = [], []
for text, label in RAW_DATA:
    for aug in augment(text, 15):
        all_texts.append(aug)
        all_labels.append(label)

print(f"Отзывов после аугментации: {len(all_texts)}")

# ─────────────────────────────────────────
# 2. СЛОВАРЬ И ТОКЕНИЗАЦИЯ
# ─────────────────────────────────────────
def tokenize(text):
    return text.lower().split()

# Строим словарь
counter = Counter()
for t in all_texts:
    counter.update(tokenize(t))

vocab = {'<PAD>': 0, '<UNK>': 1}
for word, _ in counter.most_common(500):
    vocab[word] = len(vocab)

vocab_size = len(vocab)
MAX_LEN    = 20
print(f"Словарь: {vocab_size} слов\n")

def encode(text, max_len=MAX_LEN):
    ids = [vocab.get(w, 1) for w in tokenize(text)]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids

# Датасет
X_enc = np.array([encode(t) for t in all_texts])
Y_enc = np.array(all_labels)

# Перемешиваем
idx = np.random.permutation(len(X_enc))
X_enc, Y_enc = X_enc[idx], Y_enc[idx]

split = int(0.8 * len(X_enc))
X_train = torch.tensor(X_enc[:split], dtype=torch.long).to(device)
Y_train = torch.tensor(Y_enc[:split], dtype=torch.long).to(device)
X_test  = torch.tensor(X_enc[split:], dtype=torch.long).to(device)
Y_test  = torch.tensor(Y_enc[split:], dtype=torch.long).to(device)

# ─────────────────────────────────────────
# 3. МОДЕЛЬ — LSTM классификатор
# ─────────────────────────────────────────
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, n_classes=3):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm    = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                               bidirectional=True, num_layers=2, dropout=0.3)
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        emb = self.dropout(self.embed(x))
        out, (h, _) = self.lstm(emb)
        # Берём последнее скрытое состояние обоих направлений
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))

model = SentimentLSTM(vocab_size).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Параметров: {params:,}\n")

# ─────────────────────────────────────────
# 4. ОБУЧЕНИЕ
# ─────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

EPOCHS = 200
best_acc = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 64):
        xb = X_train[idx[i:i+64]]
        yb = Y_train[idx[i:i+64]]
        loss = criterion(model(xb), yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    scheduler.step()

    if epoch % 40 == 0:
        model.eval()
        with torch.no_grad():
            preds = model(X_test).argmax(1)
            acc   = (preds == Y_test).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'sentiment_model.pth')
        print(f"Эпоха {epoch:3d}/{EPOCHS} | Точность: {acc*100:.1f}%")

model.load_state_dict(torch.load('sentiment_model.pth', weights_only=True))

# ─────────────────────────────────────────
# 5. ТЕСТ
# ─────────────────────────────────────────
LABEL_NAMES = {0: "😡 Негативный", 1: "😐 Нейтральный", 2: "😊 Позитивный"}

model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(1)
    acc   = (preds == Y_test).float().mean().item()

print(f"\n{'='*45}")
print(f"Итоговая точность: {acc*100:.1f}%  (лучшая: {best_acc*100:.1f}%)\n")

# Матрица ошибок
from collections import defaultdict
matrix = defaultdict(lambda: defaultdict(int))
for p, t in zip(preds.cpu().numpy(), Y_test.cpu().numpy()):
    matrix[t][p] += 1

print("Матрица ошибок (строки=реальные, столбцы=предсказанные):")
print(f"{'':12} {'Негатив':>9} {'Нейтрал':>9} {'Позитив':>9}")
for t in range(3):
    row = [matrix[t][p] for p in range(3)]
    print(f"{LABEL_NAMES[t][:8]:12} {row[0]:>9} {row[1]:>9} {row[2]:>9}")

# ─────────────────────────────────────────
# 6. ИНТЕРАКТИВНЫЙ РЕЖИМ
# ─────────────────────────────────────────
def predict_sentiment(text):
    enc = torch.tensor([encode(text)], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(enc)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred   = probs.argmax()
    return pred, probs

print(f"\n{'='*45}")
print("Введите отзыв для анализа (Enter = выход):\n")

# Тест на конкретных примерах
test_phrases = [
    "отличный товар очень доволен",
    "ужасное качество сломалось сразу",
    "товар пришёл нормально всё ок",
    "превосходно рекомендую всем",
    "разочарован плохое качество",
]
print("Примеры:")
for phrase in test_phrases:
    pred, probs = predict_sentiment(phrase)
    bar = f"neg={probs[0]*100:.0f}% neu={probs[1]*100:.0f}% pos={probs[2]*100:.0f}%"
    print(f"  '{phrase[:35]:35}' → {LABEL_NAMES[pred]} ({bar})")

print()
while True:
    try:
        text = input("Отзыв: ").strip()
        if not text or text.lower() in ('выход', 'exit'):
            break
        pred, probs = predict_sentiment(text)
        print(f"  → {LABEL_NAMES[pred]}")
        print(f"     Негатив: {probs[0]*100:.1f}%  Нейтрал: {probs[1]*100:.1f}%  Позитив: {probs[2]*100:.1f}%\n")
    except (EOFError, KeyboardInterrupt):
        break

print("Готово!")
