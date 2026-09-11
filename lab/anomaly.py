"""
Обнаружение аномалий — поиск мошеннических банковских транзакций
Задача: из 10 000 транзакций найти ~2% подозрительных

Проблема: данные сильно несбалансированы (98% нормальных, 2% мошенничество)
Решение: Autoencoder — учится восстанавливать нормальные транзакции,
         на аномальных ошибка восстановления выше → флажок "подозрительно"
"""
import sys, io
if hasattr(sys.stdout, "buffer"): sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}\n")

# ─────────────────────────────────────────
# 1. ДАННЫЕ — банковские транзакции
# ─────────────────────────────────────────
# Признаки каждой транзакции:
# сумма, время суток, частота (транзакций в день), страна (0/1),
# тип (онлайн/терминал), баланс до/после, расстояние от дома

N_NORMAL = 9800
N_FRAUD  = 200
N        = N_NORMAL + N_FRAUD

def make_normal(n):
    return np.column_stack([
        np.random.lognormal(6, 1, n),       # сумма: 100–5000 руб (нормальное)
        np.random.uniform(8, 22, n),         # время: рабочие часы
        np.random.poisson(3, n).astype(float),  # частота: ~3 в день
        np.random.binomial(1, 0.05, n).astype(float),  # заграница: 5%
        np.random.binomial(1, 0.6, n).astype(float),   # онлайн: 60%
        np.random.uniform(1000, 100000, n),  # баланс
        np.random.uniform(0, 10, n),         # км от дома
    ]).astype(np.float32)

def make_fraud(n):
    return np.column_stack([
        np.random.lognormal(9, 1.5, n),     # АНОМАЛИЯ: крупные суммы
        np.random.uniform(0, 6, n),          # АНОМАЛИЯ: ночное время
        np.random.poisson(15, n).astype(float),  # АНОМАЛИЯ: много транзакций
        np.random.binomial(1, 0.7, n).astype(float),  # АНОМАЛИЯ: часто заграница
        np.random.binomial(1, 0.95, n).astype(float), # АНОМАЛИЯ: почти всё онлайн
        np.random.uniform(100, 5000, n),     # АНОМАЛИЯ: низкий баланс
        np.random.uniform(500, 5000, n),     # АНОМАЛИЯ: далеко от дома
    ]).astype(np.float32)

X_normal = make_normal(N_NORMAL)
X_fraud  = make_fraud(N_FRAUD)

X = np.vstack([X_normal, X_fraud])
y = np.array([0] * N_NORMAL + [1] * N_FRAUD)  # 0=норма, 1=мошенничество

print(f"Транзакций всего:      {N}")
print(f"  Нормальных:          {N_NORMAL} ({N_NORMAL/N*100:.0f}%)")
print(f"  Мошеннических:       {N_FRAUD}  ({N_FRAUD/N*100:.0f}%)\n")

# Нормализация
X_mean = X_normal.mean(axis=0)
X_std  = X_normal.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

# Autoencoder обучается ТОЛЬКО на нормальных транзакциях
X_train = torch.tensor(X_norm[:N_NORMAL], dtype=torch.float32).to(device)
X_all   = torch.tensor(X_norm, dtype=torch.float32).to(device)

# ─────────────────────────────────────────
# 2. AUTOENCODER
# ─────────────────────────────────────────
# Encoder: сжимает 7 признаков → 2 (бутылочное горлышко)
# Decoder: восстанавливает 2 → 7
#
# Нормальные транзакции: сеть учится хорошо восстанавливать → ошибка мала
# Мошеннические:        паттерн незнаком → ошибка восстановления велика

class Autoencoder(nn.Module):
    def __init__(self, input_dim=7, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),   # сжатое представление
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),    # восстановленный вектор
        )

    def forward(self, x):
        z = self.encoder(x)              # сжимаем
        return self.decoder(z)           # восстанавливаем

model = Autoencoder().to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Параметров: {params:,}\n")

# ─────────────────────────────────────────
# 3. ОБУЧЕНИЕ
# ─────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 200
BATCH  = 256

print("Обучение на НОРМАЛЬНЫХ транзакциях...")
losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = torch.randperm(N_NORMAL)
    epoch_loss = 0
    for i in range(0, N_NORMAL, BATCH):
        xb = X_train[idx[i:i+BATCH]]
        reconstructed = model(xb)
        loss = criterion(reconstructed, xb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg = epoch_loss / (N_NORMAL // BATCH)
    losses.append(avg)
    if epoch % 40 == 0:
        print(f"Эпоха {epoch:3d}/{EPOCHS} | Loss: {avg:.6f}")

# ─────────────────────────────────────────
# 4. ОБНАРУЖЕНИЕ АНОМАЛИЙ
# ─────────────────────────────────────────
model.eval()
with torch.no_grad():
    reconstructed = model(X_all)
    # Ошибка восстановления для каждой транзакции
    errors = ((X_all - reconstructed) ** 2).mean(dim=1).cpu().numpy()

# Порог: 95-й перцентиль ошибок на нормальных данных
threshold = np.percentile(errors[:N_NORMAL], 95)
predictions = (errors > threshold).astype(int)

# Метрики
TP = ((predictions == 1) & (y == 1)).sum()  # поймали мошенничество
FP = ((predictions == 1) & (y == 0)).sum()  # ложная тревога
FN = ((predictions == 0) & (y == 1)).sum()  # пропустили мошенничество
TN = ((predictions == 0) & (y == 0)).sum()  # правильно пропустили

precision = TP / (TP + FP + 1e-8)
recall    = TP / (TP + FN + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)

print(f"\n{'='*45}")
print(f"Результаты обнаружения:")
print(f"  Найдено мошенничеств:   {TP}/{N_FRAUD} ({TP/N_FRAUD*100:.0f}%)")
print(f"  Ложных тревог:          {FP}/{N_NORMAL}")
print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
print(f"\n  Ошибка норм. транзакций:  {errors[:N_NORMAL].mean():.4f} (±{errors[:N_NORMAL].std():.4f})")
print(f"  Ошибка мошеннических:     {errors[N_NORMAL:].mean():.4f} (±{errors[N_NORMAL:].std():.4f})")

# ─────────────────────────────────────────
# 5. ИНТЕРАКТИВНЫЙ РЕЖИМ
# ─────────────────────────────────────────
print(f"\n{'='*45}")
print("Проверить транзакцию:")
print("(Enter для выхода)\n")

while True:
    try:
        print("Введите данные транзакции:")
        amount   = float(input("  Сумма (руб): "))
        hour     = float(input("  Час (0-23): "))
        freq     = float(input("  Транзакций сегодня: "))
        foreign  = float(input("  Заграница? (0/1): "))
        online   = float(input("  Онлайн? (0/1): "))
        balance  = float(input("  Баланс (руб): "))
        distance = float(input("  Км от дома: "))

        feat = np.array([amount, hour, freq, foreign, online, balance, distance], dtype=np.float32)
        feat_norm = (feat - X_mean) / X_std
        inp = torch.tensor(feat_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            rec = model(inp)
            err = ((inp - rec) ** 2).mean().item()

        risk = err / threshold
        if risk > 1.5:
            verdict = "🚨 ВЫСОКИЙ РИСК МОШЕННИЧЕСТВА"
        elif risk > 1.0:
            verdict = "⚠️  Подозрительно"
        else:
            verdict = "✅ Нормальная транзакция"

        print(f"\n  Ошибка восстановления: {err:.4f} (порог: {threshold:.4f})")
        print(f"  Риск: {risk:.2f}x  →  {verdict}\n")

    except (ValueError, EOFError, KeyboardInterrupt):
        print("\nГотово!")
        break

# График
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Ошибки восстановления
axes[0].hist(errors[:N_NORMAL], bins=50, alpha=0.7, label='Нормальные', color='blue')
axes[0].hist(errors[N_NORMAL:], bins=30, alpha=0.7, label='Мошеннические', color='red')
axes[0].axvline(threshold, color='black', linestyle='--', label=f'Порог={threshold:.3f}')
axes[0].set_title('Ошибки восстановления')
axes[0].set_xlabel('MSE')
axes[0].legend()

# 2. Latent space (2D представление)
with torch.no_grad():
    latent = model.encoder(X_all).cpu().numpy()
axes[1].scatter(latent[:N_NORMAL, 0], latent[:N_NORMAL, 1], alpha=0.1, s=5, c='blue', label='Норма')
axes[1].scatter(latent[N_NORMAL:, 0], latent[N_NORMAL:, 1], alpha=0.7, s=20, c='red', label='Мошенничество')
axes[1].set_title('Латентное пространство (2D)')
axes[1].legend()

# 3. Loss
axes[2].plot(losses)
axes[2].set_title('Кривая обучения')
axes[2].set_xlabel('Эпоха')
axes[2].set_ylabel('Loss')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('anomaly_results.png', dpi=100)
print("График сохранён: anomaly_results.png")
