"""
Обучение Autoencoder для обнаружения аномалий в банковских транзакциях
Сохраняет anomaly_model.pth
"""
import sys, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

BASE   = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {DEVICE}\n")

# ── Данные ────────────────────────────────────────────────────────────────────
N_NORMAL = 9800
N_FRAUD  = 200

def make_normal(n):
    # Диапазоны соответствуют слайдерам UI:
    # amount(10..100000), hour(0..23), freq(0..30),
    # foreign(0/1), online(0/1), balance(0..200000), distance(0..5000)
    return np.column_stack([
        np.clip(np.random.lognormal(7.5, 1.2, n), 10, 50000),  # сумма 10–50 000 руб
        np.random.uniform(8, 22, n),                             # час: рабочее время
        np.minimum(np.random.poisson(3, n), 10).astype(float),  # транзакций сегодня 0–10
        np.random.binomial(1, 0.05, n).astype(float),           # заграница: редко
        np.random.binomial(1, 0.6, n).astype(float),            # онлайн: 60%
        np.random.uniform(5000, 200000, n),                      # баланс 5 000–200 000
        np.random.uniform(0, 50, n),                             # расстояние 0–50 км
    ]).astype(np.float32)

def make_fraud(n):
    return np.column_stack([
        np.random.uniform(60000, 100000, n),                     # сумма: очень крупная
        np.random.uniform(0, 6, n),                              # час: ночь
        np.minimum(np.random.poisson(18, n), 30).astype(float), # много транзакций
        np.random.binomial(1, 0.75, n).astype(float),           # заграница: часто
        np.random.binomial(1, 0.95, n).astype(float),           # онлайн: почти всегда
        np.random.uniform(500, 8000, n),                         # баланс: почти пуст
        np.random.uniform(500, 5000, n),                         # расстояние: далеко
    ]).astype(np.float32)

X_normal = make_normal(N_NORMAL)
X_fraud  = make_fraud(N_FRAUD)
X        = np.vstack([X_normal, X_fraud])
y        = np.array([0]*N_NORMAL + [1]*N_FRAUD)

X_mean = X_normal.mean(axis=0)
X_std  = X_normal.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

X_train = torch.tensor(X_norm[:N_NORMAL], dtype=torch.float32).to(DEVICE)
X_all   = torch.tensor(X_norm, dtype=torch.float32).to(DEVICE)

print(f"Транзакций: {N_NORMAL+N_FRAUD} (нормальных: {N_NORMAL}, мошеннических: {N_FRAUD})")

# ── Модель ────────────────────────────────────────────────────────────────────
class AnomalyAE(nn.Module):
    def __init__(self, input_dim=7, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32),         nn.ReLU(),
            nn.Linear(32, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model     = AnomalyAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ── Обучение (только на нормальных!) ──────────────────────────────────────────
EPOCHS = 200
print(f"\nОбучение: {EPOCHS} эпох (только нормальные транзакции)...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = torch.randperm(N_NORMAL)
    epoch_loss = 0
    for i in range(0, N_NORMAL, 256):
        xb = X_train[idx[i:i+256]]
        loss = criterion(model(xb), xb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        epoch_loss += loss.item()
    if epoch % 50 == 0:
        print(f"  Эпоха {epoch}/{EPOCHS} | Loss: {epoch_loss:.5f}")

# ── Порог и метрики ───────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    rec    = model(X_all)
    errors = ((X_all - rec) ** 2).mean(dim=1).cpu().numpy()

threshold = float(np.percentile(errors[:N_NORMAL], 95))
preds = (errors > threshold).astype(int)

TP = ((preds==1)&(y==1)).sum()
FP = ((preds==1)&(y==0)).sum()
FN = ((preds==0)&(y==1)).sum()
precision = TP/(TP+FP+1e-8)
recall    = TP/(TP+FN+1e-8)
f1        = 2*precision*recall/(precision+recall+1e-8)

print(f"\nПорог: {threshold:.4f}")
print(f"Найдено мошенничеств: {TP}/{N_FRAUD} ({TP/N_FRAUD*100:.0f}%)")
print(f"Ложных тревог: {FP}/{N_NORMAL}")
print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")

# ── Сохранение ────────────────────────────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "X_mean":      X_mean.tolist(),
    "X_std":       X_std.tolist(),
    "threshold":   threshold,
}, BASE / "anomaly_model.pth")

print("\nМодель сохранена: anomaly_model.pth")
