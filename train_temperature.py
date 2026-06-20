"""
Обучение модели прогноза температуры (TempNet)
Сохраняет temperature_model.pth с весами и параметрами нормализации
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
N = 5000

month       = np.random.randint(1, 13, N).astype(float)
base_temp   = -10 + 22 * np.sin((month - 3) * np.pi / 6)
temp_today  = base_temp + np.random.normal(0, 5, N)
pressure    = 760 + np.random.normal(0, 10, N)
humidity    = 50 + 30 * np.random.rand(N)
wind        = np.random.exponential(3, N)
cloud       = np.random.uniform(0, 1, N)

temp_tomorrow = (
    temp_today * 0.8
    + base_temp * 0.2
    + (pressure - 760) * 0.05
    + wind * (-0.3)
    + cloud * (-2)
    + np.random.normal(0, 2, N)
)

X = np.stack([temp_today, pressure, humidity, wind, cloud, month], axis=1).astype(np.float32)
Y = temp_tomorrow.astype(np.float32).reshape(-1, 1)

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
Y_mean = float(Y.mean())
Y_std  = float(Y.std())

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

split   = int(0.8 * N)
X_train = torch.tensor(X_norm[:split]).to(DEVICE)
Y_train = torch.tensor(Y_norm[:split]).to(DEVICE)
X_test  = torch.tensor(X_norm[split:]).to(DEVICE)
Y_test  = torch.tensor(Y_norm[split:]).to(DEVICE)

# ── Модель ────────────────────────────────────────────────────────────────────
class TempNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),  nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)

model     = TempNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ── Обучение ──────────────────────────────────────────────────────────────────
EPOCHS = 300
print(f"Данных: {N} примеров | Обучение: {EPOCHS} эпох\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = torch.randperm(split)
    for i in range(0, split, 128):
        xb = X_train[idx[i:i+128]]
        yb = Y_train[idx[i:i+128]]
        loss = criterion(model(xb), yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    if epoch % 60 == 0:
        model.eval()
        with torch.no_grad():
            tl = criterion(model(X_test), Y_test).item()
        rmse = (tl ** 0.5) * Y_std
        print(f"Эпоха {epoch:3d}/{EPOCHS} | RMSE: {rmse:.2f}°C")

# ── Финальная оценка ──────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    pred = model(X_test).cpu().numpy() * Y_std + Y_mean
    real = Y_test.cpu().numpy() * Y_std + Y_mean

mae = np.abs(pred - real).mean()
print(f"\nСредняя ошибка: {mae:.2f}°C")

# ── Сохранение ────────────────────────────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "X_mean": X_mean.tolist(),
    "X_std":  X_std.tolist(),
    "Y_mean": Y_mean,
    "Y_std":  Y_std,
}, BASE / "temperature_model.pth")

print("Модель сохранена: temperature_model.pth")
