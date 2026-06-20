"""
Регрессия #2 — предсказание температуры завтра
Входные данные: температура сегодня, давление, влажность, скорость ветра, месяц
Выход: температура завтра (°C)
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
# 1. ДАННЫЕ — синтетическая погода (Москва)
# ─────────────────────────────────────────
N = 5000

month    = np.random.randint(1, 13, N).astype(float)
# Базовая температура по месяцам (как в Москве)
base_temp = -10 + 22 * np.sin((month - 3) * np.pi / 6)  # пик в июле, минимум в январе

temp_today  = base_temp + np.random.normal(0, 5, N)
pressure    = 760 + np.random.normal(0, 10, N)           # давление в мм рт.ст.
humidity    = 50 + 30 * np.random.rand(N)                # влажность 50–80%
wind        = np.random.exponential(3, N)                # скорость ветра м/с
cloud       = np.random.uniform(0, 1, N)                 # облачность 0–1

# Температура завтра = сегодня + небольшие изменения
temp_tomorrow = (
    temp_today * 0.8 +                    # инерция температуры
    base_temp * 0.2 +                     # тяготение к норме
    (pressure - 760) * 0.05 +            # высокое давление = теплее
    wind * (-0.3) +                       # ветер = холоднее
    cloud * (-2) +                        # облака = холоднее днём
    np.random.normal(0, 2, N)             # случайность
)

X = np.stack([temp_today, pressure, humidity, wind, cloud, month], axis=1).astype(np.float32)
Y = temp_tomorrow.astype(np.float32).reshape(-1, 1)

print(f"Данных: {N} дней")
print(f"Диапазон температур: {Y.min():.0f}°C … {Y.max():.0f}°C\n")

# Нормализация
X_mean = X.mean(axis=0); X_std = X.std(axis=0)
Y_mean = float(Y.mean()); Y_std = float(Y.std())
X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

split = int(0.8 * N)
X_train = torch.tensor(X_norm[:split]).to(device)
Y_train = torch.tensor(Y_norm[:split]).to(device)
X_test  = torch.tensor(X_norm[split:]).to(device)
Y_test  = torch.tensor(Y_norm[split:]).to(device)

# ─────────────────────────────────────────
# 2. МОДЕЛЬ
# ─────────────────────────────────────────
class TempNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)

model = TempNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ─────────────────────────────────────────
# 3. ОБУЧЕНИЕ
# ─────────────────────────────────────────
EPOCHS = 300
losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = torch.randperm(split)
    for i in range(0, split, 128):
        xb = X_train[idx[i:i+128]]
        yb = Y_train[idx[i:i+128]]
        loss = criterion(model(xb), yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        tl = criterion(model(X_test), Y_test).item()
    losses.append(tl)
    if epoch % 60 == 0:
        rmse = (tl ** 0.5) * Y_std
        print(f"Эпоха {epoch:3d}/{EPOCHS} | RMSE: {rmse:.2f}°C")

# ─────────────────────────────────────────
# 4. РЕЗУЛЬТАТЫ
# ─────────────────────────────────────────
model.eval()
with torch.no_grad():
    pred = model(X_test).cpu().numpy() * Y_std + Y_mean
    real = Y_test.cpu().numpy() * Y_std + Y_mean

mae  = np.abs(pred - real).mean()
print(f"\nСредняя ошибка предсказания: {mae:.2f}°C")

print("\nПримеры:")
print(f"{'Сегодня':>8} {'Давление':>9} {'Влажн':>6} {'Ветер':>6} {'Реально':>8} {'Прогноз':>8}")
print("-" * 55)
for i in range(8):
    f = X[split + i]
    print(f"{f[0]:>8.1f} {f[1]:>9.0f} {f[2]:>6.0f}% {f[3]:>6.1f}  {real[i,0]:>7.1f}°  {pred[i,0]:>7.1f}°")

# ─────────────────────────────────────────
# 5. ИНТЕРАКТИВ
# ─────────────────────────────────────────
print("\nВведите данные для прогноза (Enter = выход):")
while True:
    try:
        t   = float(input("  Температура сегодня (°C): "))
        p   = float(input("  Давление (мм рт.ст., норма=760): "))
        h   = float(input("  Влажность (%): "))
        w   = float(input("  Ветер (м/с): "))
        c   = float(input("  Облачность (0-1): "))
        m   = float(input("  Месяц (1-12): "))
        feat = np.array([t, p, h, w, c, m], dtype=np.float32)
        feat_n = (feat - X_mean) / X_std
        inp = torch.tensor(feat_n).unsqueeze(0).to(device)
        with torch.no_grad():
            result = model(inp).item() * Y_std + Y_mean
        print(f"\n  🌡 Прогноз на завтра: {result:.1f}°C\n")
    except (ValueError, EOFError, KeyboardInterrupt):
        print("\nГотово!")
        break

# График
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.scatter(real[:200], pred[:200], alpha=0.5, s=15)
plt.plot([real.min(), real.max()], [real.min(), real.max()], 'r-')
plt.xlabel('Реальная °C'); plt.ylabel('Прогноз °C')
plt.title(f'Прогноз температуры (MAE={mae:.2f}°C)')
plt.grid(True)
plt.subplot(1,2,2)
days = range(len(real[:60]))
plt.plot(days, real[:60], label='Реальная', lw=2)
plt.plot(days, pred[:60], label='Прогноз', lw=2, linestyle='--')
plt.xlabel('День'); plt.ylabel('°C')
plt.title('60 дней: прогноз vs реальность')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig('temperature_results.png', dpi=100)
print("График сохранён: temperature_results.png")
