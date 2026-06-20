"""
Регрессия — предсказание цены квартиры
Входные данные: площадь, комнаты, этаж, район, возраст дома
Выход: цена в миллионах рублей

Отличие от классификации:
  Классификация → "это кот или собака?" (категория)
  Регрессия      → "сколько стоит эта квартира?" (число)
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
# 1. ГЕНЕРАЦИЯ ДАННЫХ
# ─────────────────────────────────────────
# Симулируем рынок недвижимости Москвы
# Реальная цена зависит от: площадь * цена_за_метр(район) - скидка_за_этаж - скидка_за_возраст

N = 10_000

# Признаки
area     = np.random.uniform(25, 150, N)          # площадь: 25–150 м²
rooms    = np.random.randint(1, 6, N).astype(float)  # 1–5 комнат
floor    = np.random.randint(1, 26, N).astype(float) # этаж 1–25
district = np.random.randint(0, 5, N).astype(float)  # район: 0=окраина … 4=центр
age      = np.random.uniform(0, 60, N)            # возраст дома: 0–60 лет

# Цена за м² в зависимости от района (тыс. руб/м²)
price_per_m2 = np.array([120, 160, 200, 260, 350])  # окраина → центр
base_price = area * price_per_m2[district.astype(int)]

# Поправки
floor_bonus  = np.where(floor == 1, -5000, 0)        # 1й этаж — дешевле
floor_bonus += np.where(floor >= 15, 3000, 0)        # высокий этаж — дороже
age_penalty  = age * 500                              # старый дом — дешевле
rooms_bonus  = rooms * 8000                           # больше комнат — дороже
noise        = np.random.normal(0, 15000, N)          # случайный шум рынка

price = (base_price + floor_bonus + rooms_bonus - age_penalty + noise) / 1_000_000

X = np.stack([area, rooms, floor, district, age], axis=1).astype(np.float32)
Y = price.astype(np.float32).reshape(-1, 1)

print(f"Данных: {N} квартир")
print(f"Цена: от {Y.min():.1f} до {Y.max():.1f} млн руб")
print(f"Средняя цена: {Y.mean():.1f} млн руб\n")

# ─────────────────────────────────────────
# 2. НОРМАЛИЗАЦИЯ
# ─────────────────────────────────────────
# Важно! Нейронки плохо работают с большими числами.
# Нормализуем признаки: среднее=0, стд=1

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
Y_mean = Y.mean()
Y_std  = Y.std()

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

# Разбиваем на train/test
split = int(0.8 * N)
X_train = torch.tensor(X_norm[:split]).to(device)
Y_train = torch.tensor(Y_norm[:split]).to(device)
X_test  = torch.tensor(X_norm[split:]).to(device)
Y_test  = torch.tensor(Y_norm[split:]).to(device)

print(f"Обучающих: {split}, тестовых: {N - split}\n")

# ─────────────────────────────────────────
# 3. МОДЕЛЬ
# ─────────────────────────────────────────
# Для регрессии:
# - Последний слой без активации (выход — любое число)
# - Функция потерь: MSE (среднеквадратичная ошибка) вместо CrossEntropy

class PriceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),      # 5 входных признаков
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 1),      # 1 выходное число — цена
        )

    def forward(self, x):
        return self.net(x)

model = PriceNet().to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Параметров: {params:,}\n")

# ─────────────────────────────────────────
# 4. ОБУЧЕНИЕ
# ─────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

EPOCHS    = 500
BATCH     = 256
train_losses, test_losses = [], []

best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    model.train()
    # Мини-батчи
    idx = torch.randperm(split)
    epoch_loss = 0
    for i in range(0, split, BATCH):
        batch_idx = idx[i:i+BATCH]
        xb, yb = X_train[batch_idx], Y_train[batch_idx]

        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Тест
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, Y_test).item()

    avg_train = epoch_loss / (split // BATCH)
    scheduler.step(test_loss)

    train_losses.append(avg_train)
    test_losses.append(test_loss)

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'price_model.pth')

    if epoch % 50 == 0:
        # Переводим MSE обратно в рубли
        rmse_mln = (test_loss ** 0.5) * Y_std
        print(f"Эпоха {epoch:3d}/{EPOCHS} | Train: {avg_train:.4f} | Test: {test_loss:.4f} | RMSE: {rmse_mln:.2f} млн руб")

# ─────────────────────────────────────────
# 5. ОЦЕНКА
# ─────────────────────────────────────────
model.load_state_dict(torch.load('price_model.pth', weights_only=True))
model.eval()

with torch.no_grad():
    pred_norm = model(X_test).cpu().numpy()
    pred = pred_norm * Y_std + Y_mean
    real = Y_test.cpu().numpy() * Y_std + Y_mean

mae  = np.abs(pred - real).mean()
rmse = ((pred - real) ** 2).mean() ** 0.5

print(f"\n{'='*45}")
print(f"Результаты на тестовых данных:")
print(f"  MAE  (средняя ошибка): {mae:.3f} млн руб")
print(f"  RMSE (корень из MSE):  {rmse:.3f} млн руб")

# ─────────────────────────────────────────
# 6. ПРИМЕРЫ ПРЕДСКАЗАНИЙ
# ─────────────────────────────────────────
print(f"\nПримеры предсказаний:")
print(f"{'Площадь':>8} {'Комн':>5} {'Этаж':>5} {'Район':>6} {'Возраст':>8} | {'Реальная':>10} {'Предсказ':>10}")
print("-" * 65)

for i in range(10):
    feat = X[split + i]
    real_p = Y[split + i, 0]
    feat_norm = (feat - X_mean) / X_std
    inp = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_p = (model(inp).item() * Y_std + Y_mean)
    district_names = ["окраина", "спальный", "средний", "близко", "центр"]
    print(f"{feat[0]:>8.0f} {feat[1]:>5.0f} {feat[2]:>5.0f} {district_names[int(feat[3])]:>6} {feat[4]:>8.0f}  | {real_p:>9.2f} {pred_p:>9.2f}")

# ─────────────────────────────────────────
# 7. ИНТЕРАКТИВНЫЙ РЕЖИМ
# ─────────────────────────────────────────
print(f"\n{'='*45}")
print("Введите параметры квартиры:\n")

district_names = {"окраина": 0, "спальный": 1, "средний": 2, "близко": 3, "центр": 4}

while True:
    try:
        print("Районы: окраина / спальный / средний / близко / центр")
        area_in   = float(input("Площадь (м²): "))
        rooms_in  = float(input("Комнат: "))
        floor_in  = float(input("Этаж: "))
        dist_in   = input("Район: ").strip().lower()
        age_in    = float(input("Возраст дома (лет): "))

        d = district_names.get(dist_in, 2)
        feat = np.array([area_in, rooms_in, floor_in, d, age_in], dtype=np.float32)
        feat_norm = (feat - X_mean) / X_std
        inp = torch.tensor(feat_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            price_pred = model(inp).item() * Y_std + Y_mean

        print(f"\n💰 Предсказанная цена: {price_pred:.2f} млн руб\n")

    except (ValueError, EOFError, KeyboardInterrupt):
        print("\nГотово!")
        break

# График
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train', alpha=0.7)
plt.plot(test_losses, label='Test', alpha=0.7)
plt.xlabel('Эпоха'); plt.ylabel('MSE Loss')
plt.title('Кривая обучения'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(real[:500], pred[:500], alpha=0.3, s=10)
plt.plot([real.min(), real.max()], [real.min(), real.max()], 'r-', lw=2, label='Идеал')
plt.xlabel('Реальная цена (млн)'); plt.ylabel('Предсказанная цена (млн)')
plt.title('Реальная vs Предсказанная'); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig('regression_results.png', dpi=100)
print("График сохранён: regression_results.png")
