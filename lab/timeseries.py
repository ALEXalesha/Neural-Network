"""
Прогноз временных рядов — предсказание продаж на следующий месяц
Метод: LSTM смотрит на последние N значений и предсказывает следующее

Временной ряд — данные где порядок важен:
  [100, 110, 105, 120, 115, ???] → предсказать следующее значение
"""
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
# 1. ДАННЫЕ — ежедневные продажи магазина
# ─────────────────────────────────────────
# Симулируем 3 года продаж с трендом, сезонностью и случайным шумом

days = np.arange(3 * 365)

trend      = days * 0.05                           # медленный рост
weekly     = 20 * np.sin(2 * np.pi * days / 7)    # недельная сезонность
yearly     = 80 * np.sin(2 * np.pi * days / 365 - np.pi/2)  # годовая (спад летом)
noise      = np.random.normal(0, 15, len(days))
sales      = 300 + trend + weekly + yearly + noise
sales      = np.clip(sales, 50, None).astype(np.float32)

print(f"Данных: {len(days)} дней ({len(days)//365} года)")
print(f"Продажи: {sales.min():.0f} – {sales.max():.0f} ед/день\n")

# Нормализация
s_mean = sales.mean()
s_std  = sales.std()
sales_norm = (sales - s_mean) / s_std

# ─────────────────────────────────────────
# 2. СОЗДАНИЕ ОКОН (sliding window)
# ─────────────────────────────────────────
# Берём WINDOW последних дней → предсказываем следующие HORIZON дней
WINDOW  = 30   # смотрим на 30 дней назад
HORIZON = 7    # предсказываем 7 дней вперёд

def make_sequences(data, window, horizon):
    X, Y = [], []
    for i in range(len(data) - window - horizon):
        X.append(data[i:i+window])
        Y.append(data[i+window:i+window+horizon])
    return np.array(X), np.array(Y)

X_seq, Y_seq = make_sequences(sales_norm, WINDOW, HORIZON)
print(f"Последовательностей: {len(X_seq)}")

split = int(0.8 * len(X_seq))
X_train = torch.tensor(X_seq[:split]).unsqueeze(-1).to(device)  # [B, T, 1]
Y_train = torch.tensor(Y_seq[:split]).to(device)
X_test  = torch.tensor(X_seq[split:]).unsqueeze(-1).to(device)
Y_test  = torch.tensor(Y_seq[split:]).to(device)

# ─────────────────────────────────────────
# 3. МОДЕЛЬ — LSTM + Attention
# ─────────────────────────────────────────
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden=128, layers=2, horizon=7):
        super().__init__()
        self.lstm    = nn.LSTM(input_dim, hidden, layers,
                               batch_first=True, dropout=0.2)
        self.attn    = nn.Linear(hidden, 1)
        self.fc      = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
        )

    def forward(self, x):
        out, _ = self.lstm(x)                        # [B, T, H]
        # Attention: взвешиваем все шаги
        weights = torch.softmax(self.attn(out), dim=1)  # [B, T, 1]
        context = (weights * out).sum(dim=1)             # [B, H]
        return self.fc(context)                          # [B, horizon]

model = TimeSeriesLSTM(horizon=HORIZON).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Параметров: {params:,}\n")

# ─────────────────────────────────────────
# 4. ОБУЧЕНИЕ
# ─────────────────────────────────────────
criterion = nn.HuberLoss()   # устойчивее к выбросам чем MSE
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

EPOCHS = 200
BATCH  = 128
best_loss = float('inf')
train_losses, test_losses = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = torch.randperm(len(X_train))
    ep_loss = 0
    for i in range(0, len(X_train), BATCH):
        xb = X_train[idx[i:i+BATCH]]
        yb = Y_train[idx[i:i+BATCH]]
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        ep_loss += loss.item()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        tl = criterion(model(X_test), Y_test).item()
    train_losses.append(ep_loss / (len(X_train) // BATCH))
    test_losses.append(tl)

    if tl < best_loss:
        best_loss = tl
        torch.save(model.state_dict(), 'timeseries_model.pth')

    if epoch % 40 == 0:
        rmse = (tl ** 0.5) * s_std
        print(f"Эпоха {epoch:3d}/{EPOCHS} | Test Loss: {tl:.5f} | RMSE: {rmse:.1f} ед.")

# ─────────────────────────────────────────
# 5. РЕЗУЛЬТАТЫ
# ─────────────────────────────────────────
model.load_state_dict(torch.load('timeseries_model.pth', weights_only=True))
model.eval()

with torch.no_grad():
    pred_norm = model(X_test).cpu().numpy()
    real_norm = Y_test.cpu().numpy()

pred = pred_norm * s_std + s_mean
real = real_norm * s_std + s_mean

mae  = np.abs(pred - real).mean()
rmse = ((pred - real) ** 2).mean() ** 0.5

print(f"\n{'='*45}")
print(f"MAE:  {mae:.1f} единиц продаж")
print(f"RMSE: {rmse:.1f} единиц продаж")
print(f"Средние продажи: {s_mean:.0f} → ошибка {mae/s_mean*100:.1f}%")

# ─────────────────────────────────────────
# 6. ИНТЕРАКТИВ — прогноз на 7 дней вперёд
# ─────────────────────────────────────────
print(f"\n{'='*45}")
print("Введите последние 30 дней продаж через запятую")
print("(или Enter для прогноза на последних реальных данных)\n")

while True:
    try:
        inp = input("Продажи за 30 дней (через запятую): ").strip()
        if inp.lower() in ('выход', 'exit', 'q'):
            break

        if inp:
            vals = [float(x.strip()) for x in inp.split(',')]
            if len(vals) != WINDOW:
                print(f"Нужно ровно {WINDOW} значений\n")
                continue
            window_data = np.array(vals, dtype=np.float32)
        else:
            # Берём последние 30 дней из реальных данных
            window_data = sales[-WINDOW:]

        window_norm = (window_data - s_mean) / s_std
        x = torch.tensor(window_norm).unsqueeze(0).unsqueeze(-1).to(device)

        with torch.no_grad():
            p = model(x)[0].cpu().numpy() * s_std + s_mean

        print(f"\n  Прогноз продаж на следующие 7 дней:")
        for i, val in enumerate(p, 1):
            bar = "█" * int(val / s_mean * 10)
            print(f"  День {i}: {val:6.0f} ед.  {bar}")
        print(f"  Средний прогноз: {p.mean():.0f} ед.\n")

    except (ValueError, EOFError, KeyboardInterrupt):
        print("\nГотово!")
        break

# Графики
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 1. Весь ряд продаж
axes[0,0].plot(sales, lw=0.8, alpha=0.7)
axes[0,0].set_title('Ежедневные продажи (3 года)')
axes[0,0].set_xlabel('День'); axes[0,0].set_ylabel('Единиц')
axes[0,0].grid(True)

# 2. Кривые обучения
axes[0,1].plot(train_losses, label='Train')
axes[0,1].plot(test_losses, label='Test')
axes[0,1].set_title('Кривая обучения')
axes[0,1].legend(); axes[0,1].grid(True)

# 3. Последние 60 дней: прогноз vs реальность
last_n = 60
real_last = real[-last_n:, 0]
pred_last = pred[-last_n:, 0]
axes[1,0].plot(real_last, label='Реальные продажи', lw=2)
axes[1,0].plot(pred_last, label='Прогноз', lw=2, linestyle='--')
axes[1,0].set_title('Последние 60 дней')
axes[1,0].legend(); axes[1,0].grid(True)

# 4. Scatter реальные vs предсказанные
axes[1,1].scatter(real.flatten()[:500], pred.flatten()[:500], alpha=0.2, s=5)
axes[1,1].plot([real.min(), real.max()], [real.min(), real.max()], 'r-')
axes[1,1].set_title(f'Реальные vs Предсказанные (RMSE={rmse:.1f})')
axes[1,1].set_xlabel('Реальные'); axes[1,1].set_ylabel('Предсказанные')
axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('timeseries_results.png', dpi=100)
print("График сохранён: timeseries_results.png")
