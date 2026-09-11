"""
Обучение модели контроля качества (DefectNet)
Вход: 8 измерений детали → Выход: норма / брак
Сохраняет defect_model.pth
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

# ── Данные — измерения деталей ────────────────────────────────────────────────
# Признаки: толщина(мм), масса(г), твёрдость(HB), шероховатость(Ra),
#           длина(мм), ширина(мм), температура обработки(°C), время(с)

def make_normal(n):
    """Детали в пределах допуска"""
    return np.column_stack([
        np.random.normal(10.0, 0.05, n),    # толщина: 10.0 ± 0.05 мм
        np.random.normal(250.0, 5.0, n),    # масса: 250 ± 5 г
        np.random.normal(200.0, 10.0, n),   # твёрдость: 200 ± 10 HB
        np.random.normal(1.6, 0.2, n),      # шероховатость: 1.6 ± 0.2 Ra
        np.random.normal(100.0, 0.1, n),    # длина: 100 ± 0.1 мм
        np.random.normal(50.0, 0.08, n),    # ширина: 50 ± 0.08 мм
        np.random.normal(850.0, 20.0, n),   # температура: 850 ± 20 °C
        np.random.normal(120.0, 5.0, n),    # время обработки: 120 ± 5 с
    ]).astype(np.float32)

def make_defect_thickness(n):
    """Брак: отклонение по толщине (трещина/деформация)"""
    d = make_normal(n)
    d[:, 0] += np.random.choice([-1, 1], n) * np.random.uniform(0.15, 0.4, n)
    d[:, 3] += np.random.uniform(1.0, 3.0, n)  # шероховатость выше
    return d

def make_defect_mass(n):
    """Брак: отклонение по массе (пустоты/включения)"""
    d = make_normal(n)
    d[:, 1] += np.random.choice([-1, 1], n) * np.random.uniform(20, 50, n)
    d[:, 2] += np.random.choice([-1, 1], n) * np.random.uniform(30, 60, n)
    return d

def make_defect_thermal(n):
    """Брак: нарушение термообработки"""
    d = make_normal(n)
    d[:, 6] += np.random.choice([-1, 1], n) * np.random.uniform(80, 150, n)
    d[:, 7] += np.random.choice([-1, 1], n) * np.random.uniform(30, 60, n)
    d[:, 2] -= np.random.uniform(40, 80, n)  # твёрдость ниже
    return d

N = 4000
N_norm = N * 3 // 4
N_def  = N - N_norm

X_norm = make_normal(N_norm)
X_t    = make_defect_thickness(N_def // 3)
X_m    = make_defect_mass(N_def // 3)
X_th   = make_defect_thermal(N_def - 2*(N_def//3))

X = np.vstack([X_norm, X_t, X_m, X_th])
y = np.array([0]*N_norm + [1]*(N_def))

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + 1e-8
X_norm_all = ((X - X_mean) / X_std).astype(np.float32)

split   = int(0.8 * len(X))
X_train = torch.tensor(X_norm_all[:split]).to(DEVICE)
Y_train = torch.tensor(y[:split], dtype=torch.long).to(DEVICE)
X_test  = torch.tensor(X_norm_all[split:]).to(DEVICE)
Y_test  = torch.tensor(y[split:], dtype=torch.long).to(DEVICE)

print(f"Деталей: {len(X)} | Норма: {(y==0).sum()} | Брак: {(y==1).sum()}")

# ── Модель ────────────────────────────────────────────────────────────────────
class DefectNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),  nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x): return self.net(x)

model     = DefectNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Взвешенный loss (класс брака реже)
w = torch.tensor([1.0, 3.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=w)

# ── Обучение ──────────────────────────────────────────────────────────────────
EPOCHS = 100
print(f"\nОбучение: {EPOCHS} эпох...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = torch.randperm(split)
    for i in range(0, split, 128):
        xb = X_train[idx[i:i+128]]
        yb = Y_train[idx[i:i+128]]
        loss = criterion(model(xb), yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    if epoch % 25 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(X_test).argmax(1) == Y_test).float().mean().item()
        print(f"  Эпоха {epoch}/{EPOCHS} | Acc: {acc*100:.1f}%")

# ── Оценка ────────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    probs = torch.softmax(model(X_test), dim=1).cpu().numpy()
    preds = probs.argmax(1)
    y_t   = Y_test.cpu().numpy()

TP = ((preds==1)&(y_t==1)).sum()
FP = ((preds==1)&(y_t==0)).sum()
FN = ((preds==0)&(y_t==1)).sum()
TN = ((preds==0)&(y_t==0)).sum()
acc  = (TP+TN)/len(y_t)
prec = TP/(TP+FP+1e-8)
rec  = TP/(TP+FN+1e-8)
print(f"\nТочность: {acc*100:.1f}%  Precision: {prec:.3f}  Recall: {rec:.3f}")
print(f"Выявлено брака: {TP}/{(y_t==1).sum()} ({TP/(y_t==1).sum()*100:.0f}%)")

# ── Сохранение ────────────────────────────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "X_mean":      X_mean.tolist(),
    "X_std":       X_std.tolist(),
    "feature_names": ["Толщина(мм)", "Масса(г)", "Твёрдость(HB)", "Шероховатость(Ra)",
                      "Длина(мм)", "Ширина(мм)", "Температура(°C)", "Время(с)"],
    "limits": {
        "thickness": [9.85, 10.15], "mass": [230, 270], "hardness": [160, 240],
        "roughness": [0.8, 2.4],    "length": [99.7, 100.3], "width": [49.76, 50.24],
        "temp": [750, 950],         "time": [100, 140],
    }
}, BASE / "defect_model.pth")

print("\nМодель сохранена: defect_model.pth")
