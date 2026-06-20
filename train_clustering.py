"""
Обучение модели кластеризации (Autoencoder + K-Means)
Сохраняет clustering_model.pth с весами, центрами кластеров и параметрами scaler
"""
import sys, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

BASE   = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {DEVICE}\n")

# ── Данные — профили 4 типов покупателей ─────────────────────────────────────
def gen_segment(n, age_m, age_s, orders_m, orders_s, avg_m, avg_s,
                total_m, total_s, days_m, days_s, visits_m, visits_s):
    return np.column_stack([
        np.random.normal(age_m,    age_s,    n),
        np.random.normal(orders_m, orders_s, n),
        np.random.normal(avg_m,    avg_s,    n),
        np.random.normal(total_m,  total_s,  n),
        np.random.normal(days_m,   days_s,   n),
        np.random.normal(visits_m, visits_s, n),
    ]).astype(np.float32)

N = 2000
X_vip     = gen_segment(N//4, 42, 8,   8, 3,  15000, 3000, 120, 40,  10, 5,  5, 2)
X_active  = gen_segment(N//4, 32, 6,  25, 5,   3000,  500,  75, 20,   7, 3, 20, 5)
X_young   = gen_segment(N//4, 23, 4,   3, 2,   1500,  400,   5,  3,  20,10, 40,10)
X_passive = gen_segment(N//4, 50,10,   2, 1,   2000,  600,   4,  2,  90,30,  2, 1)

X = np.clip(np.vstack([X_vip, X_active, X_young, X_passive]), 0, None)
true_labels = np.array([0]*500 + [1]*500 + [2]*500 + [3]*500)

scaler = StandardScaler()
X_norm   = scaler.fit_transform(X).astype(np.float32)
X_tensor = torch.tensor(X_norm).to(DEVICE)

print(f"Клиентов: {N} | Признаков: 6")

# ── Autoencoder 6D → 2D ───────────────────────────────────────────────────────
class ClusterAutoencoder(nn.Module):
    def __init__(self, input_dim=6, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32),        nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64),         nn.ReLU(),
            nn.Linear(64, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

model     = ClusterAutoencoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ── Обучение ──────────────────────────────────────────────────────────────────
EPOCHS = 300
print(f"\nОбучение Autoencoder: {EPOCHS} эпох...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = torch.randperm(N)
    epoch_loss = 0
    for i in range(0, N, 128):
        xb = X_tensor[idx[i:i+128]]
        reconstructed, _ = model(xb)
        loss = criterion(reconstructed, xb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        epoch_loss += loss.item()
    if epoch % 100 == 0:
        print(f"  Эпоха {epoch}/{EPOCHS} | Loss: {epoch_loss:.4f}")

# ── K-Means в латентном пространстве ─────────────────────────────────────────
model.eval()
with torch.no_grad():
    _, latent = model(X_tensor)
    latent_np = latent.cpu().numpy()

K      = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(latent_np)

# Сопоставляем кластеры с реальными сегментами
from scipy.stats import mode
mapping = {}
for c in range(K):
    mask = labels == c
    if mask.sum() > 0:
        mapping[c] = int(mode(true_labels[mask], keepdims=True).mode[0])

seg_names = {0: "VIP", 1: "Активные", 2: "Молодые", 3: "Пассивные"}
correct   = sum(mapping[labels[i]] == true_labels[i] for i in range(N))
accuracy  = correct / N
print(f"\nТочность кластеризации: {accuracy*100:.1f}%\n")

# Профили кластеров
feature_names = ["Возраст", "Заказов", "Ср.чек(руб)", "Сумма(тыс)", "Дней назад", "Визитов"]
print(f"{'Сегмент':<12}", end="")
for f in feature_names: print(f"{f:>13}", end="")
print()
print("-"*90)
for c in range(K):
    mask = labels == c
    means = X[mask].mean(axis=0)
    name  = seg_names.get(mapping.get(c, c), f"Группа {c}")
    print(f"{name:<12}", end="")
    for v in means: print(f"{v:>13.1f}", end="")
    print(f"  ({mask.sum()} чел.)")

# ── Сохранение ────────────────────────────────────────────────────────────────
torch.save({
    "model_state":    model.state_dict(),
    "kmeans_centers": kmeans.cluster_centers_.tolist(),
    "cluster_mapping":mapping,                          # {cluster_id: segment_id}
    "scaler_mean":    scaler.mean_.tolist(),
    "scaler_scale":   scaler.scale_.tolist(),
    "K":              K,
}, BASE / "clustering_model.pth")

print("\nМодель сохранена: clustering_model.pth")
