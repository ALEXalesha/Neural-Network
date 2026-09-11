"""
Кластеризация — сегментация клиентов интернет-магазина
Задача: разбить покупателей на группы БЕЗ заранее заданных меток

Метод: Deep Clustering = Autoencoder + K-Means
  1. Autoencoder сжимает данные в 2D (чтобы похожие клиенты оказались рядом)
  2. K-Means разбивает 2D-пространство на кластеры

Отличие от классификации: у нас нет правильных ответов — сеть сама находит группы
"""
import sys, io
if hasattr(sys.stdout, "buffer"): sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}\n")

# ─────────────────────────────────────────
# 1. ДАННЫЕ — профили покупателей
# ─────────────────────────────────────────
# 4 типа клиентов (но модель этого не знает — ей надо самой найти):
# 1. VIP       — богатые, редко но много тратят
# 2. Активные  — средний чек, часто покупают
# 3. Молодые   — много просматривают, мало покупают
# 4. Пассивные — давно не появлялись, мало потратили

def gen_segment(n, age_m, age_s, orders_m, orders_s, avg_m, avg_s,
                total_m, total_s, days_m, days_s, visits_m, visits_s):
    return np.column_stack([
        np.random.normal(age_m, age_s, n),          # возраст
        np.random.normal(orders_m, orders_s, n),     # кол-во заказов
        np.random.normal(avg_m, avg_s, n),           # средний чек (руб)
        np.random.normal(total_m, total_s, n),       # сумма всего (тыс руб)
        np.random.normal(days_m, days_s, n),         # дней с последней покупки
        np.random.normal(visits_m, visits_s, n),     # визитов в месяц
    ]).astype(np.float32)

N = 2000
X_vip      = gen_segment(N//4, 42, 8,   8, 3,  15000, 3000, 120, 40,  10, 5,  5, 2)
X_active   = gen_segment(N//4, 32, 6,  25, 5,   3000,  500,  75, 20,  7,  3, 20, 5)
X_young    = gen_segment(N//4, 23, 4,   3, 2,   1500,  400,   5,  3, 20, 10, 40, 10)
X_passive  = gen_segment(N//4, 50, 10,  2, 1,   2000,  600,   4,  2, 90, 30,  2, 1)

X = np.vstack([X_vip, X_active, X_young, X_passive])
true_labels = np.array([0]*500 + [1]*500 + [2]*500 + [3]*500)  # для проверки

# Clip отрицательные значения
X = np.clip(X, 0, None)

print(f"Клиентов: {len(X)}")
print(f"Признаков: 6 (возраст, заказы, средний чек, сумма, дней с покупки, визиты)\n")

# Нормализация
scaler = StandardScaler()
X_norm = scaler.fit_transform(X).astype(np.float32)
X_tensor = torch.tensor(X_norm).to(device)

# ─────────────────────────────────────────
# 2. AUTOENCODER — сжимаем 6D → 2D
# ─────────────────────────────────────────
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

model = ClusterAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ─────────────────────────────────────────
# 3. ОБУЧЕНИЕ
# ─────────────────────────────────────────
print("Обучение autoencoder...")
EPOCHS = 300
losses = []

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

    losses.append(epoch_loss)
    if epoch % 60 == 0:
        print(f"  Эпоха {epoch}/{EPOCHS} | Loss: {epoch_loss//(N//128):.4f}")

# ─────────────────────────────────────────
# 4. КЛАСТЕРИЗАЦИЯ K-MEANS
# ─────────────────────────────────────────
model.eval()
with torch.no_grad():
    _, latent = model(X_tensor)
    latent_np = latent.cpu().numpy()

# K-Means в 2D латентном пространстве
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(latent_np)

# Сопоставляем кластеры с реальными сегментами (по совпадению)
from scipy.stats import mode

def match_clusters(pred, true, k):
    mapping = {}
    for c in range(k):
        mask = pred == c
        if mask.sum() > 0:
            m = mode(true[mask], keepdims=True).mode[0]
            mapping[c] = m
    return mapping

mapping = match_clusters(cluster_labels, true_labels, K)
seg_names = {0: "VIP", 1: "Активные", 2: "Молодые", 3: "Пассивные"}
cluster_names = {c: seg_names.get(mapping[c], f"Группа {c}") for c in range(K)}

# Точность кластеризации
correct = sum(mapping[cluster_labels[i]] == true_labels[i] for i in range(N))
accuracy = correct / N

print(f"\n{'='*45}")
print(f"Точность кластеризации: {accuracy*100:.1f}%\n")

# Профили кластеров
feature_names = ["Возраст", "Заказов", "Ср.чек(руб)", "Сумма(тыс)", "Дней назад", "Визитов"]
print(f"{'Кластер':<12}", end="")
for f in feature_names:
    print(f"{f:>12}", end="")
print()
print("-" * 85)

for c in range(K):
    mask = cluster_labels == c
    means = X[mask].mean(axis=0)
    name = cluster_names[c]
    print(f"{name:<12}", end="")
    for v in means:
        print(f"{v:>12.1f}", end="")
    print(f"  ({mask.sum()} чел.)")

# ─────────────────────────────────────────
# 5. ИНТЕРАКТИВ — к какому сегменту относится клиент
# ─────────────────────────────────────────
print(f"\n{'='*45}")
print("Определить сегмент нового клиента (Enter = выход):\n")

while True:
    try:
        age     = float(input("  Возраст: "))
        orders  = float(input("  Кол-во заказов: "))
        avg_chk = float(input("  Средний чек (руб): "))
        total   = float(input("  Сумма покупок (тыс руб): "))
        days    = float(input("  Дней с последней покупки: "))
        visits  = float(input("  Визитов в месяц: "))

        feat = np.array([age, orders, avg_chk, total, days, visits], dtype=np.float32)
        feat_norm = scaler.transform(feat.reshape(1, -1)).astype(np.float32)
        inp = torch.tensor(feat_norm).to(device)

        with torch.no_grad():
            _, z = model(inp)
            z_np = z.cpu().numpy()

        c = kmeans.predict(z_np)[0]
        name = cluster_names[c]
        print(f"\n  Сегмент клиента: {name}\n")

        recs = {
            "VIP":       "Персональный менеджер, эксклюзивные предложения",
            "Активные":  "Программа лояльности, скидки за частые покупки",
            "Молодые":   "Акции, геймификация, соцсети",
            "Пассивные": "Реактивационное письмо, скидка 20%",
        }
        print(f"  Рекомендация: {recs.get(name, '—')}\n")

    except (ValueError, EOFError, KeyboardInterrupt):
        print("\nГотово!")
        break

# Графики
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['gold', 'green', 'blue', 'gray']
seg_labels = ["VIP", "Активные", "Молодые", "Пассивные"]

# Реальные сегменты
for i, (label, color) in enumerate(zip(seg_labels, colors)):
    mask = true_labels == i
    axes[0].scatter(latent_np[mask, 0], latent_np[mask, 1],
                    c=color, alpha=0.4, s=15, label=label)
axes[0].set_title("Реальные сегменты (для проверки)")
axes[0].legend(); axes[0].grid(True)

# Найденные кластеры
cluster_colors = [colors[mapping.get(c, c)] for c in cluster_labels]
axes[1].scatter(latent_np[:, 0], latent_np[:, 1],
                c=cluster_colors, alpha=0.4, s=15)
centers = kmeans.cluster_centers_
axes[1].scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', zorder=5)
for c in range(K):
    axes[1].annotate(cluster_names[c], centers[c], fontsize=10, ha='center')
axes[1].set_title(f"Найденные кластеры (точность {accuracy*100:.1f}%)")
axes[1].grid(True)

plt.tight_layout()
plt.savefig('clustering_results.png', dpi=100)
print("График сохранён: clustering_results.png")
