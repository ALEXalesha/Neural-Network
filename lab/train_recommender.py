"""
Рекомендательная система (Neural Collaborative Filtering)
120 фильмов × 6 жанров, 500 пользователей
Inference: вводишь понравившиеся фильмы → получаешь похожие
Сохраняет recommender_model.pth
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

# ── Каталог фильмов ────────────────────────────────────────────────────────────
# Жанры: Action, Comedy, Drama, Sci-Fi, Romance, Thriller
GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller"]

MOVIES = [
    # Action
    ("Inception Strike",    [1,0,0,0,0,1]),
    ("Dark Pursuit",        [1,0,0,0,0,1]),
    ("Thunder Force",       [1,0,0,0,0,0]),
    ("Iron Protocol",       [1,0,0,1,0,0]),
    ("Shadowfall",          [1,0,1,0,0,0]),
    ("Desert Eagle",        [1,0,0,0,0,1]),
    ("Blackout Mission",    [1,0,0,0,0,0]),
    ("Rapid Fire",          [1,0,0,0,0,0]),
    ("Final Strike",        [1,0,0,0,0,1]),
    ("Ghost Protocol",      [1,0,0,0,0,1]),
    ("Zero Hour",           [1,0,0,0,0,1]),
    ("Steel Rising",        [1,0,0,1,0,0]),
    ("The Last Stand",      [1,0,1,0,0,0]),
    ("Night Hawk",          [1,0,0,0,0,1]),
    ("Neon Runner",         [1,0,0,1,0,0]),
    ("Combat Zone",         [1,0,0,0,0,0]),
    ("Red Horizon",         [1,0,1,0,0,0]),
    ("Iron Storm",          [1,0,0,0,0,0]),
    ("Broken Arrow",        [1,0,0,0,0,1]),
    ("Force Delta",         [1,0,0,0,0,0]),
    # Comedy
    ("Weekend Chaos",       [0,1,0,0,0,0]),
    ("Office Mania",        [0,1,0,0,0,0]),
    ("Perfect Disaster",    [0,1,0,0,1,0]),
    ("My Crazy Family",     [0,1,1,0,0,0]),
    ("The Big Surprise",    [0,1,0,0,0,0]),
    ("Panic at the Party",  [0,1,0,0,0,0]),
    ("Wedding Crasher",     [0,1,0,0,1,0]),
    ("Totally Awkward",     [0,1,0,0,0,0]),
    ("Boss Day",            [0,1,0,0,0,0]),
    ("Summer Fools",        [0,1,0,0,0,0]),
    ("Accidental Hero",     [0,1,1,0,0,0]),
    ("The Wrong Date",      [0,1,0,0,1,0]),
    ("Lunch Break",         [0,1,0,0,0,0]),
    ("Road Trip Fail",      [0,1,0,0,0,0]),
    ("Game Night",          [0,1,0,0,0,0]),
    ("Surprise Package",    [0,1,0,0,0,0]),
    ("Neighbors War",       [0,1,0,0,0,0]),
    ("Selfie Squad",        [0,1,0,0,0,0]),
    ("Late Show",           [0,1,1,0,0,0]),
    ("Midnight Mischief",   [0,1,0,0,1,0]),
    # Drama
    ("Broken Wings",        [0,0,1,0,0,0]),
    ("Silent Storm",        [0,0,1,0,0,0]),
    ("The Last Letter",     [0,0,1,0,1,0]),
    ("All She Left",        [0,0,1,0,0,0]),
    ("Winter Garden",       [0,0,1,0,1,0]),
    ("Paper Hearts",        [0,0,1,0,1,0]),
    ("The Bridge",          [0,0,1,0,0,0]),
    ("Another Day",         [0,0,1,0,0,0]),
    ("Fading Light",        [0,0,1,0,0,0]),
    ("Long Road Home",      [0,0,1,0,0,0]),
    ("Second Chapter",      [0,0,1,0,0,0]),
    ("Lost in Translation", [0,0,1,0,1,0]),
    ("The Quiet Storm",     [0,0,1,0,0,0]),
    ("Ordinary People",     [0,0,1,0,0,0]),
    ("Between the Lines",   [0,0,1,0,0,0]),
    ("After the Rain",      [0,0,1,0,1,0]),
    ("Untold Story",        [0,0,1,0,0,0]),
    ("Crossroads",          [0,0,1,0,0,0]),
    ("The Promise",         [0,0,1,0,1,0]),
    ("One Last Chance",     [0,0,1,0,0,0]),
    # Sci-Fi
    ("Quantum Dawn",        [0,0,0,1,0,0]),
    ("Mars Protocol",       [0,0,0,1,0,0]),
    ("Neural Echo",         [0,0,0,1,0,0]),
    ("Galaxy's Edge",       [0,0,0,1,0,0]),
    ("The Singularity",     [0,0,1,1,0,0]),
    ("Cyber Horizon",       [1,0,0,1,0,0]),
    ("Deep Space Nine",     [0,0,1,1,0,0]),
    ("Project Omega",       [0,0,0,1,0,1]),
    ("Parallel Code",       [0,0,0,1,0,0]),
    ("Void Walkers",        [1,0,0,1,0,0]),
    ("The Algorithm",       [0,0,1,1,0,0]),
    ("Nano Storm",          [1,0,0,1,0,0]),
    ("Orbital Decay",       [0,0,0,1,0,0]),
    ("Last Transmission",   [0,0,1,1,0,0]),
    ("Binary Sunset",       [0,0,1,1,0,0]),
    ("Xenobot",             [1,0,0,1,0,0]),
    ("Gravity Well",        [0,0,0,1,0,0]),
    ("Future Shock",        [0,0,0,1,0,1]),
    ("Warp Point",          [1,0,0,1,0,0]),
    ("Signal Lost",         [0,0,1,1,0,0]),
    # Romance
    ("Paris in Rain",       [0,0,1,0,1,0]),
    ("Second Chance",       [0,0,1,0,1,0]),
    ("Forever After",       [0,0,0,0,1,0]),
    ("Letters Never Sent",  [0,0,1,0,1,0]),
    ("Summer Kiss",         [0,1,0,0,1,0]),
    ("The Way You Smile",   [0,0,1,0,1,0]),
    ("Falling Again",       [0,0,0,0,1,0]),
    ("Midnight Blue",       [0,0,1,0,1,0]),
    ("One Perfect Day",     [0,0,0,0,1,0]),
    ("Find Me Here",        [0,0,1,0,1,0]),
    ("Always Yours",        [0,0,0,0,1,0]),
    ("Coffee & Goodbyes",   [0,0,1,0,1,0]),
    ("Something Real",      [0,0,1,0,1,0]),
    ("Just One Look",       [0,1,0,0,1,0]),
    ("Heart on Sleeve",     [0,0,1,0,1,0]),
    # Thriller
    ("Cold Trail",          [0,0,0,0,0,1]),
    ("Midnight Witness",    [0,0,0,0,0,1]),
    ("The Setup",           [0,0,1,0,0,1]),
    ("Dark Signal",         [0,0,0,0,0,1]),
    ("Under Pressure",      [1,0,0,0,0,1]),
    ("The Informant",       [0,0,1,0,0,1]),
    ("Vanishing Point",     [0,0,0,0,0,1]),
    ("Buried Evidence",     [0,0,0,0,0,1]),
    ("Mind Games",          [0,0,1,0,0,1]),
    ("The Last Alibi",      [0,0,0,0,0,1]),
    ("Smoke Screen",        [0,0,1,0,0,1]),
    ("Wired",               [1,0,0,0,0,1]),
    ("Double Cross",        [0,0,0,0,0,1]),
    ("Suspicion",           [0,0,1,0,0,1]),
    ("The Witness",         [0,0,1,0,0,1]),
    ("Exposed",             [0,0,0,0,0,1]),
    ("Shadow Jury",         [0,0,1,0,0,1]),
    ("The Mole",            [1,0,0,0,0,1]),
    ("Deadlock",            [1,0,0,0,0,1]),
    ("Fault Line",          [0,0,1,0,0,1]),
]

MOVIE_NAMES   = [m[0] for m in MOVIES]
MOVIE_GENRES  = np.array([m[1] for m in MOVIES], dtype=np.float32)
N_MOVIES      = len(MOVIES)
N_USERS       = 500
EMBED_DIM     = 32

print(f"Фильмов: {N_MOVIES} | Пользователей: {N_USERS} | Жанров: {len(GENRES)}")

# ── Генерация синтетических оценок ─────────────────────────────────────────────
# Каждый пользователь — вектор предпочтений по жанрам
user_prefs = np.zeros((N_USERS, len(GENRES)), dtype=np.float32)
for i in range(N_USERS):
    # Предпочитает 1-2 жанра
    fav = np.random.choice(len(GENRES), size=np.random.randint(1, 3), replace=False)
    user_prefs[i, fav] = np.random.uniform(0.7, 1.0, len(fav))
    # Лёгкий интерес к остальным
    for j in range(len(GENRES)):
        if user_prefs[i, j] == 0:
            user_prefs[i, j] = np.random.uniform(0, 0.2)

# Рейтинг = соответствие жанров + шум → clamp 1..5
interactions = []
for u in range(N_USERS):
    for m in range(N_MOVIES):
        score = np.dot(user_prefs[u], MOVIE_GENRES[m])
        rating = score * 4 + 1 + np.random.normal(0, 0.3)
        rating = float(np.clip(rating, 1.0, 5.0))
        if np.random.random() < 0.4:   # только часть оценок наблюдаема
            interactions.append((u, m, rating))

interactions = np.array(interactions, dtype=np.float32)
np.random.shuffle(interactions)

split = int(0.85 * len(interactions))
train_data = interactions[:split]
test_data  = interactions[split:]
print(f"Оценок: {len(interactions)} | Обучение: {len(train_data)} | Тест: {len(test_data)}")

# ── Модель ────────────────────────────────────────────────────────────────────
class RecommenderNCF(nn.Module):
    def __init__(self, n_users, n_movies, embed_dim=EMBED_DIM):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users,  embed_dim)
        self.movie_emb = nn.Embedding(n_movies, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,  32),            nn.ReLU(),
            nn.Linear(32,  1),
        )
        nn.init.normal_(self.user_emb.weight,  std=0.1)
        nn.init.normal_(self.movie_emb.weight, std=0.1)

    def forward(self, u, m):
        ue = self.user_emb(u)
        me = self.movie_emb(m)
        return self.mlp(torch.cat([ue, me], dim=1)).squeeze(1)

model     = RecommenderNCF(N_USERS, N_MOVIES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

# ── Обучение ──────────────────────────────────────────────────────────────────
EPOCHS     = 80
BATCH_SIZE = 512
print(f"\nОбучение: {EPOCHS} эпох...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx  = np.random.permutation(len(train_data))
    loss_sum = 0
    steps = 0
    for i in range(0, len(train_data), BATCH_SIZE):
        batch = train_data[idx[i:i+BATCH_SIZE]]
        u = torch.tensor(batch[:, 0].astype(int)).to(DEVICE)
        m = torch.tensor(batch[:, 1].astype(int)).to(DEVICE)
        r = torch.tensor(batch[:, 2]).to(DEVICE)
        pred = model(u, m)
        loss = criterion(pred, r)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_sum += loss.item(); steps += 1

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            tu = torch.tensor(test_data[:, 0].astype(int)).to(DEVICE)
            tm = torch.tensor(test_data[:, 1].astype(int)).to(DEVICE)
            tr = torch.tensor(test_data[:, 2]).to(DEVICE)
            mse = criterion(model(tu, tm), tr).item()
        print(f"  Эпоха {epoch}/{EPOCHS} | Train loss: {loss_sum/steps:.4f} | Test MSE: {mse:.4f} | RMSE: {mse**0.5:.3f}")

# ── Извлекаем эмбеддинги фильмов для inference ──────────────────────────────
model.eval()
with torch.no_grad():
    all_ids    = torch.arange(N_MOVIES).to(DEVICE)
    movie_vecs = model.movie_emb(all_ids).cpu().numpy()  # [N_MOVIES, EMBED_DIM]

# ── Демо-рекомендации ─────────────────────────────────────────────────────────
def recommend_by_likes(liked_indices, top_n=5):
    mean_vec = movie_vecs[liked_indices].mean(axis=0)
    scores   = movie_vecs @ mean_vec  # dot product
    scores[liked_indices] = -np.inf   # исключаем уже понравившиеся
    top = np.argsort(scores)[::-1][:top_n]
    return [(MOVIE_NAMES[i], float(scores[i])) for i in top]

print("\nПример (понравились экшен-фильмы):")
likes = [0, 1, 2]  # Inception Strike, Dark Pursuit, Thunder Force
for name, score in recommend_by_likes(likes):
    print(f"  → {name}  (score: {score:.3f})")

print("\nПример (понравились романтика + драма):")
likes2 = [80, 81, 45, 46]  # Paris in Rain, Second Chance, Winter Garden, Paper Hearts
for name, score in recommend_by_likes(likes2):
    print(f"  → {name}  (score: {score:.3f})")

# ── Жанры каждого фильма в читаемом виде ──────────────────────────────────────
movie_genre_strs = []
for _, gvec in MOVIES:
    gs = [GENRES[i] for i, v in enumerate(gvec) if v]
    movie_genre_strs.append(", ".join(gs))

# ── Сохранение ────────────────────────────────────────────────────────────────
torch.save({
    "model_state":    model.state_dict(),
    "movie_vecs":     movie_vecs.tolist(),
    "movie_names":    MOVIE_NAMES,
    "movie_genres":   MOVIE_GENRES.tolist(),
    "movie_genre_str":movie_genre_strs,
    "genres":         GENRES,
    "n_users":        N_USERS,
    "n_movies":       N_MOVIES,
    "embed_dim":      EMBED_DIM,
}, BASE / "recommender_model.pth")

print("\nМодель сохранена: recommender_model.pth")
