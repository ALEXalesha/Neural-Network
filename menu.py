"""
Главное меню — нейронная лаборатория v2.0
Объединяет все обученные модели в одном приложении
"""
import sys, io
if hasattr(sys.stdout, "buffer"): sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os, sys, subprocess
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════════
# ЗАГРУЗКА МОДЕЛЕЙ
# ══════════════════════════════════════════════════════

def load_model(path, model):
    p = BASE / path
    if p.exists():
        model.load_state_dict(torch.load(p, weights_only=True, map_location=device))
        model.eval()
        return model
    return None

# --- Math Solver ---
CHARS_M = "0123456789+-*/=<>^~"
c2i_m = {c: i for i, c in enumerate(CHARS_M)}
i2c_m = {i: c for i, c in enumerate(CHARS_M)}
PAD_M, END_M, SOS_M, NEG_M = '<', '>', '^', '~'
VOCAB_M, MAX_L = len(CHARS_M), 16

class MEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_M, 64, padding_idx=c2i_m[PAD_M])
        self.lstm  = nn.LSTM(64, 256, 2, batch_first=True, dropout=0.1)
    def forward(self, x): return self.lstm(self.embed(x))

class MAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(512, 256); self.v = nn.Linear(256, 1, bias=False)
    def forward(self, h, enc):
        T = enc.size(1); h = h.unsqueeze(1).repeat(1,T,1)
        return torch.softmax(self.v(torch.tanh(self.attn(torch.cat([h,enc],2)))).squeeze(2), 1)

class MDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_M, 64, padding_idx=c2i_m[PAD_M])
        self.attn  = MAttention()
        self.lstm  = nn.LSTM(64+256, 256, 2, batch_first=True, dropout=0.1)
        self.fc    = nn.Linear(512, VOCAB_M)
    def forward(self, tok, hid, enc):
        e = self.embed(tok.unsqueeze(1)); h = hid[0][-1]
        a = self.attn(h, enc); ctx = (a.unsqueeze(1) @ enc)
        out, hid = self.lstm(torch.cat([e,ctx],2), hid)
        return self.fc(torch.cat([out.squeeze(1), ctx.squeeze(1)], 1)), hid

class MathSeq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MEncoder(); self.decoder = MDecoder()

def encode_math(s):
    ids = [c2i_m.get(c,0) for c in s]
    ids += [c2i_m[PAD_M]] * (MAX_L - len(ids))
    return ids[:MAX_L]

# --- Regression (Price) ---
class PriceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128,1))
    def forward(self, x): return self.net(x)

# --- Temperature ---
class TempNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6,64), nn.ReLU(), nn.Linear(64,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x): return self.net(x)

# --- Anomaly (Autoencoder) ---
class AnomalyAE(nn.Module):
    def __init__(self, input_dim=7, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim,16), nn.ReLU(), nn.Linear(16,32), nn.ReLU(), nn.Linear(32,input_dim))
    def forward(self, x): return self.decoder(self.encoder(x))

# --- Sentiment ---
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.lstm  = nn.LSTM(64, 128, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.drop  = nn.Dropout(0.4)
        self.fc    = nn.Linear(256, 3)
    def forward(self, x):
        out, (h, _) = self.lstm(self.drop(self.embed(x)))
        return self.fc(self.drop(torch.cat([h[-2], h[-1]], 1)))

# --- TimeSeries ---
class TSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 128, 2, batch_first=True, dropout=0.2)
        self.attn = nn.Linear(128, 1)
        self.fc   = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,7))
    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), 1)
        return self.fc((w * out).sum(1))

# --- GAN Generator ---
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.net = nn.Sequential(
            nn.Linear(100+10, 7*7*256), nn.Unflatten(1, (256,7,7)),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1),  nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64,1,3,1,1), nn.Tanh())
    def forward(self, z, labels=None):
        if labels is None: labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        return self.net(torch.cat([z, self.label_emb(labels)], dim=1))

# --- Spam LSTM ---
class SpamLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.drop  = nn.Dropout(0.4)
        self.fc    = nn.Linear(hidden_dim*2, 2)
    def forward(self, x):
        _, (h, _) = self.lstm(self.embed(x))
        return self.fc(self.drop(torch.cat([h[-2], h[-1]], dim=-1)))

# --- Defect Net ---
class DefectNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8,64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,64), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64,2))
    def forward(self, x): return self.net(x)

# --- Cluster Autoencoder ---
class ClusterAutoencoder(nn.Module):
    def __init__(self, input_dim=6, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64,32),        nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32,latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,32), nn.ReLU(),
            nn.Linear(32,64),         nn.ReLU(),
            nn.Linear(64,input_dim))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# --- NER Bi-LSTM ---
class NERModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_tags=9):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(hidden_dim*2, num_tags)
    def forward(self, x):
        out, _ = self.lstm(self.embed(x))
        return self.fc(self.drop(out))

# --- Recommender NCF ---
class RecommenderNCF(nn.Module):
    def __init__(self, n_users, n_movies, embed_dim=32):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users,  embed_dim)
        self.movie_emb = nn.Embedding(n_movies, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim*2, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),          nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,  32),          nn.ReLU(),
            nn.Linear(32,  1))
    def forward(self, u, m):
        return self.mlp(torch.cat([self.user_emb(u), self.movie_emb(m)], dim=1)).squeeze(1)

# ══════════════════════════════════════════════════════
# МЕНЮ
# ══════════════════════════════════════════════════════

def clear(): os.system('cls' if os.name == 'nt' else 'clear')

def header():
    print("=" * 58)
    print("       НЕЙРОННАЯ ЛАБОРАТОРИЯ  v2.0")
    print(f"       Устройство: {device}")
    print("=" * 58)

def check(path):
    return "[OK]" if (BASE / path).exists() else "[--]"

def main_menu():
    while True:
        clear(); header()
        print(f"""
  ── РЕГРЕССИЯ / ПРОГНОЗ ────────────────────────────
  {check('math_model.pth')} 1.  Решить пример          (арифметика)
  {check('price_model.pth')} 2.  Цена квартиры          (регрессия)
  {check('temperature_model.pth')} 3.  Прогноз температуры    (регрессия)
  {check('timeseries_model.pth')} 4.  Прогноз продаж         (временной ряд)

  ── КЛАССИФИКАЦИЯ / РАСПОЗНАВАНИЕ ──────────────────
  {check('sentiment_model.pth')} 5.  Тональность текста     (LSTM)
  {check('spam_model.pth')} 6.  Детектор спама         (LSTM)
  {check('ner_model.pth')} 7.  Сущности в тексте      (NER Bi-LSTM)
  {check('best_model_v3.pth')} 8.  Распознавание цифры    (CNN)

  ── АНОМАЛИИ / КОНТРОЛЬ КАЧЕСТВА ───────────────────
  {check('anomaly_model.pth')} 9.  Детектор мошенничества (Autoencoder)
  {check('defect_model.pth')} 10. Контроль качества      (дефекты деталей)

  ── КЛАСТЕРИЗАЦИЯ / РЕКОМЕНДАЦИИ ───────────────────
  {check('clustering_model.pth')} 11. Сегментация клиента   (Autoencoder+KMeans)
  {check('recommender_model.pth')} 12. Рекомендации фильмов  (Neural CF)

  ── ГЕНЕРАЦИЯ ──────────────────────────────────────
  {check('gan_generator.pth')} 13. Генерация цифр         (GAN)

     s.  Smart Assistant          (LM Studio)
     r.  Обучить все модели заново
     0.  Выход
""")
        choice = input("  Выбор: ").strip()
        if   choice == '1':  menu_math()
        elif choice == '2':  menu_price()
        elif choice == '3':  menu_temp()
        elif choice == '4':  menu_timeseries()
        elif choice == '5':  menu_sentiment()
        elif choice == '6':  menu_spam()
        elif choice == '7':  menu_ner()
        elif choice == '8':  menu_mnist()
        elif choice == '9':  menu_anomaly()
        elif choice == '10': menu_defect()
        elif choice == '11': menu_clustering()
        elif choice == '12': menu_recommender()
        elif choice == '13': menu_gan()
        elif choice in ('s', 'S'): menu_smart()
        elif choice in ('r', 'R'): retrain_all()
        elif choice == '0': print("\nПока!"); break

# ── 1. МАТЕМАТИКА ─────────────────────────────────────
def menu_math():
    clear(); header(); print("  [1] РЕШЕНИЕ ПРИМЕРОВ\n")
    model = MathSeq2Seq().to(device)
    m = load_model('math_model.pth', model)
    if not m:
        print("  Модель не найдена. Запустите math_solver.py\n")
        input("  Enter..."); return

    print("  Поддерживаемые операции: + - *")
    print("  Формат: 123+456 или 99*99 или 500-750")
    print("  (Enter = выход)\n")
    while True:
        expr = input("  Пример: ").strip()
        if not expr: break
        import re
        match = re.match(r'^(\d+)([+\-*])(\d+)$', expr.replace(' ',''))
        if not match:
            print("  Формат: число операция число (напр. 123+456)\n"); continue
        question = expr.replace(' ','') + '='
        src = torch.tensor([encode_math(question)], dtype=torch.long).to(device)
        with torch.no_grad():
            enc_out, hidden = m.encoder(src)
            tok = torch.tensor([c2i_m[SOS_M]], device=device)
            res = []
            for _ in range(MAX_L):
                pred, hidden = m.decoder(tok, hidden, enc_out)
                tok = pred.argmax(1)
                ch = i2c_m[tok.item()]
                if ch == END_M: break
                if ch != PAD_M: res.append(ch)
        raw = ''.join(res)
        answer = ('-' + raw[:-1]) if raw.endswith(NEG_M) else raw
        correct = str(eval(question.replace('=','')))
        ok = "верно" if answer == correct else f"ожидалось: {correct}"
        print(f"  {question} = {answer}  ({ok})\n")
    input("\n  Enter для возврата в меню...")

# ── 2. ЦЕНА КВАРТИРЫ ──────────────────────────────────
def menu_price():
    clear(); header(); print("  [2] ЦЕНА КВАРТИРЫ\n")
    np.random.seed(42)
    N = 10000
    area     = np.random.uniform(25, 150, N)
    rooms    = np.random.randint(1, 6, N).astype(float)
    floor    = np.random.randint(1, 26, N).astype(float)
    district = np.random.randint(0, 5, N).astype(float)
    age      = np.random.uniform(0, 60, N)
    ppm2     = np.array([120,160,200,260,350])
    base     = area * ppm2[district.astype(int)]
    fb       = np.where(floor==1,-5000,0) + np.where(floor>=15,3000,0)
    price    = (base + fb + rooms*8000 - age*500 + np.random.normal(0,15000,N)) / 1e6
    X = np.stack([area,rooms,floor,district,age],1).astype(np.float32)
    Y = price.astype(np.float32).reshape(-1,1)
    X_mean, X_std = X.mean(0), X.std(0)
    Y_mean, Y_std = float(Y.mean()), float(Y.std())

    model = load_model('price_model.pth', PriceNet().to(device))
    if not model:
        print("  Модель не найдена. Запустите regression.py\n")
        input("  Enter..."); return

    dnames = {"окраина":0,"спальный":1,"средний":2,"близко":3,"центр":4}
    print("  Районы: окраина / спальный / средний / близко / центр")
    print("  (Enter = выход)\n")
    while True:
        try:
            a  = input("  Площадь (м2): ").strip()
            if not a: break
            r  = float(input("  Комнат: "))
            fl = float(input("  Этаж: "))
            d  = dnames.get(input("  Район: ").strip().lower(), 2)
            ag = float(input("  Возраст дома (лет): "))
            feat = np.array([float(a),r,fl,d,ag], dtype=np.float32)
            fn   = (feat - X_mean) / X_std
            inp  = torch.tensor(fn).unsqueeze(0).to(device)
            with torch.no_grad():
                p = model(inp).item() * Y_std + Y_mean
            print(f"\n  Цена: {p:.2f} млн руб\n")
        except (ValueError, EOFError, KeyboardInterrupt): break
    input("\n  Enter для возврата в меню...")

# ── 3. ТЕМПЕРАТУРА ────────────────────────────────────
def menu_temp():
    clear(); header(); print("  [3] ПРОГНОЗ ТЕМПЕРАТУРЫ\n")
    model_path = BASE / 'temperature_model.pth'
    if not model_path.exists():
        print("  Модель не найдена. Запустите temperature.py\n")
        input("  Enter..."); return

    ckpt = torch.load(model_path, weights_only=True, map_location=device)
    model = TempNet().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    X_mean = np.array(ckpt['X_mean'], dtype=np.float32)
    X_std  = np.array(ckpt['X_std'],  dtype=np.float32)
    Y_mean = float(ckpt['Y_mean'])
    Y_std  = float(ckpt['Y_std'])

    print("  Введите данные для прогноза температуры на завтра:")
    print("  (Enter = выход)\n")
    while True:
        try:
            t = input("  Температура сегодня (°C): ").strip()
            if not t: break
            p    = float(input("  Давление (мм рт.ст., обычно 750-770): "))
            hum  = float(input("  Влажность (%, 0-100): "))
            wind = float(input("  Ветер (м/с): "))
            cloud= float(input("  Облачность (0.0-1.0): "))
            mon  = float(input("  Месяц (1-12): "))
            feat = np.array([float(t), p, hum, wind, cloud, mon], dtype=np.float32)
            fn   = (feat - X_mean) / X_std
            inp  = torch.tensor(fn).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(inp).item() * Y_std + Y_mean
            print(f"\n  Прогноз на завтра: {pred:.1f} °C\n")
        except (ValueError, EOFError, KeyboardInterrupt): break
    input("\n  Enter для возврата в меню...")

# ── 4. ПРОГНОЗ ПРОДАЖ ────────────────────────────────
def menu_timeseries():
    clear(); header(); print("  [4] ПРОГНОЗ ПРОДАЖ\n")
    model_path = BASE / 'timeseries_model.pth'
    if not model_path.exists():
        print("  Модель не найдена. Запустите timeseries.py\n")
        input("  Enter..."); return

    np.random.seed(42)
    days = np.arange(3*365)
    sales = 300 + days*0.05 + 20*np.sin(2*np.pi*days/7) + \
            80*np.sin(2*np.pi*days/365-np.pi/2) + np.random.normal(0,15,len(days))
    sales = np.clip(sales,50,None).astype(np.float32)
    s_mean, s_std = sales.mean(), sales.std()

    model = TSModel().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    print("  Введите продажи за последние 30 дней через запятую")
    print("  (или просто Enter — использую последние реальные данные)\n")
    while True:
        try:
            inp = input("  30 значений: ").strip()
            if inp.lower() in ('выход','exit','q'): break
            if inp:
                vals = [float(x.strip()) for x in inp.split(',')]
                if len(vals) != 30: print("  Нужно 30 значений\n"); continue
                window = np.array(vals, dtype=np.float32)
            else:
                window = sales[-30:]
            wn = (window - s_mean) / s_std
            x  = torch.tensor(wn).unsqueeze(0).unsqueeze(-1).to(device)
            with torch.no_grad():
                p = model(x)[0].cpu().numpy() * s_std + s_mean
            print(f"\n  Прогноз на 7 дней:")
            for i, v in enumerate(p, 1):
                bar = '#' * int(v / s_mean * 15)
                print(f"  День {i}: {v:6.0f} ед  {bar}")
            print(f"  Среднее: {p.mean():.0f} ед\n")
        except (ValueError, EOFError, KeyboardInterrupt): break
    input("\n  Enter для возврата в меню...")

# ── 5. ТОНАЛЬНОСТЬ ────────────────────────────────────
def menu_sentiment():
    clear(); header(); print("  [5] АНАЛИЗ ТОНАЛЬНОСТИ ТЕКСТА\n")
    model_path = BASE / 'sentiment_model.pth'
    if not model_path.exists():
        print("  Модель не найдена. Запустите sentiment.py\n")
        input("  Enter..."); return

    from collections import Counter
    RAW = [
        "отличный товар очень доволен покупкой рекомендую всем",
        "быстрая доставка качество супер спасибо продавцу",
        "великолепное качество все соответствует описанию",
        "шикарная вещь буду заказывать ещё очень нравится",
        "превосходно всё отлично работает счастлив покупкой",
        "замечательный товар упаковка хорошая пришло быстро",
        "восхитительно качество на высоте очень рад",
        "прекрасная покупка цена качество идеальное соотношение",
        "классный продукт работает отлично доволен",
        "топ товар лучшее что брал очень рекомендую",
        "ужасное качество полный брак не рекомендую никому",
        "разочарован товар не соответствует описанию обман",
        "сломалось через день деньги на ветер не берите",
        "отвратительное качество пластик дешёвый выброшу",
        "мусор а не товар зря потратил деньги",
        "товар пришёл вовремя всё как описано нормально",
        "среднее качество цена соответствует ожидал большего",
        "обычный товар ничего особенного но работает",
    ]
    counter = Counter()
    for t in RAW: counter.update(t.split())
    vocab = {'<PAD>':0,'<UNK>':1}
    for w,_ in counter.most_common(500): vocab[w] = len(vocab)
    vocab_size = len(vocab)

    model = SentimentLSTM(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    def encode(text, maxlen=20):
        ids = [vocab.get(w,1) for w in text.lower().split()][:maxlen]
        ids += [0]*(maxlen-len(ids)); return ids

    LABELS = {0:"[-] Негативный", 1:"[=] Нейтральный", 2:"[+] Позитивный"}
    print("  Введите отзыв на русском языке:")
    print("  (Enter = выход)\n")
    while True:
        text = input("  Отзыв: ").strip()
        if not text: break
        enc = torch.tensor([encode(text)], dtype=torch.long).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(enc),1)[0].cpu().numpy()
        pred = probs.argmax()
        print(f"  {LABELS[pred]}")
        print(f"  Neg:{probs[0]*100:.0f}%  Neu:{probs[1]*100:.0f}%  Pos:{probs[2]*100:.0f}%\n")
    input("\n  Enter для возврата в меню...")

# ── 6. ДЕТЕКТОР СПАМА ────────────────────────────────
def menu_spam():
    clear(); header(); print("  [6] ДЕТЕКТОР СПАМА\n")
    model_path = BASE / 'spam_model.pth'
    if not model_path.exists():
        print("  Модель не найдена. Запустите train_spam.py\n")
        input("  Enter..."); return

    ckpt  = torch.load(model_path, weights_only=True, map_location=device)
    vocab = ckpt['vocab']
    model = SpamLSTM(ckpt['vocab_size'], ckpt.get('embed_dim',64), ckpt.get('hidden_dim',128)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    max_len = ckpt.get('max_len', 40)

    def encode(text):
        ids = [vocab.get(w.lower(), 1) for w in text.split()][:max_len]
        ids += [0] * (max_len - len(ids))
        return ids

    print("  Введите текст сообщения (английский или русский):")
    print("  (Enter = выход)\n")
    while True:
        text = input("  Сообщение: ").strip()
        if not text: break
        enc = torch.tensor([encode(text)], dtype=torch.long).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(enc), 1)[0].cpu().numpy()
        is_spam = probs[1] > 0.5
        label = "[!!!] СПАМ" if is_spam else "[ OK] НЕ СПАМ"
        print(f"  {label}")
        print(f"  Спам: {probs[1]*100:.1f}%  Нормальное: {probs[0]*100:.1f}%\n")
    input("\n  Enter для возврата в меню...")

# ── 7. СУЩНОСТИ В ТЕКСТЕ (NER) ───────────────────────
def menu_ner():
    clear(); header(); print("  [7] РАСПОЗНАВАНИЕ ИМЕНОВАННЫХ СУЩНОСТЕЙ (NER)\n")
    model_path = BASE / 'ner_model.pth'
    if not model_path.exists():
        print("  Модель не найдена. Запустите train_ner.py\n")
        input("  Enter..."); return

    ckpt    = torch.load(model_path, weights_only=True, map_location=device)
    vocab   = ckpt['vocab']
    tags    = ckpt['tags']
    max_len = ckpt.get('max_len', 20)
    model   = NERModel(ckpt['vocab_size'], ckpt.get('embed_dim',64),
                       ckpt.get('hidden_dim',128), len(tags)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    TAG_NAMES = {
        'PER': 'Персона', 'ORG': 'Организация',
        'LOC': 'Локация', 'DATE': 'Дата'
    }

    def analyze(text):
        tokens = text.split()[:max_len]
        ids = [vocab.get(t.lower(), 1) for t in tokens]
        ids += [0] * (max_len - len(ids))
        x = torch.tensor([ids], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x)[0, :len(tokens)]
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        result = [(tokens[i], tags[pred_ids[i]]) for i in range(len(tokens))]
        # Группируем сущности
        entities = {}
        cur_tokens, cur_type = [], None
        for tok, tag in result:
            if tag.startswith('B-'):
                if cur_tokens: entities.setdefault(cur_type, []).append(' '.join(cur_tokens))
                cur_tokens, cur_type = [tok], tag[2:]
            elif tag.startswith('I-') and cur_type == tag[2:]:
                cur_tokens.append(tok)
            else:
                if cur_tokens: entities.setdefault(cur_type, []).append(' '.join(cur_tokens))
                cur_tokens, cur_type = [], None
        if cur_tokens: entities.setdefault(cur_type, []).append(' '.join(cur_tokens))
        return result, entities

    print("  Поддерживаемые сущности: Персоны, Организации, Локации, Даты")
    print("  Двуязычная: English + Русский")
    print("  (Enter = выход)\n")
    while True:
        text = input("  Текст: ").strip()
        if not text: break
        result, entities = analyze(text)
        print()
        # Токены с тегами
        line = ""
        for tok, tag in result:
            if tag == 'O':
                line += tok + " "
            else:
                line += f"[{tok}|{tag}] "
        print(f"  {line.strip()}")
        # Список сущностей
        if entities:
            print()
            for etype, names in entities.items():
                label = TAG_NAMES.get(etype, etype)
                print(f"  {label}: {', '.join(names)}")
        else:
            print("  Сущности не найдены")
        print()
    input("\n  Enter для возврата в меню...")

# ── 8. РАСПОЗНАВАНИЕ ЦИФР ────────────────────────────
def menu_mnist():
    clear(); header(); print("  [8] РАСПОЗНАВАНИЕ РУКОПИСНЫХ ЦИФР\n")
    print("  Загрузка тестовых примеров из MNIST...\n")
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    class ConvBlock(nn.Module):
        def __init__(self, ic, oc, drop=0.0):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(ic,oc,3,padding=1), nn.BatchNorm2d(oc), nn.ReLU(),
                nn.Conv2d(oc,oc,3,padding=1), nn.BatchNorm2d(oc), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout2d(drop))
        def forward(self,x): return self.block(x)

    class MnistCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                ConvBlock(1,32,0.1), ConvBlock(32,64,0.2), ConvBlock(64,128,0.3))
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(128*3*3,256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256,10))
        def forward(self,x): return self.classifier(self.features(x))

    model_path = BASE / 'best_model_v3.pth'
    if not model_path.exists():
        print("  Модель не найдена. Запустите mnist_cnn_v3.py\n")
        input("  Enter..."); return

    cnn = MnistCNN().to(device)
    cnn.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    cnn.eval()

    ds = datasets.MNIST('./data', train=False, download=True,
                        transform=transforms.ToTensor())
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=20, shuffle=True)
    imgs, labels = next(iter(loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        preds = cnn(imgs).argmax(1).cpu()

    correct = (preds == labels).sum().item()
    print(f"  Результаты на 20 случайных примерах: {correct}/20 верно\n")
    print(f"  {'#':>3}  {'Реально':>8}  {'Предсказ':>9}  {'Статус':>6}")
    print("  " + "-"*35)
    for i in range(20):
        ok = "OK" if preds[i]==labels[i] else "FAIL"
        print(f"  {i+1:>3}  {labels[i].item():>8}  {preds[i].item():>9}  {ok:>6}")

    input("\n  Enter для возврата в меню...")

# ── 9. ДЕТЕКТОР МОШЕННИЧЕСТВА ─────────────────────────
def menu_anomaly():
    clear(); header(); print("  [9] ДЕТЕКТОР МОШЕННИЧЕСКИХ ТРАНЗАКЦИЙ\n")
    model_path = BASE / 'anomaly_model.pth'
    if not model_path.exists():
        print("  Модель не найдена. Запустите train_anomaly.py\n")
        input("  Enter..."); return

    ckpt      = torch.load(model_path, weights_only=True, map_location=device)
    model     = AnomalyAE().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    X_mean    = np.array(ckpt['X_mean'], dtype=np.float32)
    X_std     = np.array(ckpt['X_std'],  dtype=np.float32)
    threshold = float(ckpt['threshold'])

    print("  Введите параметры транзакции:")
    print("  Признаки: сумма(руб), час(0-23), кол-во операций в день,")
    print("            нестандартная категория(0/1), онлайн(0/1),")
    print("            баланс до операции, расстояние от дома(км)")
    print("  (Enter = выход)\n")

    # Демо с предустановленными транзакциями
    examples = [
        ("Обычная покупка", [500, 14, 3, 0, 0, 50000, 2.0]),
        ("Мошенничество (ночью, далеко)", [85000, 3, 15, 1, 1, 1000, 2500.0]),
        ("Крупная покупка (норма)", [12000, 16, 1, 0, 0, 200000, 1.5]),
    ]
    print("  Готовые примеры:")
    for i, (name, vals) in enumerate(examples, 1):
        feat = (np.array(vals, dtype=np.float32) - X_mean) / X_std
        x = torch.tensor(feat).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(x)
            err = float(((recon - x)**2).mean().item())
        result = "МОШЕННИЧЕСТВО" if err > threshold else "норма"
        flag = "!!!" if err > threshold else "   "
        print(f"  {flag} {i}. {name}: {result} (ошибка={err:.4f}, порог={threshold:.4f})")
    print()

    while True:
        try:
            line = input("  Введите 7 значений через запятую (или Enter для выхода): ").strip()
            if not line: break
            vals = [float(v.strip()) for v in line.split(',')]
            if len(vals) != 7: print("  Нужно 7 значений\n"); continue
            feat = (np.array(vals, dtype=np.float32) - X_mean) / X_std
            x = torch.tensor(feat).unsqueeze(0).to(device)
            with torch.no_grad():
                recon = model(x)
                err = float(((recon - x)**2).mean().item())
            if err > threshold:
                print(f"  !!! МОШЕННИЧЕСТВО (ошибка={err:.4f} > порог={threshold:.4f})\n")
            else:
                print(f"  [OK] Нормальная транзакция (ошибка={err:.4f})\n")
        except (ValueError, EOFError, KeyboardInterrupt): break
    input("\n  Enter для возврата в меню...")

# ── 10. КОНТРОЛЬ КАЧЕСТВА ─────────────────────────────
def menu_defect():
    clear(); header(); print("  [10] КОНТРОЛЬ КАЧЕСТВА ДЕТАЛЕЙ\n")
    model_path = BASE / 'defect_model.pth'
    if not model_path.exists():
        print("  Модель не найдена. Запустите train_defect.py\n")
        input("  Enter..."); return

    ckpt   = torch.load(model_path, weights_only=True, map_location=device)
    model  = DefectNet().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    X_mean = np.array(ckpt['X_mean'], dtype=np.float32)
    X_std  = np.array(ckpt['X_std'],  dtype=np.float32)

    FEATURES = [
        ("Толщина",         "мм",  10.0),
        ("Масса",           "г",   250.0),
        ("Твёрдость",       "HB",  200.0),
        ("Шероховатость",   "Ra",  1.6),
        ("Длина",           "мм",  100.0),
        ("Ширина",          "мм",  50.0),
        ("Температура",     "°C",  850.0),
        ("Время обработки", "с",   120.0),
    ]

    # Демо с образцами
    examples = [
        ("Нормальная деталь",    [10.01, 251.0, 201.0, 1.65, 100.05, 50.02, 852.0, 121.0]),
        ("Брак (деформация)",    [9.65,  253.0, 198.0, 4.20, 100.04, 49.98, 848.0, 120.5]),
        ("Брак (термообработка)",[10.00, 249.0, 145.0, 1.60, 99.98, 50.01, 1020.0, 95.0]),
    ]
    print("  Норма: " + ", ".join(f"{n}={v} {u}" for n, u, v in FEATURES) + "\n")
    print("  Примеры:")
    for name, vals in examples:
        feat = (np.array(vals, dtype=np.float32) - X_mean) / X_std
        x = torch.tensor(feat).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), 1)[0].cpu().numpy()
        is_defect = probs[1] > 0.5
        label = "БРАК" if is_defect else "норма"
        flag  = "!!!" if is_defect else "[ ]"
        print(f"  {flag} {name}: {label} (брак={probs[1]*100:.1f}%)")
    print()

    print("  Введите 8 измерений через запятую:")
    print("  (толщина, масса, твёрдость, шероховатость, длина, ширина, температура, время)")
    while True:
        try:
            line = input("  8 значений (или Enter для выхода): ").strip()
            if not line: break
            vals = [float(v.strip()) for v in line.split(',')]
            if len(vals) != 8: print("  Нужно 8 значений\n"); continue
            feat = (np.array(vals, dtype=np.float32) - X_mean) / X_std
            x = torch.tensor(feat).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(x), 1)[0].cpu().numpy()
            if probs[1] > 0.5:
                print(f"  !!! БРАК (вероятность={probs[1]*100:.1f}%)\n")
            else:
                print(f"  [OK] Деталь в норме (вероятность нормы={probs[0]*100:.1f}%)\n")
        except (ValueError, EOFError, KeyboardInterrupt): break
    input("\n  Enter для возврата в меню...")

# ── 11. СЕГМЕНТАЦИЯ КЛИЕНТОВ ──────────────────────────
def menu_clustering():
    clear(); header(); print("  [11] СЕГМЕНТАЦИЯ КЛИЕНТОВ\n")
    model_path = BASE / 'clustering_model.pth'
    if not model_path.exists():
        print("  Модель не найдена. Запустите train_clustering.py\n")
        input("  Enter..."); return

    ckpt    = torch.load(model_path, weights_only=True, map_location=device)
    model   = ClusterAutoencoder().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    centers = np.array(ckpt['kmeans_centers'], dtype=np.float32)
    mapping = {int(k): int(v) for k, v in ckpt['cluster_mapping'].items()}
    sc_mean = np.array(ckpt['scaler_mean'],  dtype=np.float32)
    sc_std  = np.array(ckpt['scaler_scale'], dtype=np.float32)

    SEGMENTS = {
        0: ("VIP",       "Высокий чек, редкие крупные покупки, лояльный"),
        1: ("Активный",  "Частые покупки, средний чек, регулярный"),
        2: ("Молодой",   "Молодой, редкие мелкие покупки, много визитов"),
        3: ("Пассивный", "Зрелый, давно не покупал, низкая активность"),
    }

    examples = [
        ("VIP клиент",     [42, 8, 15000, 120000, 10, 5]),
        ("Активный",       [32, 25, 3000, 75000, 7, 20]),
        ("Студент",        [23, 3, 1500, 5000, 20, 40]),
        ("Пассивный",      [55, 2, 2000, 4000, 90, 2]),
    ]
    FEAT = ["Возраст", "Заказов", "Ср. чек", "Итого потрачено", "Дней с покупки", "Визитов"]
    print("  Признаки: " + ", ".join(FEAT) + "\n")
    print("  Примеры:")
    for name, vals in examples:
        feat = (np.array(vals, dtype=np.float32) - sc_mean) / sc_std
        x = torch.tensor(feat).unsqueeze(0).to(device)
        with torch.no_grad():
            _, z = model(x)
        z = z[0].cpu().numpy()
        dists = np.linalg.norm(centers - z, axis=1)
        best_kmeans = int(np.argmin(dists))
        seg_idx = mapping.get(best_kmeans, best_kmeans)
        seg_name, seg_desc = SEGMENTS.get(seg_idx, (f"Кластер {seg_idx}", ""))
        print(f"  {name:18} → {seg_name} ({seg_desc})")
    print()

    print("  Введите 6 значений через запятую:")
    print("  (возраст, кол-во заказов, ср.чек(руб), итого(руб), дней с покупки, визитов)")
    while True:
        try:
            line = input("  6 значений (или Enter для выхода): ").strip()
            if not line: break
            vals = [float(v.strip()) for v in line.split(',')]
            if len(vals) != 6: print("  Нужно 6 значений\n"); continue
            feat = (np.array(vals, dtype=np.float32) - sc_mean) / sc_std
            x = torch.tensor(feat).unsqueeze(0).to(device)
            with torch.no_grad():
                _, z = model(x)
            z = z[0].cpu().numpy()
            dists = np.linalg.norm(centers - z, axis=1)
            best_kmeans = int(np.argmin(dists))
            seg_idx  = mapping.get(best_kmeans, best_kmeans)
            seg_name, seg_desc = SEGMENTS.get(seg_idx, (f"Кластер {seg_idx}", ""))
            print(f"\n  Сегмент: {seg_name}")
            print(f"  {seg_desc}\n")
        except (ValueError, EOFError, KeyboardInterrupt): break
    input("\n  Enter для возврата в меню...")

# ── 12. РЕКОМЕНДАЦИИ ФИЛЬМОВ ──────────────────────────
def menu_recommender():
    clear(); header(); print("  [12] РЕКОМЕНДАЦИИ ФИЛЬМОВ\n")
    model_path = BASE / 'recommender_model.pth'
    if not model_path.exists():
        print("  Модель не найдена. Запустите train_recommender.py\n")
        input("  Enter..."); return

    ckpt   = torch.load(model_path, weights_only=True, map_location=device)
    model  = RecommenderNCF(ckpt['n_users'], ckpt['n_movies'], ckpt['embed_dim']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    movie_vecs   = np.array(ckpt['movie_vecs'],      dtype=np.float32)
    movie_names  = ckpt['movie_names']
    movie_genres = ckpt['movie_genre_str']
    n_movies     = ckpt['n_movies']

    def recommend(liked_indices, top_n=5):
        mean_vec = movie_vecs[liked_indices].mean(axis=0)
        scores   = movie_vecs @ mean_vec
        scores[liked_indices] = -np.inf
        top = np.argsort(scores)[::-1][:top_n]
        return [(movie_names[i], movie_genres[i], float(scores[i])) for i in top]

    print("  Доступные фильмы (первые 20):")
    for i in range(min(20, n_movies)):
        print(f"  {i+1:>3}. {movie_names[i]:<28} [{movie_genres[i]}]")
    print(f"  ... и ещё {n_movies-20} фильмов. Всего: {n_movies}\n")

    print("  Введите номера понравившихся фильмов через запятую (1-based)")
    print("  Пример: 1,2,3")
    print("  Или введите часть названия для поиска")
    print("  (Enter = выход)\n")

    while True:
        try:
            inp = input("  Понравились (номера или поиск): ").strip()
            if not inp: break

            liked = []
            # Числа?
            parts = [p.strip() for p in inp.split(',')]
            all_nums = all(p.isdigit() for p in parts)
            if all_nums:
                for p in parts:
                    idx = int(p) - 1
                    if 0 <= idx < n_movies:
                        liked.append(idx)
                    else:
                        print(f"  Фильм {p} не найден (диапазон 1-{n_movies})")
            else:
                # Поиск по названию
                query = inp.lower()
                for i, name in enumerate(movie_names):
                    if query in name.lower():
                        liked.append(i)
                        print(f"  Найден: {i+1}. {name} [{movie_genres[i]}]")

            if not liked:
                print("  Не найдено\n"); continue

            liked_names = [movie_names[i] for i in liked]
            print(f"\n  Выбрано: {', '.join(liked_names)}")
            print("  Рекомендации:\n")
            for name, genre, score in recommend(liked):
                bar = '█' * int((score + 2) * 4)
                print(f"  → {name:<28} [{genre:<25}]  {score:.3f}")
            print()
        except (ValueError, EOFError, KeyboardInterrupt): break
    input("\n  Enter для возврата в меню...")

# ── 13. GAN — ГЕНЕРАЦИЯ ЦИФР ──────────────────────────
def menu_gan():
    clear(); header(); print("  [13] ГЕНЕРАЦИЯ ЦИФР (GAN)\n")
    model_path = BASE / 'gan_generator.pth'
    if not model_path.exists():
        print("  Модель ещё обучается. Подождите завершения gan.py\n")
        input("  Enter..."); return

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    G = Generator().to(device)
    G.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
    G.eval()

    print("  Какую цифру генерировать? (0-9 или Enter = случайные)\n")
    d_inp = input("  Цифра: ").strip()
    digit = int(d_inp) if d_inp.isdigit() else -1
    count = 64

    with torch.no_grad():
        z = torch.randn(count, 100, device=device)
        if digit == -1:
            labels = torch.randint(0, 10, (count,), device=device)
        else:
            labels = torch.full((count,), digit, device=device)
        imgs = G(z, labels).cpu()

    title = f"GAN: цифра {digit}" if digit != -1 else "GAN: случайные цифры"
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i,0], cmap='gray', vmin=-1, vmax=1)
        ax.axis('off')
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    fname = BASE / 'menu_gan_output.png'
    plt.savefig(fname, dpi=100); plt.close()
    print(f"  Сохранено: {fname}")
    input("\n  Enter для возврата в меню...")

# ── S. SMART ASSISTANT ────────────────────────────────
def menu_smart():
    clear(); header()
    print("  [S] SMART ASSISTANT\n")
    print("  Запускаю Smart Assistant...")
    subprocess.run([sys.executable, str(BASE/'smart_assistant.py')], cwd=BASE)

# ── R. ОБУЧИТЬ ВСЁ ───────────────────────────────────
def retrain_all():
    clear(); header()
    print("  [R] ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ\n")
    scripts = [
        ('math_solver.py',      'Арифметика (seq2seq)'),
        ('regression.py',       'Цена квартиры'),
        ('temperature.py',      'Прогноз температуры'),
        ('timeseries.py',       'Прогноз временных рядов'),
        ('sentiment.py',        'Анализ тональности'),
        ('train_spam.py',       'Детектор спама'),
        ('train_ner.py',        'Распознавание сущностей (NER)'),
        ('train_anomaly.py',    'Детектор мошенничества'),
        ('train_defect.py',     'Контроль качества (дефекты)'),
        ('train_clustering.py', 'Кластеризация клиентов'),
        ('train_recommender.py','Рекомендательная система'),
        ('gan.py',              'Генерация изображений (GAN)'),
        ('mnist_cnn_v3.py',     'Распознавание цифр (CNN)'),
    ]
    print("  Какие модели обучить?")
    for i, (f, name) in enumerate(scripts, 1):
        exists = (BASE / f).exists()
        trained = check(f.replace('.py', '_model.pth'))
        print(f"  {i:>2}. {name:<36} {'[скрипт найден]' if exists else '[скрипт не найден]'}")
    print("  a. Все сразу")
    print("  0. Назад\n")
    choice = input("  Выбор: ").strip()
    if choice == '0': return
    if choice == 'a':
        selected = [s for s, _ in scripts if (BASE / s).exists()]
    else:
        try:
            i = int(choice)-1
            selected = [scripts[i][0]] if (BASE / scripts[i][0]).exists() else []
            if not selected:
                print(f"  Скрипт {scripts[i][0]} не найден")
                input("  Enter..."); return
        except: return

    for script in selected:
        print(f"\n  Запускаю {script}...")
        subprocess.Popen([sys.executable, str(BASE/script)], cwd=BASE)
    print(f"\n  Запущено {len(selected)} скриптов в фоне.")
    input("  Enter для возврата в меню...")

# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    main_menu()
