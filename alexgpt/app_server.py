"""
AlexGPT — Единый сервер
Все нейросети + AI ассистент в одном Flask приложении
"""
import base64, binascii, io, json, math, os, re, subprocess, sys, tempfile, threading, time, traceback, warnings
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torch.nn as nn
from flask import Flask, Response, abort, jsonify, request, stream_with_context
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from paths import APP_DIR, DATA_DIR, FROZEN, MODELS_DIR as MODELS, lm_config_path, script_cmd

VERSION = "1.1.0"

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NO_WINDOW   = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

app = Flask(__name__,
            template_folder=str(APP_DIR / "templates"),
            static_folder=str(APP_DIR / "static"))

# ═══════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ
# ═══════════════════════════════════════════════════════
_log_buf    = deque(maxlen=1000)
_lm_log_buf = deque(maxlen=500)

def log(msg, level="INFO"):
    _log_buf.append({"time": datetime.now().strftime("%H:%M:%S"), "level": level, "msg": msg})
    print(f"[{level}] {msg}", flush=True)

def lm_log(msg, level="INFO"):
    _lm_log_buf.append({"time": datetime.now().strftime("%H:%M:%S"), "level": level, "msg": msg})

@app.before_request
def _local_origin_only():
    # Любой сайт в браузере может слать запросы на localhost — пускаем только свой UI
    origin = request.headers.get("Origin")
    if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost"):
        abort(403)

@app.after_request
def _log_request(response):
    if request.path.startswith("/api/") and request.path not in ("/api/logs", "/api/lm/logs"):
        level = "ERROR" if response.status_code >= 500 else "WARNING" if response.status_code >= 400 else "INFO"
        log(f"{request.method} {request.path} → {response.status_code}", level)
    return response

@app.errorhandler(HTTPException)
def _http_error(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": e.description}), e.code
    return e

@app.errorhandler(Exception)
def _log_exception(e):
    log(f"ИСКЛЮЧЕНИЕ {request.method} {request.path}: {traceback.format_exc()}", "ERROR")
    return jsonify({"error": str(e)}), 500

# ═══════════════════════════════════════════════════════
# РАЗБОР ЗАПРОСОВ
# ═══════════════════════════════════════════════════════
def body():
    d = request.get_json(silent=True)
    if not isinstance(d, dict):
        abort(400, "Ожидается JSON-объект")
    return d

def text_field(d, key, required=True):
    v = d.get(key) or ""
    if not isinstance(v, str):
        abort(400, f"Поле «{key}» должно быть строкой")
    v = v.strip()
    if required and not v:
        abort(400, "Пустой текст")
    return v

def num_field(d, key, default):
    try:
        x = float(d.get(key, default))
    except (TypeError, ValueError, OverflowError):
        abort(400, f"Поле «{key}» должно быть числом")
    if not math.isfinite(x):
        abort(400, f"Поле «{key}» вне допустимого диапазона")
    return x

def int_field(d, key, default, lo, hi):
    return int(min(max(num_field(d, key, default), lo), hi))

def features(d, spec):
    return np.array([num_field(d, k, v) for k, v in spec], dtype=np.float32)

def model_input(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        abort(400, "Значения вне допустимого диапазона")
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)

def finite(x):
    if not np.isfinite(np.asarray(x, dtype=np.float64)).all():
        abort(400, "Модель не может посчитать ответ для таких значений")
    return x

def decode_image(b64):
    if not isinstance(b64, str):
        abort(400, "Поле «image» должно быть строкой base64")
    b64 = b64.split(",", 1)[1] if "," in b64 else b64
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64, validate=True)))
        img.load()
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError):
        abort(400, "Не удалось прочитать изображение")
    return img

def sse(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

def event_stream(gen):
    return Response(stream_with_context(gen), mimetype="text/event-stream", headers=SSE_HEADERS)

# ═══════════════════════════════════════════════════════
# АРХИТЕКТУРЫ МОДЕЛЕЙ
# ═══════════════════════════════════════════════════════

# ── MiniGPT ──
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
    def forward(self, x, mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / self.head_dim ** 0.5
        if mask is not None: att = att.masked_fill(mask == 0, -1e9)
        att = torch.softmax(att, -1)
        return self.proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn  = SelfAttention(embed_dim, num_heads)
        self.ff    = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.ff(self.norm2(x))

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=6, ff_dim=1024, context_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb   = nn.Embedding(context_len, embed_dim)
        self.drop      = nn.Dropout(0.1)
        self.blocks    = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.norm      = nn.LayerNorm(embed_dim)
        self.head      = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.ctx       = context_len
    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        emb  = self.drop(self.token_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            emb = block(emb, mask)
        return self.head(self.norm(emb))

# ── MNIST ResNet ──
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(x + self.block(x))

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.block1 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.down1  = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.block2 = nn.Sequential(ResidualBlock(128), ResidualBlock(128))
        self.down2  = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.block3 = nn.Sequential(ResidualBlock(256), ResidualBlock(256))
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 10))
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x); x = self.down1(x)
        x = self.block2(x); x = self.down2(x)
        x = self.block3(x); x = self.gap(x)
        return self.classifier(x)

# ── Sentiment LSTM ──
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, n_classes=3):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm    = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(hidden_dim * 2, n_classes)
    def forward(self, x):
        emb = self.dropout(self.embed(x))
        _, (h, _) = self.lstm(emb)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))

# ── Translator ──
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class TranslatorBPE(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=8, num_enc=4, num_dec=4):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len=40)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_enc, num_decoder_layers=num_dec,
            dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(d_model, tgt_vocab)
        self.d  = d_model

# ── GAN Generator ──
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.net = nn.Sequential(
            nn.Linear(100 + 10, 7 * 7 * 256), nn.Unflatten(1, (256, 7, 7)),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1), nn.Tanh())
    def forward(self, z, labels=None):
        if labels is None: labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        return self.net(torch.cat([z, self.label_emb(labels)], dim=1))

# ── Price MLP ──
class PriceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x)

# ── TimeSeries LSTM ──
class TSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 128, 2, batch_first=True, dropout=0.2)
        self.attn = nn.Linear(128, 1)
        self.fc   = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 7))
    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), 1)
        return self.fc((w * out).sum(1))

# ── ClusterAutoencoder ──
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

# ── AnomalyAE ──
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

# ── DefectNet ──
class DefectNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x): return self.net(x)

# ── TempNet ──
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

# ── SpamLSTM ──
class SpamLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                             bidirectional=True, num_layers=2, dropout=0.3)
        self.drop  = nn.Dropout(0.4)
        self.fc    = nn.Linear(hidden_dim * 2, 2)
    def forward(self, x):
        e = self.embed(x)
        _, (h, _) = self.lstm(e)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(self.drop(h))

# ── RecommenderNCF ──
class RecommenderNCF(nn.Module):
    def __init__(self, n_users, n_movies, embed_dim=32):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users,  embed_dim)
        self.movie_emb = nn.Embedding(n_movies, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,  32),            nn.ReLU(),
            nn.Linear(32,  1),
        )
    def forward(self, u, m):
        return self.mlp(torch.cat([self.user_emb(u), self.movie_emb(m)], dim=1)).squeeze(1)

# ── NERModel ──
class NERModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_tags=9):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                             bidirectional=True, num_layers=2, dropout=0.3)
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(hidden_dim * 2, num_tags)
    def forward(self, x):
        e   = self.embed(x)
        out, _ = self.lstm(e)
        return self.fc(self.drop(out))  # [B, T, num_tags]

# ═══════════════════════════════════════════════════════
# КОНСТАНТЫ НОРМАЛИЗАЦИИ (те же что при обучении)
# ═══════════════════════════════════════════════════════

# -- Price normalization --
_rng = np.random.RandomState(42)
_N   = 10000
_area = _rng.uniform(25, 150, _N)
_rooms = _rng.randint(1, 6, _N).astype(float)
_floor = _rng.randint(1, 26, _N).astype(float)
_dist  = _rng.randint(0, 5, _N).astype(float)
_age   = _rng.uniform(0, 60, _N)
_ppm2  = np.array([120, 160, 200, 260, 350])
_base  = _area * _ppm2[_dist.astype(int)]
_fb    = np.where(_floor == 1, -5000, 0) + np.where(_floor >= 15, 3000, 0)
_price = (_base + _fb + _rooms * 8000 - _age * 500 + _rng.normal(0, 15000, _N)) / 1e6
_Xp    = np.stack([_area, _rooms, _floor, _dist, _age], 1).astype(np.float32)
_Yp    = _price.astype(np.float32).reshape(-1, 1)
PRICE_X_MEAN = _Xp.mean(0); PRICE_X_STD = _Xp.std(0)
PRICE_Y_MEAN = float(_Yp.mean()); PRICE_Y_STD = float(_Yp.std())

# -- TimeSeries normalization --
_rng2  = np.random.RandomState(42)
_days  = np.arange(3 * 365)
_sales = (300 + _days * 0.05
          + 20 * np.sin(2 * np.pi * _days / 7)
          + 80 * np.sin(2 * np.pi * _days / 365 - np.pi / 2)
          + _rng2.normal(0, 15, len(_days)))
_sales = np.clip(_sales, 50, None).astype(np.float32)
TS_SALES = _sales; TS_MEAN = float(_sales.mean()); TS_STD = float(_sales.std())

# -- Sentiment vocab --
_SENT_RAW = [
    "отличный товар очень доволен покупкой рекомендую всем",
    "быстрая доставка качество супер спасибо продавцу",
    "великолепное качество все соответствует описанию",
    "шикарная вещь буду заказывать ещё очень нравится",
    "превосходно всё отлично работает счастлив покупкой",
    "замечательный товар упаковка хорошая пришло быстро",
    "восхитительно качество на высоте очень рад",
    "прекрасная покупка цена качество идеальное соотношение",
    "классный продукт работает отлично доволен",
    "ужасное качество товар сломался через день",
    "не рекомендую деньги выброшены на ветер",
    "полный брак не соответствует описанию обман",
    "очень плохо товар пришёл повреждённым",
    "разочарован покупкой не работает вернул",
    "отвратительное обслуживание больше не закажу",
    "просто ужас качество ноль звёзд",
    "мусор а не товар требую возврат",
    "нормальный товар ничего особенного",
    "пришло в срок упаковка целая",
    "как описано без сюрпризов",
    "среднее качество цена соответствует",
    "неплохо но есть недостатки",
    "обычный товар для своей цены",
    "ничего не понравилось но и не расстроил",
    "нейтральный отзыв товар как товар",
]
_cnt = Counter()
for t in _SENT_RAW: _cnt.update(t.lower().split())
SENT_VOCAB = {"<PAD>": 0, "<UNK>": 1}
for w, _ in _cnt.most_common(500): SENT_VOCAB[w] = len(SENT_VOCAB)

PRICE_FIELDS    = [("area", 60), ("rooms", 2), ("floor", 5), ("district", 2), ("age", 10)]
TEMP_FIELDS     = [("temp_today", 10), ("pressure", 760), ("humidity", 60), ("wind", 3), ("cloud", 0.5), ("month", 6)]
CUSTOMER_FIELDS = [("age", 35), ("orders", 10), ("avg", 3000), ("total", 30), ("days", 20), ("visits", 10)]
TXN_FIELDS      = [("amount", 500), ("hour", 14), ("freq", 3), ("foreign", 0), ("online", 1),
                   ("balance", 50000), ("distance", 2)]
PART_FIELDS     = [("thickness", 10.0), ("mass", 250.0), ("hardness", 200.0), ("roughness", 1.6),
                   ("length", 100.0), ("width", 50.0), ("temp", 850.0), ("time", 120.0)]

MODEL_FILES = {
    "gpt": "gpt_multilingual.pth", "mnist": "mnist_web.pth", "sentiment": "sentiment_model.pth",
    "translator": "translator_bpe_en2ru.pth", "gan": "gan_generator.pth", "price": "price_model.pth",
    "timeseries": "timeseries_model.pth", "temperature": "temperature_model.pth", "spam": "spam_model.pth",
    "clustering": "clustering_model.pth", "anomaly": "anomaly_model.pth", "defect": "defect_model.pth",
    "ner": "ner_model.pth", "recommender": "recommender_model.pth",
}

# ═══════════════════════════════════════════════════════
# КЭШ МОДЕЛЕЙ
# ═══════════════════════════════════════════════════════
_cache = {}
_lock  = threading.Lock()

def cached(key, build):
    with _lock:
        if key not in _cache:
            log(f"Загрузка {key}…")
            _cache[key] = build()
            log(f"{key} готов")
        return _cache[key]

def ckpt(name, weights_only=True):
    return torch.load(MODELS / name, weights_only=weights_only, map_location=DEVICE)

def ready(model, state):
    model.load_state_dict(state)
    return model.to(DEVICE).eval()

def tokenizer(name):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(MODELS / name))

def load_gpt():
    def build():
        tok = tokenizer("bpe_tokenizer.json")
        return SimpleNamespace(model=ready(MiniGPT(tok.get_vocab_size()), ckpt("gpt_multilingual.pth")), tok=tok)
    return cached("gpt", build)

def load_mnist():
    return cached("mnist", lambda: ready(ResNet(), ckpt("mnist_web.pth")))

def load_sentiment():
    def build():
        state = ckpt("sentiment_model.pth")
        return ready(SentimentLSTM(state["embed.weight"].shape[0]), state)
    return cached("sentiment", build)

def load_translator(direction):
    def build():
        en_tok, ru_tok = tokenizer("bpe_en.json"), tokenizer("bpe_ru.json")
        src, tgt = (en_tok, ru_tok) if direction == "en2ru" else (ru_tok, en_tok)
        model = ready(TranslatorBPE(src.get_vocab_size(), tgt.get_vocab_size()),
                      ckpt(f"translator_bpe_{direction}.pth", weights_only=False)["model"])
        # При обучении encoder видел паддинг как обычные позиции, без nested tensor
        model.transformer.encoder.enable_nested_tensor = False
        model.transformer.encoder.use_nested_tensor = False
        return SimpleNamespace(model=model, src=src, tgt=tgt)
    return cached(f"trans_{direction}", build)

def load_gan():
    return cached("gan", lambda: ready(Generator(), ckpt("gan_generator.pth")))

def load_price():
    return cached("price", lambda: ready(PriceNet(), ckpt("price_model.pth")))

def load_timeseries():
    return cached("ts", lambda: ready(TSModel(), ckpt("timeseries_model.pth")))

def load_temperature():
    def build():
        c = ckpt("temperature_model.pth")
        return SimpleNamespace(model=ready(TempNet(), c["model_state"]),
                               x_mean=np.array(c["X_mean"], np.float32), x_std=np.array(c["X_std"], np.float32),
                               y_mean=float(c["Y_mean"]), y_std=float(c["Y_std"]))
    return cached("temp", build)

def load_spam():
    def build():
        c = ckpt("spam_model.pth")
        model = SpamLSTM(c["vocab_size"], c.get("embed_dim", 64), c.get("hidden_dim", 128))
        return SimpleNamespace(model=ready(model, c["model_state"]), vocab=c["vocab"], max_len=c.get("max_len", 40))
    return cached("spam", build)

def load_clustering():
    def build():
        c = ckpt("clustering_model.pth")
        return SimpleNamespace(model=ready(ClusterAutoencoder(), c["model_state"]),
                               centers=np.array(c["kmeans_centers"], np.float32),
                               mapping={int(k): int(v) for k, v in c["cluster_mapping"].items()},
                               mean=np.array(c["scaler_mean"], np.float32),
                               std=np.array(c["scaler_scale"], np.float32))
    return cached("cluster", build)

def load_anomaly():
    def build():
        c = ckpt("anomaly_model.pth")
        return SimpleNamespace(model=ready(AnomalyAE(), c["model_state"]),
                               x_mean=np.array(c["X_mean"], np.float32), x_std=np.array(c["X_std"], np.float32),
                               threshold=float(c["threshold"]))
    return cached("anomaly", build)

def load_defect():
    def build():
        c = ckpt("defect_model.pth")
        return SimpleNamespace(model=ready(DefectNet(), c["model_state"]),
                               x_mean=np.array(c["X_mean"], np.float32), x_std=np.array(c["X_std"], np.float32))
    return cached("defect", build)

def load_recommender():
    def build():
        c = ckpt("recommender_model.pth")
        return SimpleNamespace(model=ready(RecommenderNCF(c["n_users"], c["n_movies"], c["embed_dim"]), c["model_state"]),
                               vecs=np.array(c["movie_vecs"], np.float32), names=c["movie_names"],
                               genres=c["movie_genre_str"], genre_labels=c["genres"])
    return cached("rec", build)

def load_ner():
    def build():
        c = ckpt("ner_model.pth")
        model = NERModel(c["vocab_size"], c.get("embed_dim", 64), c.get("hidden_dim", 128), len(c["tags"]))
        return SimpleNamespace(model=ready(model, c["model_state"]), vocab=c["vocab"], tags=c["tags"],
                               max_len=c.get("max_len", 20))
    return cached("ner", build)

# ═══════════════════════════════════════════════════════
# LM STUDIO
# ═══════════════════════════════════════════════════════
def _load_lm_cfg():
    p = lm_config_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

LM_CFG  = _load_lm_cfg()
LM_URL  = LM_CFG.get("lm_studio_url", "http://localhost:1234/v1")
LM_HOST = LM_URL.rsplit("/v1", 1)[0]
LM_MDLS = LM_CFG.get("models", {})
LM_KWS  = LM_CFG.get("routing_keywords", {})
_stop_ev = threading.Event()

def lm_get(url, timeout=2):
    try:
        r = requests.get(url, timeout=timeout)
        return r.json() if r.ok else None
    except (requests.RequestException, ValueError):
        return None

def lm_loaded(timeout=2):
    """id загруженных моделей; None — LM Studio офлайн."""
    j = lm_get(f"{LM_URL}/models", timeout)
    return None if j is None else [m.get("id", m.get("name", "")) for m in j.get("data", [])]

def lm_error(loaded):
    if loaded is None:
        return "LM Studio офлайн. Запустите LM Studio и загрузите модель."
    if not loaded:
        return "В LM Studio нет загруженных моделей. Загрузите модель в LM Studio."
    return None

def strip_think(text):
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)
    return text.split("</think>")[-1].strip()

def _route_model(query: str) -> str:
    q = query.lower()
    scores = {n: sum(1 for kw in kws if kw in q) for n, kws in LM_KWS.items()}
    best = max(scores, key=scores.get) if scores else "reasoner"
    return best if scores.get(best, 0) > 0 else "reasoner"

# ═══════════════════════════════════════════════════════
# МАРШРУТЫ
# ═══════════════════════════════════════════════════════

@app.route("/")
def index():
    html = (APP_DIR / "templates" / "app.html").read_text(encoding="utf-8")
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}

@app.route("/api/logs")
def api_logs():
    return jsonify(list(_log_buf))

@app.route("/api/lm/logs")
def api_lm_logs():
    return jsonify(list(_lm_log_buf))

@app.route("/api/lm/models")
def api_lm_models():
    loaded = lm_loaded()
    if loaded is None:
        return jsonify({"models": [], "online": False, "error": "LM Studio недоступна"})
    return jsonify({"models": loaded, "online": True})

@app.route("/api/lm/config")
def api_lm_config():
    return jsonify({"models": LM_MDLS, "url": LM_URL})

@app.route("/api/lm/setup_check")
def lm_setup_check():
    """Проверяет статус LM Studio и нужных моделей для мастера настройки."""
    loaded = lm_loaded(timeout=3)
    online = loaded is not None
    loaded = loaded or []

    # Скачанные модели (LM Studio v0 API отдаёт {"data": [...]})
    downloaded = []
    if online:
        j = lm_get(f"{LM_HOST}/api/v0/models", timeout=3) or {}
        items = j.get("data", []) if isinstance(j, dict) else j
        downloaded = [m.get("id", m.get("path", "")) for m in items if isinstance(m, dict)]

    def _filename(path):
        return path.replace("\\", "/").split("/")[-1].lower()

    loaded_files = [_filename(ld) for ld in loaded]
    downloaded_files = [_filename(d) for d in downloaded]

    ROLE_NAMES = {"coder": "Кодер", "reasoner": "Аналитик", "writer": "Писатель", "vision": "Vision"}
    MODEL_META = {
        "deepseek-coder-6.7b-instruct.Q4_K_S.gguf": {"size": "3.9 GB", "arch": "llama",   "arch_color": "#1f6feb", "publisher": "TheBloke",           "icon": "💻"},
        "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf":  {"size": "4.7 GB", "arch": "qwen2",   "arch_color": "#da3633", "publisher": "lmstudio-community", "icon": "🧠"},
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf":          {"size": "4.7 GB", "arch": "qwen2",   "arch_color": "#da3633", "publisher": "lmstudio-community", "icon": "✍️"},
        "Qwen2-VL-7B-Instruct-Q4_K_M.gguf":         {"size": "7.4 GB", "arch": "qwen2vl", "arch_color": "#da3633", "publisher": "lmstudio-community", "icon": "👁️"},
    }
    required = []
    seen = set()
    for role, cfg in LM_MDLS.items():
        mid = cfg.get("model_id", "")
        if mid and mid not in seen:
            seen.add(mid)
            fname = _filename(mid)
            is_loaded     = any(fname in lf or lf in fname for lf in loaded_files)
            is_downloaded = is_loaded or any(fname in df or df in fname for df in downloaded_files)
            meta = MODEL_META.get(mid.split("/")[-1], {})
            required.append({
                "role":        role,
                "role_name":   ROLE_NAMES.get(role, role),
                "model_id":    mid,
                "loaded":      is_loaded,
                "downloaded":  is_downloaded,
                "size":        meta.get("size", ""),
                "arch":        meta.get("arch", ""),
                "arch_color":  meta.get("arch_color", "#555"),
                "publisher":   meta.get("publisher", ""),
                "icon":        meta.get("icon", "🤖"),
            })

    return jsonify({
        "online":     online,
        "loaded":     loaded,
        "required":   required,
        "all_loaded": online and all(m["loaded"] for m in required),
    })


@app.route("/api/lm/download", methods=["POST"])
def lm_download():
    """Запускает скачивание модели через LM Studio API."""
    model_id = text_field(body(), "model_id")
    try:
        r = requests.post(f"{LM_HOST}/api/v0/models/download", json={"model": model_id}, timeout=10)
        if r.ok:
            return jsonify({"ok": True, "msg": f"Скачивание начато: {model_id}"})
        # LM Studio может не поддерживать API скачивания — даём ссылку
        return jsonify({"ok": False, "msg": f"LM Studio вернула {r.status_code}. Скачай вручную в LM Studio.", "manual": True})
    except requests.RequestException as e:
        return jsonify({"ok": False, "msg": str(e), "manual": True})

@app.route("/api/lm/download_progress")
def lm_download_progress():
    j = lm_get(f"{LM_HOST}/api/v0/models/download/progress", timeout=3)
    return jsonify({"ok": j is not None, "data": j or {}})


def parse_homework(raw):
    def answer_of(block):
        ans = re.search(r"ОТВЕТ\s*:?[ \t]*(.*)", block, re.I)
        exp = re.search(r"ПОЯСНЕНИЕ\s*:?\s*(.*)", block, re.I | re.S)
        a = re.split(r"ПОЯСНЕНИЕ", ans.group(1), flags=re.I)[0].strip() if ans else block.strip().split("\n")[0]
        return a, exp.group(1).strip() if exp else ""

    blocks = [b for b in re.split(r"ЗАДАНИЕ\s*\d+\s*:?", raw, flags=re.I)[1:] if b.strip()]
    if not blocks:
        return answer_of(raw)
    parsed = [answer_of(b) for b in blocks]
    answer = "\n".join(f"Задание {i}: {a}" for i, (a, _) in enumerate(parsed, 1))
    return answer, "\n".join(e for _, e in parsed if e)

@app.route("/api/homework/solve", methods=["POST"])
def homework_solve():
    d       = body()
    text    = text_field(d, "text", required=False)
    image   = text_field(d, "image", required=False)   # base64 data-url
    subject = text_field(d, "subject", required=False)
    if not text and not image:
        abort(400, "Нет вопроса")

    loaded = lm_loaded()
    if err := lm_error(loaded):
        return jsonify({"error": err}), 503

    subject_hint = f"Предмет: {subject}. " if subject else ""
    system_prompt = (
        "Ты помощник школьника. Отвечай КРАТКО и ТОЧНО на русском языке.\n"
        "Если заданий несколько — отвечай на КАЖДОЕ по порядку:\n"
        "ЗАДАНИЕ 1:\nОТВЕТ: <ответ>\nПОЯСНЕНИЕ: <одна фраза>\n\n"
        "ЗАДАНИЕ 2:\nОТВЕТ: <ответ>\nПОЯСНЕНИЕ: <одна фраза>\n\n"
        "...и т.д.\n"
        "Если задание одно — просто:\nОТВЕТ: <ответ>\nПОЯСНЕНИЕ: <одна фраза>\n"
        "Если тест с вариантами — укажи букву/номер. Никакого лишнего текста."
    )
    user_msg = f"{subject_hint}{text}" if text else subject_hint + "Реши задание на изображении."
    role = "vision" if image else "reasoner"
    model_id = (LM_MDLS.get(role, {}).get("model_id") or LM_MDLS.get("reasoner", {}).get("model_id") or loaded[0])
    content = [{"type": "text", "text": user_msg}, {"type": "image_url", "image_url": {"url": image}}] if image else user_msg

    try:
        r = requests.post(f"{LM_URL}/chat/completions", json={
            "model": model_id,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
            "temperature": 0.1, "max_tokens": 512, "stream": False,
        }, timeout=60)
        r.raise_for_status()
        raw = strip_think(r.json()["choices"][0]["message"]["content"])
    except requests.ConnectionError:
        return jsonify({"error": "LM Studio офлайн"}), 503
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        return jsonify({"error": f"Ошибка LM Studio: {e}"}), 502

    answer, explanation = parse_homework(raw)
    return jsonify({"answer": answer, "explanation": explanation, "raw": raw})

@app.route("/api/status")
def api_status():
    return jsonify({"device": str(DEVICE),
                    "models": {k: (MODELS / f).exists() for k, f in MODEL_FILES.items()},
                    "lm_studio": lm_loaded() is not None,
                    "torch_available": True, "version": VERSION})

# ── Математика ──
import ast, operator as _op

_MATH_OPS = {ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul, ast.Div: _op.truediv,
             ast.Mod: _op.mod, ast.Pow: _op.pow, ast.USub: _op.neg, ast.UAdd: _op.pos}

def _math_eval(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _MATH_OPS:
        return _MATH_OPS[type(node.op)](_math_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _MATH_OPS:
        a, b = _math_eval(node.left), _math_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(a) > 1 and abs(b) * math.log10(abs(a)) > 1000:
            raise OverflowError
        return _MATH_OPS[type(node.op)](a, b)
    raise ValueError("Недопустимая операция")

@app.route("/api/math/solve", methods=["POST"])
def math_solve():
    expr  = text_field(body(), "expression")[:1000]
    clean = expr.replace("^", "**").replace("×", "*").replace("÷", "/").replace(",", ".")
    if not re.fullmatch(r"[0-9+\-*/().%\s]+", clean):
        return jsonify({"error": "Допустимы только числа, скобки и + - * / ** %"})
    try:
        value = _math_eval(ast.parse(clean, mode="eval").body)
        if isinstance(value, complex) or not math.isfinite(value):
            raise OverflowError
    except ZeroDivisionError:
        return jsonify({"error": "Деление на ноль"})
    except (OverflowError, MemoryError):
        return jsonify({"error": "Слишком большое число"})
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return jsonify({"error": "Не понял выражение"})
    value = round(float(value), 8)
    return jsonify({"expression": expr, "result": int(value) if value.is_integer() and abs(value) < 1e15 else value})

# ── MNIST ──
@app.route("/api/mnist/predict", methods=["POST"])
def mnist_predict():
    img = decode_image(body().get("image"))
    if img.width * img.height > 4096 * 4096:
        abort(400, "Слишком большое изображение")
    arr = np.array(img.convert("L"), dtype=np.uint8)

    rows = np.any(arr > 20, axis=1); cols = np.any(arr > 20, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]; c0, c1 = np.where(cols)[0][[0, -1]]
        h, w   = r1 - r0, c1 - c0; side = max(h, w)
        pad    = max(int(side * 0.3), 10)
        cr, cc = (r0 + r1) // 2, (c0 + c1) // 2; half = side // 2 + pad
        r0 = max(0, cr - half); r1 = min(arr.shape[0], cr + half)
        c0 = max(0, cc - half); c1 = min(arr.shape[1], cc + half)
        arr = arr[r0:r1, c0:c1]

    h, w   = arr.shape; scale = 20.0 / max(h, w)
    new_h  = max(1, int(h * scale)); new_w = max(1, int(w * scale))
    small  = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)
    canvas = np.zeros((28, 28), dtype=np.float32)
    y0 = (28 - new_h) // 2; x0 = (28 - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = np.array(small, dtype=np.float32)

    x = torch.from_numpy((canvas / 255.0 - 0.1307) / 0.3081).view(1, 1, 28, 28).float().to(DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(load_mnist()(x), dim=1)[0].tolist()
    pred = int(np.argmax(probs))
    return jsonify({"digit": pred, "confidence": round(max(probs) * 100, 1),
                    "probs": [round(p * 100, 1) for p in probs]})

# ── Тональность ──
@app.route("/api/sentiment/analyze", methods=["POST"])
def sentiment_analyze():
    text = text_field(body(), "text").lower()
    MAX  = 20
    ids  = [SENT_VOCAB.get(w, 1) for w in text.split()][:MAX]
    ids += [0] * (MAX - len(ids))
    with torch.inference_mode():
        probs = torch.softmax(load_sentiment()(torch.tensor([ids], device=DEVICE)), dim=1)[0].tolist()
    labels = ["негативный", "нейтральный", "позитивный"]
    emojis = ["😠", "😐", "😊"]
    pred   = int(np.argmax(probs))
    return jsonify({"label": labels[pred], "emoji": emojis[pred],
                    "scores": {l: round(p * 100, 1) for l, p in zip(labels, probs)}})

# ── Перевод ──
@app.route("/api/translate", methods=["POST"])
def translate():
    d         = body()
    text      = text_field(d, "text")
    direction = d.get("direction", "en2ru")
    if direction not in ("en2ru", "ru2en"):
        abort(400, "direction должен быть en2ru или ru2en")
    t = load_translator(direction)
    m = t.model
    MAX = 40; PAD, SOS, EOS = 0, 1, 2
    ids = [SOS] + t.src.encode(text).ids[:MAX - 2] + [EOS]
    ids += [PAD] * (MAX - len(ids))
    src = torch.tensor([ids], device=DEVICE)
    with torch.inference_mode():
        se  = m.pos_enc(m.src_embed(src) * math.sqrt(m.d))
        mem = m.transformer.encoder(se, src_key_padding_mask=(src == PAD))
        tgt_ids = [SOS]
        # Позиционное кодирование знает только MAX позиций
        for _ in range(MAX - 1):
            tgt = torch.tensor([tgt_ids], device=DEVICE)
            te  = m.pos_enc(m.tgt_embed(tgt) * math.sqrt(m.d))
            T   = tgt.size(1)
            tm  = torch.triu(torch.ones(T, T, device=DEVICE), diagonal=1).bool()
            out = m.transformer.decoder(te, mem, tgt_mask=tm)
            nid = m.fc(out[:, -1]).argmax(-1).item()
            if nid == EOS: break
            tgt_ids.append(nid)
    return jsonify({"translation": t.tgt.decode(tgt_ids[1:]).strip() or "(пустой результат)"})

# ── GPT стриминг ──
@app.route("/api/gpt/stream", methods=["POST"])
def gpt_stream():
    d           = body()
    seed        = text_field(d, "seed", required=False) or "[RU] Однажды"
    length      = int_field(d, "length", 200, 1, 400)
    temperature = min(num_field(d, "temperature", 0.8), 100.0)
    g   = load_gpt()
    CTX = 128
    ids = list(g.tok.encode(seed).ids[-CTX:])
    if not ids:
        abort(400, "Не удалось разобрать начало текста")

    def gen():
        yield sse({"text": seed})
        new_ids, shown = [], ""
        try:
            with torch.inference_mode():
                for _ in range(length):
                    logits = g.model(torch.tensor([ids[-CTX:]], device=DEVICE))[0, -1]
                    if temperature <= 0.01:
                        nid = int(logits.argmax())
                    else:
                        nid = int(torch.multinomial(torch.softmax(logits / temperature, 0), 1))
                    ids.append(nid); new_ids.append(nid)
                    # Byte-level BPE: один токен может быть половиной UTF-8 символа
                    text = g.tok.decode(new_ids)
                    if text.endswith("�"):
                        continue
                    delta = text[len(os.path.commonprefix([shown, text])):]
                    shown = text
                    if delta:
                        yield sse({"text": delta})
        except Exception as e:
            log(f"GPT: {e}", "ERROR")
            yield sse({"error": str(e)})
            return
        yield sse({"done": True})
    return event_stream(gen())

# ── GAN ──
@app.route("/api/gan/generate", methods=["POST"])
def gan_generate():
    d = body()
    try:
        digit = int(d.get("digit", -1))
    except (TypeError, ValueError, OverflowError):
        abort(400, "digit должен быть числом")
    if not -1 <= digit <= 9:
        abort(400, "digit: от 0 до 9 или -1 для случайных")
    count  = int_field(d, "count", 4, 1, 16)
    labels = torch.randint(0, 10, (count,)) if digit == -1 else torch.full((count,), digit)
    with torch.inference_mode():
        imgs = load_gan()(torch.randn(count, 100, device=DEVICE), labels.to(DEVICE)).cpu()
    results = []
    for i in range(count):
        img = ((imgs[i, 0].numpy() + 1) / 2 * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img).resize((140, 140), Image.BICUBIC).save(buf, format="PNG")
        results.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    return jsonify({"images": results, "digit": digit})

# ── Цена квартиры ──
@app.route("/api/price/predict", methods=["POST"])
def price_predict():
    inp = model_input((features(body(), PRICE_FIELDS) - PRICE_X_MEAN) / PRICE_X_STD)
    with torch.inference_mode():
        # Модель обучена на данных где ppm2 в руб/м² без *1000,
        # масштабируем обратно чтобы получить реалистичные цены
        p = finite((load_price()(inp).item() * PRICE_Y_STD + PRICE_Y_MEAN) * 1000)
    return jsonify({"price": round(p, 1), "price_str": f"{p:.1f} млн ₽"})

# ── Прогноз продаж ──
@app.route("/api/timeseries/forecast", methods=["POST"])
def ts_forecast():
    vals = body().get("values") or []
    if not isinstance(vals, list):
        abort(400, "values должен быть списком чисел")
    if 0 < len(vals) < 30:
        abort(400, "Нужно 30 значений")
    window = (np.array([num_field({"v": v}, "v", 0) for v in vals[-30:]], np.float32) if vals
              else TS_SALES[-30:])
    x = model_input(((window - TS_MEAN) / TS_STD)[:, None])
    with torch.inference_mode():
        p = finite(load_timeseries()(x)[0].cpu().numpy() * TS_STD + TS_MEAN)
    return jsonify({"forecast": [round(float(v), 1) for v in p], "mean": round(float(p.mean()), 1)})

# ── Прогноз температуры ──
@app.route("/api/temperature/predict", methods=["POST"])
def temperature_predict():
    m   = load_temperature()
    inp = model_input((features(body(), TEMP_FIELDS) - m.x_mean) / m.x_std)
    with torch.inference_mode():
        t = finite(m.model(inp).item() * m.y_std + m.y_mean)
    return jsonify({"temp_tomorrow": round(t, 1)})

# ── Обнаружение спама ──
@app.route("/api/spam/analyze", methods=["POST"])
def spam_analyze():
    text = text_field(body(), "text")
    m    = load_spam()
    ids  = [m.vocab.get(w, 1) for w in text.lower().split()[:m.max_len]]
    ids += [0] * (m.max_len - len(ids))
    with torch.inference_mode():
        probs = torch.softmax(m.model(torch.tensor([ids], device=DEVICE)), dim=1)[0].cpu().numpy()
    spam_prob, ham_prob = float(probs[1]), float(probs[0])
    return jsonify({
        "label":     "SPAM" if spam_prob > 0.5 else "HAM",
        "spam_prob": round(spam_prob * 100, 1),
        "ham_prob":  round(ham_prob  * 100, 1),
    })

# ── Кластеризация клиентов ──
_SEG_NAMES = {0: "VIP", 1: "Активные", 2: "Молодые", 3: "Пассивные"}
_SEG_RECS  = {
    "VIP":       "Персональный менеджер, эксклюзивные предложения, приоритетная поддержка",
    "Активные":  "Программа лояльности, скидки за частые покупки, ранний доступ к новинкам",
    "Молодые":   "Акции, геймификация, промокоды в соцсетях, рассрочка",
    "Пассивные": "Реактивационное письмо, скидка 20%, напоминание о брошенной корзине",
}

@app.route("/api/cluster/customer", methods=["POST"])
def cluster_customer():
    m   = load_clustering()
    inp = model_input((features(body(), CUSTOMER_FIELDS) - m.mean) / m.std)
    with torch.inference_mode():
        z = m.model(inp)[1].cpu().numpy()
    dists   = finite(np.linalg.norm(m.centers - z, axis=1))
    cluster = int(dists.argmin())
    seg     = _SEG_NAMES.get(m.mapping.get(cluster, 0), f"Группа {cluster}")
    return jsonify({
        "segment":    seg,
        "cluster":    cluster,
        "confidence": round(float(1 / (1 + dists.min())), 3),
        "all_dists":  [round(float(v), 3) for v in dists],
        "rec":        _SEG_RECS.get(seg, "—"),
    })

# ── Обнаружение аномалий (транзакции) ──
@app.route("/api/anomaly/check", methods=["POST"])
def anomaly_check():
    m   = load_anomaly()
    inp = model_input((features(body(), TXN_FIELDS) - m.x_mean) / m.x_std)
    with torch.inference_mode():
        err = finite(float(((inp - m.model(inp)) ** 2).mean().item()))
    risk = round(err / m.threshold, 2)
    if   risk > 1.5: label = "ВЫСОКИЙ РИСК"
    elif risk > 1.0: label = "Подозрительно"
    else:            label = "Норма"
    return jsonify({
        "label":     label,
        "risk":      risk,
        "recon_err": round(err, 5),
        "threshold": round(m.threshold, 5),
        "risk_pct":  min(100, round(risk * 50)),
    })

# ── Контроль качества (дефекты) ──
@app.route("/api/defect/check", methods=["POST"])
def defect_check():
    m   = load_defect()
    inp = model_input((features(body(), PART_FIELDS) - m.x_mean) / m.x_std)
    with torch.inference_mode():
        probs = finite(torch.softmax(m.model(inp), dim=1)[0].cpu().numpy())
    defect_prob = float(probs[1])
    return jsonify({
        "label":       "БРАК" if defect_prob > 0.5 else "НОРМА",
        "defect_prob": round(defect_prob * 100, 1),
        "ok_prob":     round(float(probs[0]) * 100, 1),
    })

# ── NER ──
@app.route("/api/ner/analyze", methods=["POST"])
def ner_analyze():
    text   = text_field(body(), "text")
    m      = load_ner()
    tokens = text.split()[:m.max_len]
    ids    = [m.vocab.get(t.lower(), 1) for t in tokens]
    ids   += [0] * (m.max_len - len(ids))
    with torch.inference_mode():
        preds = m.model(torch.tensor([ids], device=DEVICE))[0][:len(tokens)].argmax(-1).cpu().numpy()
    result = [{"token": t, "tag": m.tags[p]} for t, p in zip(tokens, preds)]
    # Группируем в сущности
    entities = []
    cur_ent, cur_type = [], None
    for item in result:
        tag = item["tag"]
        if tag.startswith("B-"):
            if cur_ent: entities.append({"text": " ".join(cur_ent), "type": cur_type})
            cur_ent  = [item["token"]]
            cur_type = tag[2:]
        elif tag.startswith("I-") and cur_type == tag[2:]:
            cur_ent.append(item["token"])
        else:
            if cur_ent: entities.append({"text": " ".join(cur_ent), "type": cur_type})
            cur_ent, cur_type = [], None
    if cur_ent: entities.append({"text": " ".join(cur_ent), "type": cur_type})
    return jsonify({"tokens": result, "entities": entities})

# ── Extractive Summarization ──
_STOP_WORDS = {"the", "a", "an", "is", "was", "are", "were", "be", "been", "being", "to", "of", "in", "for",
               "on", "with", "at", "by", "from", "that", "this", "it", "as", "or", "and", "but", "not",
               "have", "has", "had", "will", "would", "could", "should", "may", "might", "do", "did", "does",
               "i", "you", "he", "she", "we", "they",
               "в", "и", "на", "с", "по", "за", "из", "к", "у", "от", "до", "при", "о", "что", "не", "он", "она"}

def content_words(text):
    return [w for w in re.findall(r"\b\w+\b", text.lower()) if w not in _STOP_WORDS and len(w) > 2]

@app.route("/api/summarize", methods=["POST"])
def summarize():
    d       = body()
    text    = text_field(d, "text")
    n_sents = int_field(d, "n_sentences", 3, 1, 50)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
    if len(sentences) <= n_sents:
        return jsonify({"summary": text, "selected": list(range(len(sentences))), "sentences": sentences})
    # TF-IDF scoring
    word_freq = Counter(w for s in sentences for w in content_words(s))
    max_freq  = max(word_freq.values()) if word_freq else 1
    scores = []
    for i, s in enumerate(sentences):
        words = content_words(s)
        score = sum(word_freq[w] / max_freq for w in words) / max(len(words), 1)
        # Первое предложение часто задаёт тему
        if i == 0: score *= 1.2
        scores.append((score, i))
    scores.sort(key=lambda si: (-si[0], si[1]))
    selected = sorted(i for _, i in scores[:n_sents])
    return jsonify({"summary": " ".join(sentences[i] for i in selected), "selected": selected, "sentences": sentences})

# ── Кластеризация документов ──
@app.route("/api/cluster/documents", methods=["POST"])
def cluster_documents():
    d   = body()
    raw = d.get("documents") or []
    if not isinstance(raw, list):
        abort(400, "documents должен быть списком строк")
    docs = [str(s).strip() for s in raw if s is not None and str(s).strip()]
    if len(docs) < 2:
        abort(400, "Нужно минимум 2 документа")
    k = int_field(d, "k", 3, 1, len(docs))
    tok_docs = [content_words(t) for t in docs]
    vocab = sorted({w for td in tok_docs for w in td})
    if not vocab:
        abort(400, "Документы не содержат значимых слов")
    v2i = {w: i for i, w in enumerate(vocab)}
    # TF-IDF
    n, V = len(docs), len(vocab)
    tf  = np.zeros((n, V), dtype=np.float32)
    for i, td in enumerate(tok_docs):
        for w, c in Counter(td).items():
            tf[i, v2i[w]] = c / len(td)
    idf   = np.log(n / ((tf > 0).sum(axis=0) + 1))
    tfidf = tf * idf
    tfidf = tfidf / (np.linalg.norm(tfidf, axis=1, keepdims=True) + 1e-8)
    # K-Means (numpy)
    rng     = np.random.default_rng(42)
    centers = tfidf[rng.choice(n, k, replace=False)]
    labels  = np.zeros(n, dtype=int)
    for _ in range(50):
        new_lb = np.linalg.norm(tfidf[:, None] - centers[None], axis=2).argmin(axis=1)
        if (new_lb == labels).all(): break
        labels = new_lb
        for c in range(k):
            members = tfidf[labels == c]
            if len(members): centers[c] = members.mean(axis=0)
    # Ключевые слова кластера
    topic_names = []
    for c in range(k):
        top = centers[c].argsort()[-5:][::-1]
        kws = [vocab[i] for i in top if centers[c][i] > 0]
        topic_names.append(", ".join(kws[:3]) if (labels == c).any() and kws else f"Тема {c+1}")
    result = [{"doc": docs[i], "cluster": int(labels[i]), "topic": topic_names[int(labels[i])]} for i in range(n)]
    return jsonify({"clusters": result, "k": k, "topics": topic_names})

# ── Рекомендации ──
@app.route("/api/recommender/catalog", methods=["GET"])
def recommender_catalog():
    m = load_recommender()
    return jsonify({"movies": [{"name": n, "genre": g} for n, g in zip(m.names, m.genres)],
                    "genres": m.genre_labels})

@app.route("/api/recommender/recommend", methods=["POST"])
def recommender_recommend():
    d     = body()
    liked = d.get("liked") or []   # список названий фильмов
    if not isinstance(liked, list):
        abort(400, "liked должен быть списком названий")
    m = load_recommender()
    name2idx  = {n: i for i, n in enumerate(m.names)}
    liked_idx = sorted({name2idx[n] for n in liked if isinstance(n, str) and n in name2idx})
    if not liked_idx:
        abort(400, "Фильмы не найдены в каталоге")
    top_n  = min(int_field(d, "top_n", 6, 1, len(m.names)), len(m.names) - len(liked_idx))
    scores = m.vecs @ m.vecs[liked_idx].mean(axis=0)
    scores[liked_idx] = -np.inf
    top_idx = np.argsort(-scores, kind="stable")[:top_n]
    return jsonify({"recommendations": [{"name": m.names[i], "genre": m.genres[i],
                                         "score": round(float(scores[i]), 3)} for i in top_idx]})

# ── AI чат (LM Studio) ──
@app.route("/api/chat", methods=["POST"])
def chat():
    d       = body()
    message = text_field(d, "message")
    image   = text_field(d, "image", required=False)
    forced  = d.get("model") if isinstance(d.get("model"), str) else "auto"
    if image:
        b64 = image.split(",", 1)[1] if "," in image else image
        try:
            base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            abort(400, "Ошибка декодирования изображения")
        image = image if image.startswith("data:") else f"data:image/jpeg;base64,{b64}"
        forced = "vision"

    chosen    = forced if forced in LM_MDLS else _route_model(message)
    model_cfg = LM_MDLS.get(chosen, LM_MDLS.get("reasoner", {}))
    model_id  = model_cfg.get("model_id", "")

    err = lm_error(lm_loaded())
    if not err and not model_id:
        err = "Модель не настроена. Укажи model_id в lm_config.json."
    if err:
        return event_stream(iter([sse({"type": "error", "text": err})]))

    log(f"[CHAT] модель={chosen} | {message[:60]}")
    lm_log(f"▶ Запрос | модель: {chosen} ({model_id}) | {'📎 фото + ' if image else ''}{message[:80]}")

    msgs = []
    if model_cfg.get("system_prompt"):
        msgs.append({"role": "system", "content": model_cfg["system_prompt"]})
    if image:
        msgs.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}},
                                                 {"type": "text", "text": message}]})
    else:
        msgs.append({"role": "user", "content": message})

    def generate():
        t0 = time.time()
        n_tokens = 0
        try:
            yield sse({"type": "model", "text": chosen})
            _stop_ev.clear()
            resp = requests.post(f"{LM_URL}/chat/completions",
                                 json={"model": model_id, "messages": msgs, "stream": True,
                                       "temperature": model_cfg.get("temperature", 0.7),
                                       "max_tokens": model_cfg.get("max_tokens", 2048)},
                                 stream=True, timeout=180)
            if not resp.ok:
                lm_log(f"✗ Ошибка HTTP {resp.status_code} от LM Studio", "ERROR")
                yield sse({"type": "error", "text": f"LM Studio {resp.status_code}"})
                return

            lm_log(f"↳ Соединение OK ({int((time.time()-t0)*1000)} мс) — генерация…")
            try:
                for line in resp.iter_lines():
                    if _stop_ev.is_set():
                        lm_log(f"⏹ Остановлено пользователем ({n_tokens} токенов)")
                        break
                    line = line.decode("utf-8") if isinstance(line, bytes) else line
                    if not line.startswith("data: "): continue
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]": break
                    try:
                        tok = json.loads(chunk)["choices"][0]["delta"].get("content") or ""
                    except (ValueError, KeyError, IndexError, TypeError):
                        continue
                    if tok:
                        n_tokens += 1
                        yield sse({"type": "token", "text": tok})
            finally:
                resp.close()

            elapsed = round(time.time() - t0, 1)
            lm_log(f"✓ Готово | {n_tokens} токенов | {elapsed} с | {round(n_tokens/elapsed,1) if elapsed else '?'} tok/s")
            yield sse({"type": "done"})
        except requests.ConnectionError:
            lm_log("✗ LM Studio недоступна (ConnectionError)", "ERROR")
            yield sse({"type": "error", "text": "LM Studio не запущена"})
        except Exception as e:
            lm_log(f"✗ Исключение: {e}", "ERROR")
            yield sse({"type": "error", "text": str(e)})

    return event_stream(generate())

@app.route("/api/stop", methods=["POST"])
def stop():
    _stop_ev.set()
    return jsonify({"ok": True})

@app.route("/api/unload", methods=["POST"])
def unload_all():
    """Выгружает все модели из LM Studio при закрытии приложения."""
    loaded = lm_loaded(timeout=3)
    if loaded is None:
        return jsonify({"ok": True, "msg": "LM Studio офлайн"})
    for model_id in loaded:
        try:
            requests.post(f"{LM_HOST}/api/v0/models/unload", json={"identifier": model_id}, timeout=5)
        except requests.RequestException:
            pass
    return jsonify({"ok": True, "unloaded": loaded})

# ── Управление окном (пишем файл-команду, Qt читает по таймеру) ──
_WIN_CMD = Path(os.environ.get("ALEXGPT_WIN_CMD") or DATA_DIR / ".win_cmd")
WIN_ACTIONS = {"minimize", "maximize", "hide", "quit", "startmove"} | {
    f"startresize_{e}" for e in ("n", "s", "w", "e", "nw", "ne", "sw", "se")}

@app.route("/api/win/<action>", methods=["POST"])
def win_control(action):
    if action not in WIN_ACTIONS:
        abort(400, "unknown action")
    _WIN_CMD.write_text(action)
    return jsonify({"ok": True})

@app.route("/api/lm/unload", methods=["POST"])
def lm_unload():
    names = lm_loaded(timeout=3)
    if names is None:
        return jsonify({"ok": False, "msg": "LM Studio недоступен"})
    if not names:
        return jsonify({"ok": True, "msg": "Нет загруженных моделей"})
    # Пробуем DELETE /api/v0/models/loaded (LM Studio 0.3+)
    for mid in names:
        try:
            requests.delete(f"{LM_HOST}/api/v0/models/loaded/{mid}", timeout=5)
        except requests.RequestException:
            pass
    still = lm_loaded(timeout=3)
    if still is not None and len(still) < len(names):
        return jsonify({"ok": True, "msg": f"Выгружено: {', '.join(names)}"})
    lm_log(f"API выгрузки не поддерживается — модели: {', '.join(names)}", "WARNING")
    return jsonify({"ok": False, "msg": f"Выгрузи вручную в LM Studio: {', '.join(names)}"})

@app.route("/api/unload/<model>", methods=["POST"])
def unload_model(model):
    keys = {"translator": ["trans_en2ru", "trans_ru2en"]}.get(model, [model])
    with _lock:
        removed = list(_cache) if model == "all" else [k for k in keys if k in _cache]
        for k in removed:
            del _cache[k]
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    import gc; gc.collect()
    log(f"Выгружено из памяти: {removed or ['ничего не найдено']}", "INFO")
    return jsonify({"ok": True, "removed": removed})

# ── Генерация изображений (SDXL-Turbo через Hugging Face diffusers) ──
_draw_pipe  = None
_draw_lock  = threading.Lock()
_draw_stop  = False

def load_draw():
    global _draw_pipe
    with _draw_lock:
        if _draw_pipe is None:
            try:
                from diffusers import AutoPipelineForText2Image
            except ImportError:
                abort(501, "Генерация картинок недоступна в собранной версии." if FROZEN else
                      "Нужен пакет diffusers: pip install diffusers transformers accelerate")
            log("Загрузка SDXL-Turbo (diffusers)…")
            kw = {"torch_dtype": torch.float16, "variant": "fp16"} if DEVICE.type == "cuda" else {"torch_dtype": torch.float32}
            _draw_pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", **kw).to(DEVICE)
            log("SDXL-Turbo готов")
    return _draw_pipe

@app.route("/api/draw/stop", methods=["POST"])
def draw_stop():
    global _draw_stop
    _draw_stop = True
    return jsonify({"ok": True})

@app.route("/api/draw", methods=["POST"])
def draw_image():
    global _draw_stop
    prompt = text_field(body(), "prompt")
    _draw_stop = False
    pipe = load_draw()

    def _check_stop(pipe, step, timestep, kwargs):
        if _draw_stop:
            raise InterruptedError
        return kwargs

    try:
        out = pipe(prompt=prompt, num_inference_steps=1 if DEVICE.type == "cuda" else 4,
                   guidance_scale=0.0, width=512, height=512, callback_on_step_end=_check_stop)
    except InterruptedError:
        return jsonify({"error": "Остановлено"})
    if _draw_stop:
        return jsonify({"error": "Остановлено"})
    buf = io.BytesIO()
    out.images[0].save(buf, format="PNG")
    return jsonify({"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()})

# ── Coder Team ──
_coder_procs = {}

def run_script_stream(key, cmd, out_file, env=None, cleanup=None):
    # Старый результат не должен выдаваться за новый, если прогон упадёт
    out_file.unlink(missing_ok=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace", cwd=str(DATA_DIR), creationflags=NO_WINDOW,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "ALEXGPT_DATA": str(DATA_DIR), **(env or {})})
    _coder_procs[key] = proc

    def generate():
        try:
            for line in proc.stdout:
                yield sse({"line": line.rstrip()})
            proc.wait()
            yield sse({"done": True, "code": out_file.read_text(encoding="utf-8")} if out_file.exists()
                      else {"done": True})
        finally:
            if proc.poll() is None:
                proc.terminate()
            if _coder_procs.get(key) is proc:
                del _coder_procs[key]
            if cleanup:
                cleanup()
    return event_stream(generate())

def coder_iterations(d, default):
    # 0 = без ограничений
    n = int_field(d, "iterations", default, 0, 16)
    return str(999 if n == 0 else n)

@app.route("/api/coder_team/run", methods=["POST"])
def coder_team():
    task = text_field(body(), "task")
    return run_script_stream("v1", script_cmd("coder_team", task), DATA_DIR / "coder_output.py")

@app.route("/api/coder_team/stop", methods=["POST"])
def coder_team_stop():
    proc = _coder_procs.get("v1")
    if proc and proc.poll() is None:
        proc.terminate()
    return jsonify({"ok": True})

@app.route("/api/coder_team_v2/review", methods=["POST"])
def coder_team_v2_review():
    d    = body()
    code = text_field(d, "code")
    task = text_field(d, "task", required=False)
    fd, tmp = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    return run_script_stream("v2", script_cmd("coder_team_v2", "--review", tmp, task), DATA_DIR / "coder_output_v2.py",
                             env={"CODER_V2_MAX_ITER": coder_iterations(d, 3)},
                             cleanup=lambda: Path(tmp).unlink(missing_ok=True))

@app.route("/api/coder_v2/control", methods=["POST"])
def coder_v2_control():
    """Записывает команду управления для активного цикла Coder Team v2.
    Тело: {"action": "stop"} или {"action": "comment", "text": "..."}
    """
    d = body()
    action = d.get("action")
    if action not in ("stop", "comment"):
        abort(400, "action must be 'stop' or 'comment'")
    ctrl = {"action": action}
    if action == "comment":
        ctrl["text"] = text_field(d, "text")
    (DATA_DIR / "coder_v2_control.json").write_text(json.dumps(ctrl, ensure_ascii=False), encoding="utf-8")
    return jsonify({"ok": True})

@app.route("/api/coder_team_v2/run", methods=["POST"])
def coder_team_v2():
    d = body()
    task = text_field(d, "task")
    return run_script_stream("v2", script_cmd("coder_team_v2", task, coder_iterations(d, 4)),
                             DATA_DIR / "coder_output_v2.py")


def _lm_monitor():
    """Фоновый поток: опрашивает LM Studio каждые 30 сек и пишет статус в lm_log."""
    prev_models = None
    prev_online = None
    while True:
        models = lm_loaded(timeout=3)
        online = models is not None
        if online != prev_online:
            lm_log("🟢 LM Studio онлайн" if online else "🔴 LM Studio офлайн или недоступна",
                   "INFO" if online else "WARNING")
            prev_online = online
        if online and models != prev_models:
            lm_log(f"📦 Загружено моделей: {len(models)}" if models else "📭 Нет загруженных моделей")
            for m in models:
                lm_log(f"   • {m}")
            prev_models = models
        time.sleep(30)


def _exit_with_parent():
    # Сервер не должен переживать окно, которое его запустило
    pid = int(os.environ.get("ALEXGPT_PARENT_PID", 0))
    if not pid or sys.platform != "win32":
        return
    import ctypes
    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if handle:
        k32.WaitForSingleObject(handle, 0xFFFFFFFF)
    os._exit(0)


def run(port=None):
    port = port or int(os.environ.get("ALEXGPT_PORT", 5050))
    log(f"AlexGPT {VERSION} | APP_DIR={APP_DIR} | MODELS={MODELS} | DATA={DATA_DIR}")
    log(f"Устройство: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    for name, f in {**MODEL_FILES, "translator ru2en": "translator_bpe_ru2en.pth"}.items():
        found = (MODELS / f).exists()
        log(f"Модель {name}: {'найдена' if found else 'НЕ НАЙДЕНА'}", "INFO" if found else "WARNING")
    threading.Thread(target=_exit_with_parent, daemon=True).start()
    threading.Thread(target=_lm_monitor, daemon=True).start()
    log(f"Сервер запущен на порту {port}")
    app.run(debug=False, host="127.0.0.1", port=port, threaded=True)


if __name__ == "__main__":
    run()
