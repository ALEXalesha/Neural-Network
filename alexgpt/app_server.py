"""
AlexGPT — Единый сервер
Все нейросети + AI ассистент в одном Flask приложении
"""
import base64, binascii, io, json, math, os, re, shutil, subprocess, sys, tempfile, threading, time, traceback, warnings
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

from paths import APP_DIR, DATA_DIR, FROZEN, MODELS_DIR as MODELS, lm_config_path, lm_studio_url, script_cmd

VERSION = "1.2.2"

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

# ── Классификатор текста (тональность, спам): BiLSTM + max-pooling по словам ──
# Та же токенизация, что в lab/hf_data.py
TOKEN_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]")

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, n_classes=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.drop  = nn.Dropout(0.4)
        self.fc    = nn.Linear(hidden_dim * 2, n_classes)
    def forward(self, x):
        out, _ = self.lstm(self.drop(self.embed(x)))
        out = out.masked_fill((x == 0).unsqueeze(-1), -1e4)
        return self.fc(self.drop(out.max(dim=1).values))

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
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=8, num_enc=4, num_dec=4, ff=1024,
                 norm_first=True, max_len=40):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len=max_len)
        with warnings.catch_warnings():
            # «enable_nested_tensor ... norm_first was True» — для pre-LN nested tensor не нужен, это не ошибка
            warnings.simplefilter("ignore", UserWarning)
            self.transformer = nn.Transformer(
                d_model=d_model, nhead=nhead, num_encoder_layers=num_enc, num_decoder_layers=num_dec,
                dim_feedforward=ff, dropout=0.1, batch_first=True, norm_first=norm_first)
        self.max_len = max_len
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

# ── NER: эмбеддинг слова + CNN по символам → BiLSTM (как в lab/train_ner.py) ──
class NERTagger(nn.Module):
    def __init__(self, n_words, n_chars, n_tags, word_dim=100, char_dim=32, char_filters=64, hidden=192):
        super().__init__()
        self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=0)
        self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.char_cnn = nn.Conv1d(char_dim, char_filters, 3, padding=1)
        self.lstm = nn.LSTM(word_dim + char_filters, hidden, num_layers=2, bidirectional=True,
                            batch_first=True, dropout=0.3)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden * 2, n_tags)
    def forward(self, words, chars):
        B, T, C = chars.shape
        c = self.char_emb(chars.view(B * T, C)).transpose(1, 2)
        c = torch.relu(self.char_cnn(c)).max(dim=2).values.view(B, T, -1)
        out, _ = self.lstm(self.drop(torch.cat([self.word_emb(words), c], dim=-1)))
        return self.fc(self.drop(out))

# ── SketchNet (DrawGuess: цифры, буквы, фигуры, объекты, математические знаки) ──
class SketchNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
                                  nn.Linear(256, num_classes))
    def forward(self, x): return self.head(self.features(x))

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
    "ner": "ner_model.pth", "recommender": "recommender_model.pth", "sketch": "drawguess.pth",
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

def load_classifier(name):
    """Тональность и спам: {"model_state", "vocab", "labels", "max_len", "config"} из lab/text_classifier.py"""
    def build():
        c = ckpt(name)
        return SimpleNamespace(model=ready(TextClassifier(len(c["vocab"]), **c["config"]), c["model_state"]),
                               vocab=c["vocab"], labels=c["labels"], max_len=c["max_len"])
    return cached(name, build)

def classify(m, text):
    ids = [m.vocab.get(t, 1) for t in TOKEN_RE.findall(text.lower())][:m.max_len] or [1]
    with torch.inference_mode():
        return torch.softmax(m.model(torch.tensor([ids], device=DEVICE)), dim=1)[0].tolist()

def load_sentiment():
    return load_classifier("sentiment_model.pth")

def load_translator(direction):
    def build():
        en_tok, ru_tok = tokenizer("bpe_en.json"), tokenizer("bpe_ru.json")
        src, tgt = (en_tok, ru_tok) if direction == "en2ru" else (ru_tok, en_tok)
        c = ckpt(f"translator_bpe_{direction}.pth")
        model = ready(TranslatorBPE(src.get_vocab_size(), tgt.get_vocab_size(), **c["config"]), c["model"])
        # Паддинг маскируем явно; nested tensor для pre-LN всё равно недоступен и только шлёт предупреждение
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
    return load_classifier("spam_model.pth")

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
        model = NERTagger(len(c["word_vocab"]), len(c["char_vocab"]), len(c["tags"]), **c["config"])
        return SimpleNamespace(model=ready(model, c["model_state"]), words=c["word_vocab"], chars=c["char_vocab"],
                               tags=c["tags"], max_chars=c["max_chars"])
    return cached("ner", build)

def load_sketch():
    def build():
        labels = json.loads((MODELS / "drawguess_labels.json").read_text(encoding="utf-8"))
        return SimpleNamespace(model=ready(SketchNet(len(labels)), ckpt("drawguess.pth")), labels=labels)
    return cached("sketch", build)

# ═══════════════════════════════════════════════════════
# LM STUDIO
# ═══════════════════════════════════════════════════════
def _load_lm_cfg():
    p = lm_config_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

LM_CFG  = _load_lm_cfg()
LM_URL  = lm_studio_url(LM_CFG)
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
    """Убирает рассуждения R1 в начале ответа. Раньше резалось всё до последнего </think> и до конца
    после любого <think> — и ответ, где эти слова встречаются как текст, пропадал целиком.
    Современный LM Studio отдаёт рассуждения отдельным полем reasoning_content, тут — старый формат."""
    text = text.lstrip()
    if text.startswith("<think>"):
        end = text.find("</think>")
        return "" if end < 0 else text[end + len("</think>"):].strip()
    return text.strip()

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

def lm_in_memory():
    """Модели, реально загруженные в память. /v1/models отдаёт все скачанные (JIT-загрузка)."""
    j = lm_get(f"{LM_HOST}/api/v0/models")
    if not isinstance(j, dict):
        return None
    return [m.get("id", "") for m in j.get("data", []) if m.get("state") == "loaded"]

@app.route("/api/lm/models")
def api_lm_models():
    available = lm_loaded()
    if available is None:
        return jsonify({"models": [], "available": [], "online": False, "error": "LM Studio недоступна"})
    in_memory = lm_in_memory()
    return jsonify({"models": available if in_memory is None else in_memory, "available": available, "online": True})

@app.route("/api/lm/config")
def api_lm_config():
    return jsonify({"models": LM_MDLS, "url": LM_URL})

QUANT_RE = re.compile(r"(IQ\d_\w+|Q\d_K_[SML]|Q\d_K|Q\d_\d|BF16|F16)", re.I)
ANSI_RE  = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

def lms_path():
    exe = shutil.which("lms") or str(Path.home() / ".lmstudio" / "bin" / ("lms.exe" if sys.platform == "win32" else "lms"))
    return exe if Path(exe).exists() else None

def lms_local_models():
    """Скачанные модели из `lms ls --json` (с размерами); None — утилиты lms нет."""
    exe = lms_path()
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "ls", "--json"], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=30, creationflags=NO_WINDOW).stdout
        return [m for m in json.loads(out) if isinstance(m, dict)]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None

def hf_spec(model_id):
    owner, repo = model_id.split("/")[:2]
    q = QUANT_RE.search(model_id.rsplit("/", 1)[-1])
    return f"https://huggingface.co/{owner}/{repo}" + (f"@{q.group(1).upper()}" if q else "")

def lms_bundled(info):
    """Модель, которую LM Studio поставляет вместе с собой (например, Nomic Embed): она лежит не в папке
    моделей, а в .internal/bundled-models и удаляется только вместе с LM Studio."""
    try:
        return (Path.home() / ".lmstudio" / ".internal" / "bundled-models" / info.get("path", "")).is_file()
    except (OSError, ValueError, TypeError):
        return False

@app.route("/api/lm/catalog")
def lm_catalog():
    local     = lms_local_models()
    in_memory = set(lm_in_memory() or [])
    by_path   = {m.get("path", "").lower(): m for m in local or []}
    used      = set()
    roles = []
    for role, cfg in LM_MDLS.items():
        mid  = cfg.get("model_id", "")
        info = by_path.get(mid.lower())
        if info:
            used.add(info.get("path", "").lower())
        q = QUANT_RE.search(mid.rsplit("/", 1)[-1])
        roles.append({
            "role": role, "title": cfg.get("title", role), "about": cfg.get("about", ""), "model_id": mid,
            "repo": mid.split("/")[1] if mid.count("/") >= 2 else mid, "quant": q.group(1).upper() if q else "",
            "publisher": mid.split("/")[0] if "/" in mid else "", "hf_url": hf_spec(mid).split("@")[0] if "/" in mid else "",
            "size_gb": round(info["sizeBytes"] / 1e9, 1) if info and info.get("sizeBytes") else cfg.get("size_gb"),
            "params": info.get("paramsString") if info else None,
            "downloaded": None if local is None else info is not None,
            "loaded": bool(info) and info.get("modelKey") in in_memory,
            "key": info.get("modelKey") if info else None, "deletable": bool(info) and bool(model_files(info)),
        })
    others = [{"key": m.get("modelKey"), "name": m.get("displayName") or m.get("modelKey"),
               "size_gb": round(m.get("sizeBytes", 0) / 1e9, 1), "type": m.get("type"),
               "params": m.get("paramsString"), "quant": (m.get("quantization") or {}).get("name"),
               "vision": bool(m.get("vision")), "loaded": m.get("modelKey") in in_memory,
               "deletable": bool(model_files(m)), "builtin": lms_bundled(m)}
              for m in local or [] if m.get("path", "").lower() not in used]
    return jsonify({"online": lm_loaded() is not None, "lms": lms_path() is not None, "roles": roles, "others": others,
                    "disk_gb": round(sum(m.get("sizeBytes", 0) for m in local or []) / 1e9, 1)})

SPEC_RE = re.compile(r"(https://huggingface\.co/)?[\w.-]+/[\w.-]+(@[\w.-]+)?")

def require_lms():
    exe = lms_path()
    if not exe:
        abort(501, "Не найдена утилита LM Studio (lms). Установи LM Studio и запусти его хотя бы один раз.")
    return exe

def lms_run(*args, timeout=180):
    r = subprocess.run([require_lms(), *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       stdin=subprocess.DEVNULL, timeout=timeout, creationflags=NO_WINDOW)
    return r.returncode, ANSI_RE.sub("", (r.stdout or "") + (r.stderr or "")).strip()

def local_model(key):
    if not isinstance(key, str) or not key:
        abort(400, "Не указана модель")
    info = next((m for m in lms_local_models() or [] if m.get("modelKey") == key), None)
    if not info:
        abort(404, "Модель не найдена среди скачанных")
    return info

@app.route("/api/lm/load", methods=["POST"])
def lm_load_one():
    info = local_model(body().get("key"))
    code, out = lms_run("load", info["modelKey"], "-y", timeout=300)
    return jsonify({"ok": code == 0, "msg": out.splitlines()[-1] if out else ""})

@app.route("/api/lm/unload_one", methods=["POST"])
def lm_unload_one():
    info = local_model(body().get("key"))
    code, out = lms_run("unload", info["modelKey"])
    return jsonify({"ok": code == 0, "msg": out.splitlines()[-1] if out else ""})

def lms_dirs():
    root = Path.home() / ".lmstudio"
    try:
        cfg = json.loads((root / "settings.json").read_text(encoding="utf-8"))
        models = Path(cfg.get("downloadsFolder") or root / "models")
    except (OSError, ValueError):
        models = root / "models"
    return models.resolve(), (root / "hub" / "models").resolve()

def model_files(info):
    """Что удалять: для обычной модели — её .gguf (и mmproj, если других моделей в папке нет),
    для модели из каталога LM Studio — манифест в hub и папку с весами."""
    try:
        return _model_files(info.get("path", ""))
    except (ValueError, OSError):   # путь с \0, слишком длинный, недоступный — ничего не трогаем
        return []

def _model_files(path):
    models, hub = lms_dirs()
    target = (models / path).resolve()
    if target.is_file() and target.suffix == ".gguf" and models in target.parents:
        rest = [p for p in target.parent.glob("*.gguf") if p != target and not p.name.lower().startswith("mmproj")]
        return [target] if rest else [target.parent]
    manifest = (hub / path).resolve()
    if (manifest / "manifest.json").exists() and hub in manifest.parents:
        found = [manifest]
        for dep in json.loads((manifest / "manifest.json").read_text(encoding="utf-8")).get("dependencies", []):
            for src in dep.get("sources", []):
                d = (models / src.get("user", "") / src.get("repo", "")).resolve()
                if src.get("type") == "huggingface" and d.is_dir() and models in d.parents:
                    found.append(d)
        return found
    return []

@app.route("/api/lm/delete", methods=["POST"])
def lm_delete():
    info  = local_model(body().get("key"))
    paths = model_files(info)
    if not paths:
        abort(409, "Эту модель можно удалить только в самом LM Studio (My Models)")
    lms_run("unload", info["modelKey"])
    freed = 0
    for p in paths:
        freed += sum(f.stat().st_size for f in (p.rglob("*") if p.is_dir() else [p]) if f.is_file())
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    models, _ = lms_dirs()
    parent = paths[0].parent
    while parent != models and models in parent.parents and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent
    lm_log(f"🗑 Удалена модель {info['modelKey']} ({freed / 1e9:.1f} ГБ)")
    return jsonify({"ok": True, "freed_gb": round(freed / 1e9, 1)})

@app.route("/api/lm/get", methods=["POST"])
def lm_get_model():
    d    = body()
    role = d.get("role")
    if isinstance(role, str) and role:
        cfg = LM_MDLS.get(role)
        if not cfg or "/" not in cfg.get("model_id", ""):
            abort(400, "Неизвестная роль")
        spec = hf_spec(cfg["model_id"])
    else:
        spec = text_field(d, "spec")
        if not SPEC_RE.fullmatch(spec):
            abort(400, "Нужна ссылка вида https://huggingface.co/автор/модель или имя автор/модель (можно с @Q4_K_M)")
    exe = require_lms()
    proc = subprocess.Popen([exe, "get", spec, "-y"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                            errors="replace", creationflags=NO_WINDOW)
    lm_log(f"⬇ lms get {spec}")

    def generate():
        last, sent_at = "", 0.0
        try:
            for raw in proc.stdout:
                line = ANSI_RE.sub("", raw).strip(" \r\n⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                # Прогресс-бар lms перерисовывается много раз в секунду — шлём не чаще 3 раз
                if not line or line == last or ("%" in line and time.time() - sent_at < 0.33):
                    continue
                last, sent_at = line, time.time()
                yield sse({"line": line})
            code = proc.wait()
            lm_log(f"{'✓' if code == 0 else '✗'} lms get завершён (код {code})")
            yield sse({"done": True, "ok": code == 0})
        finally:
            if proc.poll() is None:
                proc.terminate()
    return event_stream(generate())

def parse_homework(raw):
    # R1 любит Markdown: «**ОТВЕТ:**» — снимаем выделение только вокруг ключевых слов, не в самом ответе
    raw = re.sub(r"\*\*\s*(ЗАДАНИЕ\s*\d*|ОТВЕТ|ПОЯСНЕНИЕ)\s*(:?)\s*\*\*", r"\1\2", raw, flags=re.I)

    def answer_of(block):
        # Ответ — до «ПОЯСНЕНИЕ» или пустой строки: так ловится и «ОТВЕТ:\n1) x = 4\n2) б) Париж»
        ans = re.search(r"ОТВЕТ\s*:?[ \t]*(.*?)(?=ПОЯСНЕНИЕ|\n[ \t]*\n|$)", block, re.I | re.S)
        exp = re.search(r"ПОЯСНЕНИЕ\s*:?\s*(.*)", block, re.I | re.S)
        a = ans.group(1).strip() if ans else block.strip().split("\n")[0]
        return a, exp.group(1).strip() if exp else ""

    blocks = [b for b in re.split(r"ЗАДАНИЕ\s*\d+\s*:?", raw, flags=re.I)[1:] if b.strip()]
    if not blocks:
        # Модель часто пишет несколько «ОТВЕТ:» подряд без заголовков «ЗАДАНИЕ N:»
        blocks = [b for b in re.split(r"(?=ОТВЕТ\s*:)", raw, flags=re.I) if re.match(r"ОТВЕТ\s*:", b, re.I)]
        if len(blocks) < 2:
            return answer_of(raw)
    parsed = [answer_of(b) for b in blocks]
    answer = "\n".join(f"Задание {i}: {a}" for i, (a, _) in enumerate(parsed, 1))
    return answer, "\n".join(e for _, e in parsed if e)

HOMEWORK_TIMEOUT = 300   # первая загрузка модели в память + рассуждения R1

def lm_error_text(resp):
    """Текст ошибки из ответа LM Studio ({"error": "..."} или {"error": {"message": ...}}), а не просто «400»."""
    try:
        err = resp.json().get("error")
        return (err.get("message") if isinstance(err, dict) else err) or resp.reason
    except (ValueError, AttributeError):
        return resp.text[:300] or resp.reason

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

    def ask(mid, max_tokens):
        if mid == LM_MDLS.get("reasoner", {}).get("model_id"):
            # Рекомендации DeepSeek для R1: без system prompt (всё в сообщении пользователя), температура ~0.6 —
            # с system prompt и 0.1 модель на тесте «Столица Франции?» выбрала Лион
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{content}"}]
            temperature = 0.6
        else:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
            temperature = 0.1
        payload = {"model": mid, "temperature": temperature, "max_tokens": max_tokens, "stream": False,
                   "messages": messages}
        r = requests.post(f"{LM_URL}/chat/completions", json=payload, timeout=HOMEWORK_TIMEOUT)
        if not r.ok:
            # LM Studio иногда отвечает 400 «Channel Error», пока модель догружается — одна повторная попытка
            log(f"[HOMEWORK] LM Studio {r.status_code}: {lm_error_text(r)} — повтор", "WARNING")
            time.sleep(3)
            r = requests.post(f"{LM_URL}/chat/completions", json=payload, timeout=HOMEWORK_TIMEOUT)
        if not r.ok:
            raise requests.HTTPError(f"{r.status_code}: {lm_error_text(r)}")
        return strip_think(r.json()["choices"][0]["message"].get("content") or "")

    try:
        # R1 сначала рассуждает (отдельное поле reasoning_content) — ему нужен запас токенов
        raw = ask(model_id, 4096 if role == "reasoner" else 1024)
        writer = LM_MDLS.get("writer", {}).get("model_id")
        if not raw and role == "reasoner" and writer:
            log("[HOMEWORK] рассуждения съели все токены — спрашиваю writer", "WARNING")
            raw = ask(writer, 1024)
    except requests.ConnectionError:
        return jsonify({"error": "LM Studio офлайн"}), 503
    except requests.Timeout:
        return jsonify({"error": "Модель думала дольше 5 минут — попробуй ещё раз или разбей задание"}), 504
    except (requests.RequestException, KeyError, IndexError, ValueError, TypeError) as e:
        return jsonify({"error": f"Ошибка LM Studio: {e}"}), 502
    if not raw:
        return jsonify({"error": "Модель не дала ответа — попробуй ещё раз"}), 502

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

# ── Угадай рисунок (DrawGuess) ──
def sketch_frame(arr):
    """Та же нормализация, что при обучении DrawGuess: обрезка по краске,
    длинная сторона 20 px, центр кадра 28×28, значения 0..1."""
    ys, xs = np.where(arr > 0.1)
    if len(xs) == 0:
        return None
    crop = Image.fromarray((arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * 255).astype(np.uint8))
    w, h = crop.size
    scale = 20 / max(w, h)
    small = crop.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    frame = Image.new("L", (28, 28), 0)
    frame.paste(small, ((28 - small.width) // 2, (28 - small.height) // 2))
    return np.asarray(frame, dtype=np.float32) / 255.0

@app.route("/api/sketch/predict", methods=["POST"])
def sketch_predict():
    d   = body()
    img = decode_image(d.get("image"))
    if img.width * img.height > 4096 * 4096:
        abort(400, "Слишком большое изображение")
    top = int_field(d, "top", 8, 1, 20)
    frame = sketch_frame(np.asarray(img.convert("L"), dtype=np.float32) / 255.0)
    if frame is None:
        abort(400, "Холст пустой")
    m = load_sketch()
    with torch.inference_mode():
        probs = torch.softmax(m.model(torch.from_numpy(frame).view(1, 1, 28, 28).to(DEVICE)), 1)[0].cpu()
    p, idx = probs.topk(min(top, len(m.labels)))
    return jsonify({"guesses": [{"label": m.labels[i], "prob": round(float(v) * 100, 1)}
                                for v, i in zip(p.tolist(), idx.tolist())]})

# ── Тональность ──
@app.route("/api/sentiment/analyze", methods=["POST"])
def sentiment_analyze():
    m      = load_sentiment()
    probs  = classify(m, text_field(body(), "text"))
    emojis = ["😠", "😐", "😊"]
    pred   = int(np.argmax(probs))
    return jsonify({"label": m.labels[pred], "emoji": emojis[pred],
                    "scores": {l: round(p * 100, 1) for l, p in zip(m.labels, probs)}})

# ── Перевод ──
@app.route("/api/translate", methods=["POST"])
def translate():
    d         = body()
    text      = text_field(d, "text")
    direction = d.get("direction", "en2ru")
    if direction not in ("en2ru", "ru2en"):
        abort(400, "direction должен быть en2ru или ru2en")
    t = load_translator(direction)
    # Модель училась на отдельных предложениях — длинный текст переводим по предложениям
    parts = [s for s in SENT_SPLIT_RE.split(text.strip()) if s.strip()][:MAX_TRANSLATE_SENTENCES]
    result = " ".join(filter(None, (translate_sentence(t, s) for s in parts)))
    return jsonify({"translation": result or "(пустой результат)"})

SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
MAX_TRANSLATE_SENTENCES = 30

def translate_sentence(t, text):
    m = t.model
    MAX = m.max_len; PAD, SOS, EOS = 0, 1, 2
    ids = [SOS] + t.src.encode(text).ids[:MAX - 2] + [EOS]
    ids += [PAD] * (MAX - len(ids))
    src = torch.tensor([ids], device=DEVICE)
    pad = src == PAD
    with torch.inference_mode():
        mem = m.transformer.encoder(m.pos_enc(m.src_embed(src) * math.sqrt(m.d)), src_key_padding_mask=pad)
        tgt_ids = [SOS]
        # Позиционное кодирование знает только MAX позиций
        for _ in range(MAX - 1):
            tgt = torch.tensor([tgt_ids], device=DEVICE)
            T   = tgt.size(1)
            tm  = torch.triu(torch.ones(T, T, device=DEVICE, dtype=torch.bool), diagonal=1)
            out = m.transformer.decoder(m.pos_enc(m.tgt_embed(tgt) * math.sqrt(m.d)), mem,
                                        tgt_mask=tm, memory_key_padding_mask=pad)
            nid = m.fc(out[:, -1]).argmax(-1).item()
            if nid == EOS: break
            tgt_ids.append(nid)
    return t.tgt.decode(tgt_ids[1:]).strip()

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
    ham_prob, spam_prob = classify(load_spam(), text_field(body(), "text"))
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
    text  = text_field(body(), "text")
    m     = load_ner()
    found = list(TOKEN_RE.finditer(text))[:NER_MAX_TOKENS]
    if not found:
        return jsonify({"tokens": [], "entities": []})
    words = torch.tensor([[m.words.get(t.group().lower(), 1) for t in found]], device=DEVICE)
    chars = torch.zeros(1, len(found), m.max_chars, dtype=torch.long)
    for j, t in enumerate(found):
        for k, ch in enumerate(t.group()[:m.max_chars]):
            chars[0, j, k] = m.chars.get(ch, 1)
    with torch.inference_mode():
        preds = m.model(words, chars.to(DEVICE))[0].argmax(-1).tolist()
    tags = [m.tags[p] for p in preds]
    # Группируем BIO в сущности; I- без B- тоже начинает сущность. Текст берём из оригинала по позициям,
    # чтобы «August 4, 1961» не превратилось в «August 4 , 1961»
    entities, start, typ = [], None, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("I-") and tag[2:] == typ:
            continue
        if typ:
            entities.append({"text": text[found[start].start():found[i - 1].end()], "type": typ})
        start, typ = (i, tag[2:]) if tag != "O" else (None, None)
    return jsonify({"tokens": [{"token": t.group(), "tag": g} for t, g in zip(found, tags)], "entities": entities})

NER_MAX_TOKENS = 512

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
CHAT_HISTORY_MSGS, CHAT_HISTORY_CHARS = 16, 12000
THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)

def chat_history(raw):
    """Прошлые реплики чата от клиента: только текст, роли user/assistant, последние сообщения
    в пределах лимита символов (7B-модели с контекстом 4k иначе обрежут начало)."""
    if not isinstance(raw, list):
        return []
    out, total = [], 0
    for m in reversed(raw[-CHAT_HISTORY_MSGS:]):
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant") or not isinstance(m.get("content"), str):
            continue
        # Рассуждения R1 в истории не нужны — только занимают контекст
        text = THINK_RE.sub("", m["content"]).strip() if m["role"] == "assistant" else m["content"].strip()
        if not text:
            continue
        if total + len(text) > CHAT_HISTORY_CHARS:
            break
        total += len(text)
        out.append({"role": m["role"], "content": text})
    out.reverse()
    # Модели чата ожидают, что диалог начинается с пользователя
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out

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
    msgs += chat_history(d.get("history"))
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
                        delta = json.loads(chunk)["choices"][0]["delta"]
                        tok, think = delta.get("content") or "", delta.get("reasoning_content") or ""
                    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                        continue
                    # DeepSeek R1 в LM Studio шлёт рассуждения отдельным полем — показываем, чтобы не было «тишины»
                    if think:
                        n_tokens += 1
                        yield sse({"type": "think", "text": think})
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

UI_THEMES  = {"", "midnight", "graphite", "light", "sepia"}
UI_ACCENTS = {"", "blue", "green", "orange", "pink", "teal"}

@app.route("/api/ui", methods=["POST"])
def save_ui():
    """Тема и акцент для экрана загрузки. Окно показывает его до запуска сервера и читает их из ui.json."""
    d = request.get_json(silent=True)
    d = d if isinstance(d, dict) else {}
    ui = {}
    for key, allowed in (("theme", UI_THEMES), ("accent", UI_ACCENTS)):
        v = d.get(key)
        if isinstance(v, str) and v and v in allowed:
            ui[key] = v
    (DATA_DIR / "ui.json").write_text(json.dumps(ui), encoding="utf-8")
    return jsonify(ui)

@app.route("/api/unload", methods=["POST"])
def unload_all():
    """Выгружает модели из LM Studio и модуль рисования при закрытии приложения."""
    stop_draw_worker()
    if not (lm_in_memory() and lms_path()):
        return jsonify({"ok": True})
    lms_run("unload", "--all", timeout=15)
    return jsonify({"ok": True})

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
    names = lm_in_memory()
    if names is None:
        return jsonify({"ok": False, "msg": "LM Studio недоступен"})
    if not names:
        return jsonify({"ok": True, "msg": "Нет загруженных моделей"})
    if not lms_path():
        return jsonify({"ok": False, "msg": f"Выгрузи вручную в LM Studio: {', '.join(names)}"})
    code, _ = lms_run("unload", "--all")
    return jsonify({"ok": code == 0, "msg": f"Выгружено: {', '.join(names)}" if code == 0 else "Не удалось выгрузить"})

@app.route("/api/unload/<model>", methods=["POST"])
def unload_model(model):
    keys = {"translator": ["trans_en2ru", "trans_ru2en"]}.get(model, [model])
    with _lock:
        removed = list(_cache) if model == "all" else [k for k in keys if k in _cache]
        for k in removed:
            del _cache[k]
    if model in ("all", "draw") and _draw.proc is not None:
        stop_draw_worker()      # SDXL держит ~7 ГБ видеопамяти в отдельном процессе
        removed.append("draw")
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    import gc; gc.collect()
    log(f"Выгружено из памяти: {removed or ['ничего не найдено']}", "INFO")
    return jsonify({"ok": True, "removed": removed})

# ── Рисование по описанию: SDXL-Turbo в отдельном процессе («модуль рисования») ──
# CUDA-torch + diffusers весят ~5 ГБ, поэтому в установщик их не кладём: модуль ставится по кнопке
# в папку данных (uv разворачивает свой Python и пакеты), рисует отдельный процесс draw_worker.py.
# Модель SDXL-Turbo берётся из общего кэша Hugging Face и между модулем и исходниками не дублируется.
DRAW_ADDON  = DATA_DIR / "addons" / "draw"
DRAW_STOP   = DATA_DIR / "draw.stop"
DRAW_MODEL  = "stabilityai/sdxl-turbo"
UV_URL      = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
TORCH_INDEX = {"cuda": "https://download.pytorch.org/whl/cu130", "cpu": "https://download.pytorch.org/whl/cpu"}
DRAW_PKGS   = ["diffusers", "transformers", "accelerate", "safetensors"]
SNAPSHOT_CODE = ("from huggingface_hub import snapshot_download; "
                 f"snapshot_download({DRAW_MODEL!r}, allow_patterns=['*.json', '*.txt', '*.fp16.safetensors'])")
_draw = SimpleNamespace(proc=None, lock=threading.Lock(), installing=False)

def draw_addon_python():
    py = DRAW_ADDON / "venv" / "Scripts" / "python.exe"
    return py if (DRAW_ADDON / "installed.json").exists() and py.exists() else None

def draw_python():
    """Чем рисовать: Python модуля рисования, а при запуске из исходников — текущий, если в нём есть diffusers."""
    if py := draw_addon_python():
        return str(py)
    if not FROZEN:
        import importlib.util
        if importlib.util.find_spec("diffusers"):
            return sys.executable
    return None

def hf_model_dir():
    hub = os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return Path(hub) / ("models--" + DRAW_MODEL.replace("/", "--"))

def dir_size_gb(p):
    # В кэше Hugging Face snapshots/ — ссылки на blobs/; по ссылкам не идём, иначе размер удваивается
    try:
        return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) / 1e9, 1) \
            if p.exists() else 0.0
    except OSError:
        return 0.0

def clean_env(**extra):
    """Окружение для чужого Python: без переменных PyInstaller, которые сбили бы его импорты."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("_PYI", "_MEI")) and k not in ("PYTHONPATH", "PYTHONHOME")}
    return {**env, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", **extra}

def stop_draw_worker():
    proc, _draw.proc = _draw.proc, None
    if proc and proc.poll() is None:
        try:
            proc.stdin.close()          # draw_worker завершается, когда закрыт stdin
            proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()

def start_draw_worker():
    if _draw.proc and _draw.proc.poll() is None:
        return _draw.proc
    py = draw_python()
    if not py:
        abort(501, "Модуль рисования не установлен — нажми «Установить модуль рисования»")
    log("Запуск модуля рисования (загрузка SDXL-Turbo)…")
    env = clean_env(ALEXGPT_DRAW_STOP=str(DRAW_STOP), ALEXGPT_DRAW_MODEL=DRAW_MODEL)
    if hf_model_dir().exists():
        env["HF_HUB_OFFLINE"] = "1"     # модель уже скачана — не ходим в интернет при каждом запуске
    err_log = open(DATA_DIR / "draw_worker.log", "a", encoding="utf-8")
    # В сборке draw_worker.py лежит в _internal рядом с библиотеками самой программы (Python 3.13).
    # Без -P Python модуля (3.12) поставил бы эту папку первой в sys.path и взял бы оттуда чужой torch.
    flags = ["-u", "-P"] if py == str(draw_addon_python()) else ["-u"]
    proc = subprocess.Popen([py, *flags, str(APP_DIR / "draw_worker.py")], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=err_log, text=True, encoding="utf-8",
                            errors="replace", creationflags=NO_WINDOW, env=env, cwd=str(DATA_DIR))
    err_log.close()
    first = proc.stdout.readline()
    try:
        ready = json.loads(first) if first else {"error": "модуль рисования завершился при запуске"}
    except ValueError:
        ready = {"error": f"непонятный ответ модуля рисования: {first[:200]}"}
    if not ready.get("ready"):
        proc.kill()
        log(f"Модуль рисования: {ready.get('error')}", "ERROR")
        abort(502, f"{ready.get('error')} (подробности в draw_worker.log)")
    log(f"Модуль рисования готов ({ready.get('device')})")
    _draw.proc = proc
    return proc

@app.route("/api/draw/addon")
def draw_addon_status():
    info = {}
    if (DRAW_ADDON / "installed.json").exists():
        try:
            info = json.loads((DRAW_ADDON / "installed.json").read_text(encoding="utf-8"))
        except ValueError:
            pass
    py = draw_python()
    return jsonify({"ready": py is not None, "addon": draw_addon_python() is not None,
                    "builtin": py is not None and draw_addon_python() is None,
                    "gpu": bool(shutil.which("nvidia-smi")), "installing": _draw.installing,
                    "addon_gb": info.get("size_gb", 0), "torch": info.get("torch"),
                    "model_gb": dir_size_gb(hf_model_dir()),
                    "running": _draw.proc is not None and _draw.proc.poll() is None})

def stream_process(cmd, env):
    """Выполняет команду и отдаёт её вывод строками SSE (прогресс-бары не чаще 3 раз в секунду); возвращает код."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW, env=env)
    last, sent_at = "", 0.0
    try:
        for raw in proc.stdout:        # в текстовом режиме \r тоже конец строки — прогресс приходит построчно
            line = ANSI_RE.sub("", raw).strip()
            if not line or line == last or ("%" in line and time.time() - sent_at < 0.33):
                continue
            last, sent_at = line, time.time()
            yield sse({"line": line[:300]})
        return proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()

def download_uv(dest):
    import zipfile
    r = requests.get(UV_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        member = next(n for n in z.namelist() if n.endswith("uv.exe"))
        (dest / "uv.exe").write_bytes(z.read(member))

@app.route("/api/draw/addon/install", methods=["POST"])
def draw_addon_install():
    if _draw.installing:
        abort(409, "Установка уже идёт")
    gpu = bool(shutil.which("nvidia-smi"))
    _draw.installing = True

    def gen():
        try:
            DRAW_ADDON.mkdir(parents=True, exist_ok=True)
            # Свой Python — внутри папки модуля, без кэша пакетов: удаление модуля убирает всё
            env = clean_env(UV_PYTHON_INSTALL_DIR=str(DRAW_ADDON / "python"), UV_NO_CACHE="1")
            uv = shutil.which("uv") or str(DRAW_ADDON / "uv.exe")
            if not Path(uv).exists():
                yield sse({"line": "▶ Скачиваю uv — установщик Python (~20 МБ)"})
                download_uv(DRAW_ADDON)
            venv = DRAW_ADDON / "venv"
            py = str(venv / "Scripts" / "python.exe")
            steps = [
                ("Python 3.12", [uv, "venv", "--python", "3.12", "--allow-existing", str(venv)]),
                ("PyTorch для видеокарты NVIDIA (~2,5 ГБ)" if gpu else "PyTorch для процессора (~0,3 ГБ)",
                 [uv, "pip", "install", "--python", py, "torch", "--index-url", TORCH_INDEX["cuda" if gpu else "cpu"]]),
                ("diffusers и transformers (~0,1 ГБ)", [uv, "pip", "install", "--python", py, *DRAW_PKGS]),
                ("Модель SDXL-Turbo (6,5 ГБ, если ещё не скачана)", [py, "-c", SNAPSHOT_CODE]),
            ]
            for title, cmd in steps:
                yield sse({"line": f"▶ {title}"})
                code = yield from stream_process(cmd, env)
                if code != 0:
                    yield sse({"done": True, "ok": False, "error": f"Шаг «{title}» завершился с ошибкой (код {code})"})
                    return
            (DRAW_ADDON / "installed.json").write_text(json.dumps({
                "torch": "cuda" if gpu else "cpu", "size_gb": dir_size_gb(DRAW_ADDON),
                "date": datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
            log("Модуль рисования установлен")
            yield sse({"done": True, "ok": True})
        except Exception as e:
            log(f"Установка модуля рисования: {e}", "ERROR")
            yield sse({"done": True, "ok": False, "error": str(e)})
        finally:
            _draw.installing = False
    return event_stream(gen())

@app.route("/api/draw/addon/remove", methods=["POST"])
def draw_addon_remove():
    if _draw.installing:
        abort(409, "Дождись окончания установки")
    with_model = body().get("model") is True
    stop_draw_worker()
    freed = 0.0
    for p in [DRAW_ADDON] + ([hf_model_dir()] if with_model else []):
        if p.exists():
            freed += dir_size_gb(p)
            shutil.rmtree(p)
    log(f"Модуль рисования удалён{' вместе с моделью' if with_model else ''}, освобождено {freed:.1f} ГБ")
    return jsonify({"ok": True, "freed_gb": round(freed, 1)})

@app.route("/api/draw/stop", methods=["POST"])
def draw_stop():
    DRAW_STOP.touch()
    return jsonify({"ok": True})

@app.route("/api/draw", methods=["POST"])
def draw_image():
    prompt = text_field(body(), "prompt")
    with _draw.lock:
        proc = start_draw_worker()
        DRAW_STOP.unlink(missing_ok=True)   # «Стоп», нажатый до этой картинки, её не касается
        try:
            proc.stdin.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        except OSError:
            line = ""
    if not line:
        stop_draw_worker()
        abort(502, "Модуль рисования неожиданно завершился (подробности в draw_worker.log)")
    try:
        return jsonify(json.loads(line))
    except ValueError:
        abort(502, "Непонятный ответ модуля рисования")

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
