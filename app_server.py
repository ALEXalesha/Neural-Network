"""
AlexGPT — Единый сервер
Все нейросети + AI ассистент в одном Flask приложении
"""
import sys, io, os, json, base64, threading, re, math, warnings, logging
from collections import deque, Counter
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Фикс сломанного simplejson — блокируем его до импорта requests
# чтобы requests использовал встроенный json вместо сломанного simplejson
import sys as _sys
import types as _types
_fake_sj = _types.ModuleType('simplejson')
_fake_sj.JSONDecodeError = ValueError
_fake_sj.loads  = __import__('json').loads
_fake_sj.dumps  = __import__('json').dumps
_sys.modules['simplejson'] = _fake_sj

try:
    import numpy as np
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy", "-q"])
    import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    import types as _t
    nn = _t.SimpleNamespace(Module=object)
    torch = _t.SimpleNamespace(
        device=lambda x: x,
        cuda=_t.SimpleNamespace(is_available=lambda: False),
        no_grad=lambda: __import__('contextlib').nullcontext(),
    )

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE = Path(sys._MEIPASS)
elif "NEURAL_BASE" in os.environ:
    BASE = Path(os.environ["NEURAL_BASE"])
else:
    BASE = Path(__file__).parent

try:
    from flask import Flask, render_template, request, jsonify, Response, stream_with_context
    import requests as _req
except (ImportError, Exception):
    import subprocess
    # Убираем сломанный simplejson если есть, потом ставим нужные пакеты
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "simplejson", "-y", "-q"],
                   capture_output=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "flask", "requests", "-q"])
    from flask import Flask, render_template, request, jsonify, Response, stream_with_context
    import requests as _req

app = Flask(__name__,
            template_folder=str(BASE / "templates"),
            static_folder=str(BASE / "static"))

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

@app.after_request
def _log_request(response):
    if request.path.startswith("/api/") and request.path not in ("/api/logs", "/api/lm/logs"):
        level = "ERROR" if response.status_code >= 500 else "WARNING" if response.status_code >= 400 else "INFO"
        log(f"{request.method} {request.path} → {response.status_code}", level)
    return response

@app.errorhandler(Exception)
def _log_exception(e):
    import traceback
    log(f"ИСКЛЮЧЕНИЕ {request.method} {request.path}: {traceback.format_exc()}", "ERROR")
    return jsonify({"error": str(e)}), 500

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

# ═══════════════════════════════════════════════════════
# КЭШ МОДЕЛЕЙ
# ═══════════════════════════════════════════════════════
_cache = {}
_lock  = threading.Lock()

def _require_torch():
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch недоступен — локальные модели требуют установки PyTorch")

def load_gpt():
    _require_torch()
    with _lock:
        if "gpt" not in _cache:
            log("Загрузка GPT (BPE)…")
            try:
                from tokenizers import Tokenizer
                tok   = Tokenizer.from_file(str(BASE / "bpe_tokenizer.json"))
                model = MiniGPT(tok.get_vocab_size()).to(DEVICE)
                path  = BASE / "gpt_multilingual.pth"
                if not path.exists(): path = BASE / "gpt_bpe_model.pth"
                model.load_state_dict(torch.load(path, weights_only=True, map_location=DEVICE))
                model.eval()
                _cache["gpt"] = (model, tok)
                log("GPT готов")
            except Exception as e:
                log(f"Ошибка загрузки GPT: {e}", "ERROR"); raise
    return _cache["gpt"]

def load_mnist():
    _require_torch()
    with _lock:
        if "mnist" not in _cache:
            log("Загрузка MNIST CNN…")
            try:
                model = ResNet().to(DEVICE)
                path  = BASE / "mnist_web.pth"
                if not path.exists(): path = BASE / "best_model_v3.pth"
                model.load_state_dict(torch.load(path, weights_only=True, map_location=DEVICE))
                model.eval()
                _cache["mnist"] = model
                log("MNIST CNN готов")
            except Exception as e:
                log(f"Ошибка загрузки MNIST: {e}", "ERROR"); raise
    return _cache["mnist"]

def load_sentiment():
    _require_torch()
    with _lock:
        if "sentiment" not in _cache:
            log("Загрузка Sentiment LSTM…")
            try:
                ckpt = torch.load(BASE / "sentiment_model.pth", weights_only=True, map_location=DEVICE)
                vocab_size = ckpt["embed.weight"].shape[0]
                model = SentimentLSTM(vocab_size).to(DEVICE)
                model.load_state_dict(ckpt)
                model.eval()
                _cache["sentiment"] = model
                log("Sentiment LSTM готов")
            except Exception as e:
                log(f"Ошибка загрузки Sentiment: {e}", "ERROR"); raise
    return _cache["sentiment"]

def load_translator(direction="en2ru"):
    _require_torch()
    key = f"trans_{direction}"
    with _lock:
        if key not in _cache:
            log(f"Загрузка Translator ({direction})…")
            try:
                from tokenizers import Tokenizer
                en_tok = Tokenizer.from_file(str(BASE / "bpe_en.json"))
                ru_tok = Tokenizer.from_file(str(BASE / "bpe_ru.json"))
                if direction == "en2ru":
                    src_tok, tgt_tok = en_tok, ru_tok
                    mpath = BASE / "translator_bpe_en2ru.pth"
                else:
                    src_tok, tgt_tok = ru_tok, en_tok
                    mpath = BASE / "translator_bpe_ru2en.pth"
                ckpt  = torch.load(mpath, weights_only=False, map_location=DEVICE)
                model = TranslatorBPE(src_tok.get_vocab_size(), tgt_tok.get_vocab_size()).to(DEVICE)
                model.load_state_dict(ckpt["model"])
                model.eval()
                _cache[key] = (model, src_tok, tgt_tok)
                log(f"Translator ({direction}) готов")
            except Exception as e:
                log(f"Ошибка загрузки Translator: {e}", "ERROR"); raise
    return _cache[key]

def load_gan():
    _require_torch()
    with _lock:
        if "gan" not in _cache:
            log("Загрузка GAN…")
            try:
                model = Generator().to(DEVICE)
                model.load_state_dict(torch.load(BASE / "gan_generator.pth", weights_only=False, map_location=DEVICE))
                model.eval()
                _cache["gan"] = model
                log("GAN готов")
            except Exception as e:
                log(f"Ошибка загрузки GAN: {e}", "ERROR"); raise
    return _cache["gan"]

def load_price():
    _require_torch()
    with _lock:
        if "price" not in _cache:
            log("Загрузка Price MLP…")
            try:
                model = PriceNet().to(DEVICE)
                model.load_state_dict(torch.load(BASE / "price_model.pth", weights_only=True, map_location=DEVICE))
                model.eval()
                _cache["price"] = model
                log("Price MLP готов")
            except Exception as e:
                log(f"Ошибка загрузки Price: {e}", "ERROR"); raise
    return _cache["price"]

def load_timeseries():
    _require_torch()
    with _lock:
        if "ts" not in _cache:
            log("Загрузка TimeSeries LSTM…")
            try:
                model = TSModel().to(DEVICE)
                model.load_state_dict(torch.load(BASE / "timeseries_model.pth", weights_only=True, map_location=DEVICE))
                model.eval()
                _cache["ts"] = model
                log("TimeSeries LSTM готов")
            except Exception as e:
                log(f"Ошибка загрузки TimeSeries: {e}", "ERROR"); raise
    return _cache["ts"]

def load_temperature():
    _require_torch()
    with _lock:
        if "temp" not in _cache:
            log("Загрузка TempNet…")
            try:
                ckpt = torch.load(BASE / "temperature_model.pth", weights_only=True, map_location=DEVICE)
                model = TempNet().to(DEVICE)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                _cache["temp"]        = model
                _cache["temp_X_mean"] = np.array(ckpt["X_mean"], dtype=np.float32)
                _cache["temp_X_std"]  = np.array(ckpt["X_std"],  dtype=np.float32)
                _cache["temp_Y_mean"] = float(ckpt["Y_mean"])
                _cache["temp_Y_std"]  = float(ckpt["Y_std"])
                log("TempNet готов")
            except Exception as e:
                log(f"Ошибка загрузки TempNet: {e}", "ERROR"); raise
    return _cache["temp"]

def load_spam():
    _require_torch()
    with _lock:
        if "spam" not in _cache:
            log("Загрузка SpamLSTM…")
            try:
                ckpt       = torch.load(BASE / "spam_model.pth", weights_only=True, map_location=DEVICE)
                vocab_size = ckpt["vocab_size"]
                model      = SpamLSTM(vocab_size, ckpt.get("embed_dim", 64), ckpt.get("hidden_dim", 128)).to(DEVICE)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                _cache["spam"]       = model
                _cache["spam_vocab"] = ckpt["vocab"]
                _cache["spam_maxlen"]= ckpt.get("max_len", 40)
                log("SpamLSTM готов")
            except Exception as e:
                log(f"Ошибка загрузки SpamLSTM: {e}", "ERROR"); raise
    return _cache["spam"]

def load_clustering():
    _require_torch()
    with _lock:
        if "cluster" not in _cache:
            log("Загрузка ClusterAutoencoder…")
            try:
                ckpt  = torch.load(BASE / "clustering_model.pth", weights_only=True, map_location=DEVICE)
                model = ClusterAutoencoder().to(DEVICE)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                _cache["cluster"]         = model
                _cache["cluster_centers"] = np.array(ckpt["kmeans_centers"], dtype=np.float32)
                _cache["cluster_mapping"] = {int(k): int(v) for k, v in ckpt["cluster_mapping"].items()}
                _cache["cluster_sc_mean"] = np.array(ckpt["scaler_mean"],  dtype=np.float32)
                _cache["cluster_sc_std"]  = np.array(ckpt["scaler_scale"], dtype=np.float32)
                log("ClusterAutoencoder готов")
            except Exception as e:
                log(f"Ошибка загрузки Cluster: {e}", "ERROR"); raise
    return _cache["cluster"]

def load_anomaly():
    _require_torch()
    with _lock:
        if "anomaly" not in _cache:
            log("Загрузка AnomalyAE…")
            try:
                ckpt  = torch.load(BASE / "anomaly_model.pth", weights_only=True, map_location=DEVICE)
                model = AnomalyAE().to(DEVICE)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                _cache["anomaly"]           = model
                _cache["anomaly_X_mean"]    = np.array(ckpt["X_mean"], dtype=np.float32)
                _cache["anomaly_X_std"]     = np.array(ckpt["X_std"],  dtype=np.float32)
                _cache["anomaly_threshold"] = float(ckpt["threshold"])
                log("AnomalyAE готов")
            except Exception as e:
                log(f"Ошибка загрузки AnomalyAE: {e}", "ERROR"); raise
    return _cache["anomaly"]

def load_defect():
    _require_torch()
    with _lock:
        if "defect" not in _cache:
            log("Загрузка DefectNet…")
            try:
                ckpt  = torch.load(BASE / "defect_model.pth", weights_only=True, map_location=DEVICE)
                model = DefectNet().to(DEVICE)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                _cache["defect"]        = model
                _cache["defect_X_mean"] = np.array(ckpt["X_mean"], dtype=np.float32)
                _cache["defect_X_std"]  = np.array(ckpt["X_std"],  dtype=np.float32)
                log("DefectNet готов")
            except Exception as e:
                log(f"Ошибка загрузки DefectNet: {e}", "ERROR"); raise
    return _cache["defect"]

def load_recommender():
    _require_torch()
    with _lock:
        if "rec" not in _cache:
            log("Загрузка RecommenderNCF…")
            try:
                ckpt = torch.load(BASE / "recommender_model.pth", weights_only=True, map_location=DEVICE)
                model = RecommenderNCF(ckpt["n_users"], ckpt["n_movies"], ckpt["embed_dim"]).to(DEVICE)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                _cache["rec"]              = model
                _cache["rec_vecs"]         = np.array(ckpt["movie_vecs"], dtype=np.float32)
                _cache["rec_names"]        = ckpt["movie_names"]
                _cache["rec_genres"]       = ckpt["movie_genre_str"]
                _cache["rec_genre_labels"] = ckpt["genres"]
                log("RecommenderNCF готов")
            except Exception as e:
                log(f"Ошибка загрузки Recommender: {e}", "ERROR"); raise
    return _cache["rec"]

def load_ner():
    _require_torch()
    with _lock:
        if "ner" not in _cache:
            log("Загрузка NERModel…")
            try:
                ckpt  = torch.load(BASE / "ner_model.pth", weights_only=True, map_location=DEVICE)
                vocab = ckpt["vocab"]
                model = NERModel(ckpt["vocab_size"], ckpt.get("embed_dim", 64),
                                 ckpt.get("hidden_dim", 128), len(ckpt["tags"])).to(DEVICE)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                _cache["ner"]         = model
                _cache["ner_vocab"]   = vocab
                _cache["ner_tags"]    = ckpt["tags"]
                _cache["ner_max_len"] = ckpt.get("max_len", 20)
                log("NERModel готов")
            except Exception as e:
                log(f"Ошибка загрузки NER: {e}", "ERROR"); raise
    return _cache["ner"]

# ═══════════════════════════════════════════════════════
# LM STUDIO КОНФИГ
# ═══════════════════════════════════════════════════════
def _load_lm_cfg():
    p = BASE / "lm_config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

LM_CFG  = _load_lm_cfg()
LM_URL  = LM_CFG.get("lm_studio_url", "http://localhost:1234/v1")
LM_MDLS = LM_CFG.get("models", {})
LM_KWS  = LM_CFG.get("routing_keywords", {})
_stop_ev = threading.Event()

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
    tmpl = BASE / "templates" / "app.html"
    if tmpl.exists():
        return tmpl.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html; charset=utf-8"}
    # Диагностика если файл не найден
    return f"<pre>app.html not found\nBASE={BASE}\nfrozen={getattr(sys,'frozen',False)}\n_MEIPASS={getattr(sys,'_MEIPASS','N/A')}\ntmpl={tmpl}</pre>", 500

@app.route("/api/logs")
def api_logs():
    return jsonify(list(_log_buf))

@app.route("/api/lm/logs")
def api_lm_logs():
    return jsonify(list(_lm_log_buf))

@app.route("/api/lm/models")
def api_lm_models():
    try:
        r = _req.get(f"{LM_URL}/models", timeout=2)
        if r.ok:
            data = r.json()
            models = [m.get("id", m.get("name", "")) for m in data.get("data", [])]
            return jsonify({"models": models, "online": True})
    except Exception:
        pass
    return jsonify({"models": [], "online": False, "error": "LM Studio недоступна"})

@app.route("/api/lm/config")
def api_lm_config():
    return jsonify({"models": LM_MDLS, "url": LM_URL})

@app.route("/api/lm/setup_check")
def lm_setup_check():
    """Проверяет статус LM Studio и нужных моделей для мастера настройки."""
    # Проверяем онлайн + загруженные модели
    try:
        r = _req.get(f"{LM_URL}/models", timeout=3)
        r.raise_for_status()
        loaded = [m["id"] for m in r.json().get("data", [])]
        online = True
    except Exception:
        loaded = []
        online = False

    # Скачанные модели (LM Studio v0 API)
    downloaded = []
    if online:
        try:
            base = LM_URL.rsplit("/v1", 1)[0]
            r2 = _req.get(f"{base}/api/v0/models", timeout=3)
            if r2.ok:
                downloaded = [m.get("id", m.get("path", "")) for m in r2.json()]
        except Exception:
            pass

    def _filename(path):
        """Возвращает только имя файла из пути/id модели."""
        return path.replace("\\", "/").split("/")[-1].lower()

    loaded_files = [_filename(ld) for ld in loaded]
    downloaded_files = [_filename(d) for d in downloaded]

    # Нужные модели из конфига
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

    all_ready = online and all(m["loaded"] for m in required)
    return jsonify({
        "online":     online,
        "loaded":     loaded,
        "required":   required,
        "all_loaded": all_ready,
    })


@app.route("/api/lm/download", methods=["POST"])
def lm_download():
    """Запускает скачивание модели через LM Studio API."""
    model_id = (request.json or {}).get("model_id", "").strip()
    if not model_id:
        return jsonify({"ok": False, "msg": "Не указан model_id"}), 400
    base = LM_URL.rsplit("/v1", 1)[0]
    try:
        r = _req.post(f"{base}/api/v0/models/download",
                      json={"model": model_id}, timeout=10)
        if r.ok:
            return jsonify({"ok": True, "msg": f"Скачивание начато: {model_id}"})
        # LM Studio может не поддерживать API скачивания — даём ссылку
        return jsonify({"ok": False, "msg": f"LM Studio вернула {r.status_code}. Скачай вручную в LM Studio.", "manual": True})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e), "manual": True})

@app.route("/api/lm/download_progress")
def lm_download_progress():
    """Статус текущего скачивания."""
    base = LM_URL.rsplit("/v1", 1)[0]
    try:
        r = _req.get(f"{base}/api/v0/models/download/progress", timeout=3)
        if r.ok:
            return jsonify({"ok": True, "data": r.json()})
    except Exception:
        pass
    return jsonify({"ok": False, "data": {}})

@app.route("/api/homework/solve", methods=["POST"])
def homework_solve():
    data    = request.json or {}
    text    = data.get("text", "").strip()
    image   = data.get("image", "")   # base64 data-url
    subject = data.get("subject", "")

    if not text and not image:
        return jsonify({"error": "Нет вопроса"}), 400

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

    # Проверка LM Studio
    try:
        _r = _req.get(f"{LM_URL}/models", timeout=2)
        _loaded = _r.json().get("data", []) if _r.ok else []
        if not _r.ok or not _loaded:
            return jsonify({"error": "В LM Studio нет загруженных моделей. Загрузите модель в LM Studio."}), 503
    except Exception:
        return jsonify({"error": "LM Studio офлайн. Запустите LM Studio и загрузите модель."}), 503

    try:
        if image:
            # Vision модель для фото
            model_id = LM_MDLS.get("vision", {}).get("model_id", "")
            if not model_id:
                model_id = LM_MDLS.get("reasoner", {}).get("model_id", "")
            if not model_id:
                model_id = _loaded[0]["id"] if _loaded else ""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text",      "text": user_msg},
                    {"type": "image_url", "image_url": {"url": image}},
                ]},
            ]
        else:
            model_id = LM_MDLS.get("reasoner", {}).get("model_id", "")
            if not model_id:
                model_id = _loaded[0]["id"] if _loaded else ""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ]

        r = _req.post(f"{LM_URL}/chat/completions", json={
            "model": model_id, "messages": messages,
            "temperature": 0.1, "max_tokens": 512, "stream": False,
        }, timeout=60)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        import re as _re
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()

        # Парсим: несколько ЗАДАНИЕ N: / одиночный ОТВЕТ:
        tasks_m = _re.findall(
            r"ЗАДАНИЕ\s*\d+[:\s]*\n?ОТВЕТ[:\s]+(.+?)(?:\nПОЯСНЕНИЕ[:\s]+(.+?))?(?=\n\nЗАДАНИЕ|\Z)",
            raw, _re.IGNORECASE | _re.DOTALL)
        if tasks_m:
            parts = []
            for i, (ans, exp) in enumerate(tasks_m, 1):
                parts.append(f"Задание {i}: {ans.strip()}")
            answer = "\n".join(parts)
            explanation = "\n".join(exp.strip() for _, exp in tasks_m if exp.strip())
        else:
            ans_m = _re.search(r"ОТВЕТ[:\s]+(.+?)(?:\n|ПОЯСНЕНИЕ|$)", raw, _re.IGNORECASE)
            exp_m = _re.search(r"ПОЯСНЕНИЕ[:\s]+(.+)", raw, _re.IGNORECASE | _re.DOTALL)
            answer      = ans_m.group(1).strip() if ans_m else raw.split("\n")[0]
            explanation = exp_m.group(1).strip() if exp_m else ""
        return jsonify({"answer": answer, "explanation": explanation, "raw": raw})

    except _req.exceptions.ConnectionError:
        return jsonify({"error": "LM Studio офлайн"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/status")
def api_status():
    files = {
        "gpt":         (BASE / "gpt_multilingual.pth").exists() or (BASE / "gpt_bpe_model.pth").exists(),
        "mnist":       (BASE / "mnist_web.pth").exists() or (BASE / "best_model_v3.pth").exists(),
        "sentiment":   (BASE / "sentiment_model.pth").exists(),
        "translator":  (BASE / "translator_bpe_en2ru.pth").exists(),
        "gan":         (BASE / "gan_generator.pth").exists(),
        "price":       (BASE / "price_model.pth").exists(),
        "timeseries":  (BASE / "timeseries_model.pth").exists(),
        "temperature": (BASE / "temperature_model.pth").exists(),
        "spam":        (BASE / "spam_model.pth").exists(),
        "clustering":  (BASE / "clustering_model.pth").exists(),
        "anomaly":     (BASE / "anomaly_model.pth").exists(),
        "defect":      (BASE / "defect_model.pth").exists(),
        "ner":         (BASE / "ner_model.pth").exists(),
        "recommender": (BASE / "recommender_model.pth").exists(),
    }
    lm_ok = False
    try:
        r = _req.get(f"{LM_URL}/models", timeout=2)
        lm_ok = r.ok
    except Exception:
        pass
    return jsonify({"device": str(DEVICE), "models": files, "lm_studio": lm_ok,
                    "torch_available": TORCH_AVAILABLE, "version": "2.0"})

# ── Математика ──
@app.route("/api/math/solve", methods=["POST"])
def math_solve():
    expr = request.json.get("expression", "")
    try:
        import ast, operator as _op
        clean = re.sub(r"[^0-9+\-*/().\s]", "", expr)
        if not clean.strip(): raise ValueError("Пустое выражение")
        _OPS = {ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
                ast.Div: _op.truediv, ast.USub: _op.neg}
        def _eval(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
                return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
                return _OPS[type(node.op)](_eval(node.operand))
            raise ValueError("Недопустимая операция")
        result = _eval(ast.parse(clean, mode='eval').body)
        return jsonify({"expression": expr, "result": round(float(result), 8)})
    except ZeroDivisionError:
        return jsonify({"error": "Деление на ноль"})
    except Exception as e:
        return jsonify({"error": str(e)})

# ── MNIST ──
@app.route("/api/mnist/predict", methods=["POST"])
def mnist_predict():
    try:
        import PIL.Image
        data    = request.json
        img_b64 = data.get("image", "")
        img     = PIL.Image.open(__import__("io").BytesIO(base64.b64decode(img_b64.split(",")[-1] if "," in img_b64 else img_b64))).convert("L")
        arr     = np.array(img, dtype=np.uint8)

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
        img    = PIL.Image.fromarray(arr).resize((new_w, new_h), PIL.Image.LANCZOS)
        canvas = np.zeros((28, 28), dtype=np.float32)
        y0 = (28 - new_h) // 2; x0 = (28 - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = np.array(img, dtype=np.float32)

        x = torch.tensor((canvas / 255.0 - 0.1307) / 0.3081).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(load_mnist()(x), dim=1)[0].tolist()
        pred = int(np.argmax(probs))
        return jsonify({"digit": pred, "confidence": round(max(probs) * 100, 1),
                        "probs": [round(p * 100, 1) for p in probs]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Тональность ──
@app.route("/api/sentiment/analyze", methods=["POST"])
def sentiment_analyze():
    try:
        text = request.json.get("text", "").lower()
        MAX  = 20
        ids  = [SENT_VOCAB.get(w, 1) for w in text.split()][:MAX]
        ids += [0] * (MAX - len(ids))
        x    = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(load_sentiment()(x), dim=1)[0].tolist()
        labels = ["негативный", "нейтральный", "позитивный"]
        emojis = ["😠", "😐", "😊"]
        pred   = int(np.argmax(probs))
        return jsonify({"label": labels[pred], "emoji": emojis[pred],
                        "scores": {l: round(p * 100, 1) for l, p in zip(labels, probs)}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Перевод ──
@app.route("/api/translate", methods=["POST"])
def translate():
    data      = request.json
    text      = data.get("text", "").strip()
    direction = data.get("direction", "en2ru")
    try:
        model, src_tok, tgt_tok = load_translator(direction)
        MAX = 40; PAD, SOS, EOS = 0, 1, 2
        ids = [SOS] + src_tok.encode(text).ids[:MAX - 2] + [EOS]
        ids += [PAD] * (MAX - len(ids))
        src = torch.tensor([ids[:MAX]], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            se  = model.pos_enc(model.src_embed(src) * math.sqrt(model.d))
            mem = model.transformer.encoder(se)
            tgt_ids = [SOS]
            for _ in range(MAX):
                tgt = torch.tensor([tgt_ids], dtype=torch.long).to(DEVICE)
                te  = model.pos_enc(model.tgt_embed(tgt) * math.sqrt(model.d))
                T   = tgt.size(1)
                tm  = torch.triu(torch.ones(T, T, device=DEVICE), diagonal=1).bool()
                out = model.transformer.decoder(te, mem, tgt_mask=tm)
                nid = model.fc(out[:, -1]).argmax(-1).item()
                if nid == EOS: break
                tgt_ids.append(nid)
        result = tgt_tok.decode(tgt_ids[1:])
        return jsonify({"translation": result or "(пустой результат)"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── GPT стриминг ──
@app.route("/api/gpt/stream", methods=["POST"])
def gpt_stream():
    data        = request.json
    seed        = data.get("seed", "[RU] Однажды")
    length      = min(int(data.get("length", 200)), 400)
    temperature = float(data.get("temperature", 0.8))
    try:
        model, tok = load_gpt()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    CTX  = 128
    ids  = list(tok.encode(seed).ids[-CTX:])
    def gen():
        yield f"data: {json.dumps({'text': seed})}\n\n"
        with torch.no_grad():
            for _ in range(length):
                x      = torch.tensor([ids[-CTX:]], dtype=torch.long).to(DEVICE)
                logits = model(x)[0, -1] / temperature
                probs  = torch.softmax(logits, 0)
                nid    = torch.multinomial(probs, 1).item()
                ids.append(nid)
                yield f"data: {json.dumps({'text': tok.decode([nid])})}\n\n"
        yield 'data: {"done":true}\n\n'
    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── GAN ──
@app.route("/api/gan/generate", methods=["POST"])
def gan_generate():
    try:
        import PIL.Image, io as _io
        data   = request.json or {}
        digit  = data.get("digit", -1)
        count  = min(int(data.get("count", 4)), 16)
        model  = load_gan()
        z      = torch.randn(count, 100).to(DEVICE)
        labels = (torch.randint(0, 10, (count,)) if digit == -1
                  else torch.full((count,), int(digit))).to(DEVICE)
        with torch.no_grad(): imgs = model(z, labels).cpu()
        results = []
        for i in range(count):
            img = ((imgs[i, 0].numpy() + 1) / 2 * 255).astype(np.uint8)
            pil = PIL.Image.fromarray(img).resize((140, 140), PIL.Image.BICUBIC)
            buf = _io.BytesIO(); pil.save(buf, format="PNG")
            results.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
        return jsonify({"images": results, "digit": digit})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Цена квартиры ──
@app.route("/api/price/predict", methods=["POST"])
def price_predict():
    try:
        d      = request.json
        area   = float(d.get("area", 60))
        rooms  = float(d.get("rooms", 2))
        floor  = float(d.get("floor", 5))
        dist   = float(d.get("district", 2))
        age    = float(d.get("age", 10))
        feat   = np.array([area, rooms, floor, dist, age], dtype=np.float32)
        fn     = (feat - PRICE_X_MEAN) / PRICE_X_STD
        inp    = torch.tensor(fn).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # Модель обучена на данных где ppm2 в руб/м² без *1000,
            # масштабируем обратно чтобы получить реалистичные цены
            p = (load_price()(inp).item() * PRICE_Y_STD + PRICE_Y_MEAN) * 1000
        return jsonify({"price": round(p, 1), "price_str": f"{p:.1f} млн ₽"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Прогноз продаж ──
@app.route("/api/timeseries/forecast", methods=["POST"])
def ts_forecast():
    try:
        d      = request.json or {}
        vals   = d.get("values", [])
        window = np.array(vals, dtype=np.float32) if len(vals) == 30 else TS_SALES[-30:]
        wn     = (window - TS_MEAN) / TS_STD
        x      = torch.tensor(wn).unsqueeze(0).unsqueeze(-1).to(DEVICE)
        with torch.no_grad():
            p = load_timeseries()(x)[0].cpu().numpy() * TS_STD + TS_MEAN
        return jsonify({"forecast": [round(float(v), 1) for v in p],
                        "mean": round(float(p.mean()), 1)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Прогноз температуры ──
@app.route("/api/temperature/predict", methods=["POST"])
def temperature_predict():
    try:
        d    = request.json or {}
        feat = np.array([
            float(d.get("temp_today", 10)),
            float(d.get("pressure",   760)),
            float(d.get("humidity",   60)),
            float(d.get("wind",       3)),
            float(d.get("cloud",      0.5)),
            float(d.get("month",      6)),
        ], dtype=np.float32)
        load_temperature()
        Xm = _cache["temp_X_mean"]
        Xs = _cache["temp_X_std"]
        Ym = _cache["temp_Y_mean"]
        Ys = _cache["temp_Y_std"]
        fn  = (feat - Xm) / Xs
        inp = torch.tensor(fn).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            result = _cache["temp"](inp).item() * Ys + Ym
        return jsonify({"temp_tomorrow": round(result, 1)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Обнаружение спама ──
@app.route("/api/spam/analyze", methods=["POST"])
def spam_analyze():
    try:
        text = (request.json or {}).get("text", "").strip()
        if not text:
            return jsonify({"error": "Пустой текст"}), 400
        load_spam()
        vocab   = _cache["spam_vocab"]
        max_len = _cache["spam_maxlen"]
        ids = [vocab.get(w, 1) for w in text.lower().split()[:max_len]]
        ids += [0] * (max_len - len(ids))
        inp = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(_cache["spam"](inp), dim=1)[0].cpu().numpy()
        spam_prob = float(probs[1])
        ham_prob  = float(probs[0])
        label = "SPAM" if spam_prob > 0.5 else "HAM"
        return jsonify({
            "label":     label,
            "spam_prob": round(spam_prob * 100, 1),
            "ham_prob":  round(ham_prob  * 100, 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    try:
        d    = request.json or {}
        feat = np.array([
            float(d.get("age",    35)),
            float(d.get("orders", 10)),
            float(d.get("avg",  3000)),
            float(d.get("total",  30)),
            float(d.get("days",   20)),
            float(d.get("visits", 10)),
        ], dtype=np.float32)
        load_clustering()
        sc_mean = _cache["cluster_sc_mean"]
        sc_std  = _cache["cluster_sc_std"]
        feat_n  = (feat - sc_mean) / sc_std
        inp     = torch.tensor(feat_n).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, z = _cache["cluster"](inp)
            z_np = z.cpu().numpy()
        centers = _cache["cluster_centers"]
        dists   = np.linalg.norm(centers - z_np, axis=1)
        cluster = int(dists.argmin())
        mapping = _cache["cluster_mapping"]
        seg_id  = mapping.get(cluster, 0)
        seg     = _SEG_NAMES.get(seg_id, f"Группа {cluster}")
        return jsonify({
            "segment":     seg,
            "cluster":     cluster,
            "confidence":  round(float(1 / (1 + dists.min())), 3),
            "all_dists":   [round(float(d), 3) for d in dists],
            "rec":         _SEG_RECS.get(seg, "—"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Обнаружение аномалий (транзакции) ──
@app.route("/api/anomaly/check", methods=["POST"])
def anomaly_check():
    try:
        d    = request.json or {}
        feat = np.array([
            float(d.get("amount",   500)),
            float(d.get("hour",      14)),
            float(d.get("freq",       3)),
            float(d.get("foreign",    0)),
            float(d.get("online",     1)),
            float(d.get("balance", 50000)),
            float(d.get("distance",   2)),
        ], dtype=np.float32)
        load_anomaly()
        Xm  = _cache["anomaly_X_mean"]
        Xs  = _cache["anomaly_X_std"]
        thr = _cache["anomaly_threshold"]
        fn  = (feat - Xm) / Xs
        inp = torch.tensor(fn).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            rec = _cache["anomaly"](inp)
            err = float(((inp - rec) ** 2).mean().item())
        risk  = round(err / thr, 2)
        if   risk > 1.5: label = "ВЫСОКИЙ РИСК"
        elif risk > 1.0: label = "Подозрительно"
        else:            label = "Норма"
        return jsonify({
            "label":     label,
            "risk":      risk,
            "recon_err": round(err, 5),
            "threshold": round(thr, 5),
            "risk_pct":  min(100, round(risk * 50)),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Контроль качества (дефекты) ──
@app.route("/api/defect/check", methods=["POST"])
def defect_check():
    try:
        d    = request.json or {}
        feat = np.array([
            float(d.get("thickness",  10.0)),
            float(d.get("mass",      250.0)),
            float(d.get("hardness",  200.0)),
            float(d.get("roughness",   1.6)),
            float(d.get("length",    100.0)),
            float(d.get("width",      50.0)),
            float(d.get("temp",      850.0)),
            float(d.get("time",      120.0)),
        ], dtype=np.float32)
        load_defect()
        Xm  = _cache["defect_X_mean"]
        Xs  = _cache["defect_X_std"]
        fn  = (feat - Xm) / Xs
        inp = torch.tensor(fn).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(_cache["defect"](inp), dim=1)[0].cpu().numpy()
        defect_prob = float(probs[1])
        label = "БРАК" if defect_prob > 0.5 else "НОРМА"
        return jsonify({
            "label":       label,
            "defect_prob": round(defect_prob * 100, 1),
            "ok_prob":     round(float(probs[0]) * 100, 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── NER ──
@app.route("/api/ner/analyze", methods=["POST"])
def ner_analyze():
    try:
        text = (request.json or {}).get("text", "").strip()
        if not text:
            return jsonify({"error": "Пустой текст"}), 400
        load_ner()
        vocab   = _cache["ner_vocab"]
        tags    = _cache["ner_tags"]
        max_len = _cache["ner_max_len"]
        tokens  = text.split()[:max_len]
        ids     = [vocab.get(t.lower(), 1) for t in tokens]
        ids     = ids + [0] * max(0, max_len - len(ids))
        inp     = torch.tensor([ids]).to(DEVICE)
        with torch.no_grad():
            logits = _cache["ner"](inp)[0][:len(tokens)]
            preds  = logits.argmax(-1).cpu().numpy()
        result = [{"token": t, "tag": tags[p]} for t, p in zip(tokens, preds)]
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Extractive Summarization ──
@app.route("/api/summarize", methods=["POST"])
def summarize():
    try:
        d       = request.json or {}
        text    = d.get("text", "").strip()
        n_sents = int(d.get("n_sentences", 3))
        if not text:
            return jsonify({"error": "Пустой текст"}), 400
        # Разбиваем на предложения
        import re as _re
        sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
        if len(sentences) <= n_sents:
            return jsonify({"summary": text, "selected": list(range(len(sentences))), "sentences": sentences})
        # TF-IDF scoring
        stop = {"the","a","an","is","was","are","were","be","been","being","to","of","in","for",
                "on","with","at","by","from","that","this","it","as","or","and","but","not",
                "have","has","had","will","would","could","should","may","might","do","did","does",
                "в","и","на","с","по","за","из","к","у","от","до","при","о","что","не","он","она"}
        # Word frequencies across all sentences
        word_freq = Counter()
        for s in sentences:
            for w in _re.findall(r'\b\w+\b', s.lower()):
                if w not in stop and len(w) > 2:
                    word_freq[w] += 1
        max_freq = max(word_freq.values()) if word_freq else 1
        # Score each sentence
        scores = []
        for i, s in enumerate(sentences):
            words = [w for w in _re.findall(r'\b\w+\b', s.lower()) if w not in stop and len(w) > 2]
            score = sum(word_freq.get(w, 0) / max_freq for w in words) / max(len(words), 1)
            # Boost first sentence slightly (often topic sentence)
            if i == 0: score *= 1.2
            scores.append((score, i))
        scores.sort(reverse=True)
        selected = sorted([idx for _, idx in scores[:n_sents]])
        summary  = " ".join(sentences[i] for i in selected)
        return jsonify({"summary": summary, "selected": selected, "sentences": sentences})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Кластеризация документов ──
@app.route("/api/cluster/documents", methods=["POST"])
def cluster_documents():
    try:
        d    = request.json or {}
        docs = [s.strip() for s in d.get("documents", []) if str(s).strip()]
        k    = int(d.get("k", 3))
        if len(docs) < 2:
            return jsonify({"error": "Нужно минимум 2 документа"}), 400
        k = min(k, len(docs))
        import re as _re
        stop = {"the","a","an","is","was","are","were","be","been","to","of","in","for",
                "on","with","at","by","from","that","this","it","as","or","and","but","not",
                "have","has","had","will","would","could","should","do","did","does","i","you",
                "he","she","we","they","в","и","на","с","по","за","из","к","у","от","до","о","что","не"}
        # Build vocab
        tokenize = lambda t: [w for w in _re.findall(r'\b\w+\b', t.lower()) if w not in stop and len(w) > 2]
        tok_docs = [tokenize(d_) for d_ in docs]
        vocab = list({w for td in tok_docs for w in td})
        if not vocab:
            return jsonify({"error": "Документы не содержат значимых слов"}), 400
        v2i = {w: i for i, w in enumerate(vocab)}
        # TF-IDF
        n, V = len(docs), len(vocab)
        tf  = np.zeros((n, V), dtype=np.float32)
        for i, td in enumerate(tok_docs):
            if td:
                cnt = Counter(td)
                for w, c in cnt.items():
                    tf[i, v2i[w]] = c / len(td)
        df  = (tf > 0).sum(axis=0) + 1
        idf = np.log(n / df)
        tfidf = tf * idf
        # L2 normalize
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True) + 1e-8
        tfidf = tfidf / norms
        # K-Means (numpy)
        np.random.seed(42)
        centers = tfidf[np.random.choice(n, k, replace=False)]
        labels  = np.zeros(n, dtype=int)
        for _ in range(50):
            dists  = np.linalg.norm(tfidf[:, None] - centers[None], axis=2)
            new_lb = dists.argmin(axis=1)
            if (new_lb == labels).all(): break
            labels = new_lb
            for c in range(k):
                m = tfidf[labels == c]
                if len(m): centers[c] = m.mean(axis=0)
        # Top keywords per cluster
        topic_names = []
        for c in range(k):
            mask = labels == c
            if mask.any():
                center_vec = centers[c]
                top_idx = center_vec.argsort()[-5:][::-1]
                kws = [vocab[i] for i in top_idx if center_vec[i] > 0]
                topic_names.append(", ".join(kws[:3]) if kws else f"Тема {c+1}")
            else:
                topic_names.append(f"Тема {c+1}")
        result = [{"doc": docs[i], "cluster": int(labels[i]),
                   "topic": topic_names[int(labels[i])]} for i in range(n)]
        return jsonify({"clusters": result, "k": k, "topics": topic_names})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Рекомендации ──
@app.route("/api/recommender/catalog", methods=["GET"])
def recommender_catalog():
    try:
        load_recommender()
        return jsonify({
            "movies": [
                {"name": n, "genre": g}
                for n, g in zip(_cache["rec_names"], _cache["rec_genres"])
            ],
            "genres": _cache["rec_genre_labels"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/recommender/recommend", methods=["POST"])
def recommender_recommend():
    try:
        d       = request.json or {}
        liked   = d.get("liked", [])   # список названий фильмов
        top_n   = int(d.get("top_n", 6))
        load_recommender()
        names = _cache["rec_names"]
        vecs  = _cache["rec_vecs"]
        name2idx = {n: i for i, n in enumerate(names)}
        liked_idx = [name2idx[n] for n in liked if n in name2idx]
        if not liked_idx:
            return jsonify({"error": "Фильмы не найдены в каталоге"}), 400
        mean_vec = vecs[liked_idx].mean(axis=0)
        scores   = vecs @ mean_vec
        scores[liked_idx] = -np.inf
        top_idx  = np.argsort(scores)[::-1][:top_n]
        result   = [{"name": names[i], "genre": _cache["rec_genres"][i],
                     "score": round(float(scores[i]), 3)} for i in top_idx]
        return jsonify({"recommendations": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── AI чат (LM Studio) ──
@app.route("/api/chat", methods=["POST"])
def chat():
    data         = request.json
    message      = data.get("message", "").strip()
    forced_model = data.get("model", "auto")
    image_b64    = data.get("image", None)
    if not message:
        return jsonify({"error": "Пустое сообщение"}), 400

    image_path = None
    if image_b64:
        try:
            img_bytes  = base64.b64decode(image_b64.split(",")[-1] if "," in image_b64 else image_b64)
            image_path = str(BASE / "_upload_tmp.jpg")
            with open(image_path, "wb") as f: f.write(img_bytes)
            forced_model = "vision"
        except Exception as e:
            return jsonify({"error": f"Ошибка декодирования изображения: {e}"}), 400

    chosen    = forced_model if forced_model != "auto" else _route_model(message)
    model_cfg = LM_MDLS.get(chosen, LM_MDLS.get("reasoner", {}))
    model_id  = model_cfg.get("model_id", "")

    # Проверяем доступность LM Studio и наличие моделей
    def _lm_check_error():
        try:
            r = _req.get(f"{LM_URL}/models", timeout=2)
            if not r.ok:
                return "LM Studio не отвечает. Запустите LM Studio."
            loaded = r.json().get("data", [])
            if not loaded:
                return "В LM Studio нет загруженных моделей. Загрузите модель в LM Studio."
            if not model_id:
                return "Модель не настроена. Открой настройки → lm_config.json и укажи model_id."
        except Exception:
            return "LM Studio офлайн. Запустите LM Studio и загрузите модель."
        return None

    err = _lm_check_error()
    if err:
        def _err_gen():
            yield f"data: {json.dumps({'type':'error','text':err})}\n\n"
        return Response(stream_with_context(_err_gen()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    log(f"[CHAT] модель={chosen} | {message[:60]}")
    lm_log(f"▶ Запрос | модель: {chosen} ({model_id}) | {'📎 фото + ' if image_path else ''}{message[:80]}")

    def generate():
        import time
        t0 = time.time()
        token_count = [0]
        try:
            yield f"data: {json.dumps({'type':'model','text':chosen})}\n\n"
            msgs = []
            sp   = model_cfg.get("system_prompt", "")
            if sp: msgs.append({"role": "system", "content": sp})
            if image_path and chosen == "vision":
                with open(image_path, "rb") as f: b64 = base64.b64encode(f.read()).decode()
                msgs.append({"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": message}]})
            else:
                msgs.append({"role": "user", "content": message})

            _stop_ev.clear()
            resp = _req.post(f"{LM_URL}/chat/completions",
                             json={"model": model_id, "messages": msgs, "stream": True,
                                   "temperature": model_cfg.get("temperature", 0.7),
                                   "max_tokens": model_cfg.get("max_tokens", 2048)},
                             stream=True, timeout=180)
            if not resp.ok:
                lm_log(f"✗ Ошибка HTTP {resp.status_code} от LM Studio", "ERROR")
                yield f"data: {json.dumps({'type':'error','text':f'LM Studio {resp.status_code}'})}\n\n"
                return

            lm_log(f"↳ Соединение OK ({int((time.time()-t0)*1000)} мс) — генерация…")
            try:
                for line in resp.iter_lines():
                    if _stop_ev.is_set():
                        lm_log(f"⏹ Остановлено пользователем ({token_count[0]} токенов)")
                        break
                    if not line: continue
                    line = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk.strip() == "[DONE]": break
                        try:
                            tok = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                            if tok:
                                token_count[0] += 1
                                yield f"data: {json.dumps({'type':'token','text':tok})}\n\n"
                        except Exception: pass
            finally:
                resp.close()

            elapsed = round(time.time() - t0, 1)
            lm_log(f"✓ Готово | {token_count[0]} токенов | {elapsed} с | {round(token_count[0]/elapsed,1) if elapsed else '?'} tok/s")
            if image_path and Path(image_path).exists(): os.remove(image_path)
            yield 'data: {"type":"done"}\n\n'
        except _req.exceptions.ConnectionError:
            lm_log("✗ LM Studio недоступна (ConnectionError)", "ERROR")
            yield f"data: {json.dumps({'type':'error','text':'LM Studio не запущена'})}\n\n"
        except Exception as e:
            lm_log(f"✗ Исключение: {e}", "ERROR")
            yield f"data: {json.dumps({'type':'error','text':str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/stop", methods=["POST"])
def stop():
    _stop_ev.set()
    return jsonify({"ok": True})

@app.route("/api/unload", methods=["POST"])
def unload_all():
    """Выгружает все модели из LM Studio при закрытии приложения."""
    base = LM_URL.rsplit("/v1", 1)[0]
    try:
        # Получаем список загруженных моделей
        r = _req.get(f"{LM_URL}/models", timeout=3)
        if not r.ok:
            return jsonify({"ok": True, "msg": "LM Studio офлайн"})
        loaded = [m["id"] for m in r.json().get("data", [])]
        # Выгружаем каждую
        for model_id in loaded:
            try:
                _req.post(f"{base}/api/v0/models/unload",
                          json={"identifier": model_id}, timeout=5)
            except Exception:
                pass
        return jsonify({"ok": True, "unloaded": loaded})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})

# ── Управление окном (пишем файл-команду, Qt читает по таймеру) ──
# В frozen-режиме exe находится на уровень выше _MEIPASS, совпадает с EXE_DIR в main.py
_WIN_CMD = (Path(os.environ["ALEXGPT_WIN_CMD"]) if "ALEXGPT_WIN_CMD" in os.environ
            else (Path(sys.executable).parent if getattr(sys, 'frozen', False) else BASE) / ".win_cmd")

@app.route("/api/win/<action>", methods=["POST"])
def win_control(action):
    if action in ("minimize", "maximize", "hide", "quit", "startmove") or action.startswith("startresize_"):
        try:
            _WIN_CMD.write_text(action)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"ok": True})
    return jsonify({"error": "unknown action"}), 400

@app.route("/api/lm/unload", methods=["POST"])
def lm_unload():
    try:
        r = _req.get(f"{LM_URL}/models", timeout=3)
        if not r.ok:
            return jsonify({"ok": False, "msg": "LM Studio недоступен"})
        models = r.json().get("data", [])
        if not models:
            return jsonify({"ok": True, "msg": "Нет загруженных моделей"})
        base = LM_URL.rsplit("/v1", 1)[0]
        names = [m.get("id", "?") for m in models]
        # Пробуем DELETE /api/v0/models/loaded (LM Studio 0.3+)
        for mid in names:
            try:
                _req.delete(f"{base}/api/v0/models/loaded/{mid}", timeout=5)
            except Exception:
                pass
        # Проверяем, выгрузилось ли
        r2 = _req.get(f"{LM_URL}/models", timeout=3)
        still = r2.json().get("data", []) if r2.ok else models
        if len(still) < len(models):
            return jsonify({"ok": True, "msg": f"Выгружено: {', '.join(names)}"})
        lm_log(f"API выгрузки не поддерживается — модели: {', '.join(names)}", "WARNING")
        return jsonify({"ok": False, "msg": f"Выгрузи вручную в LM Studio: {', '.join(names)}"})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})

@app.route("/api/unload/<model>", methods=["POST"])
def unload_model(model):
    with _lock:
        keys = list(_cache.keys())
        removed = []
        if model == "all":
            removed = keys
            _cache.clear()
        elif model in _cache:
            del _cache[model]
            removed = [model]
        # Для переводчика два ключа
        for k in [f"trans_en2ru", f"trans_ru2en"]:
            if model == "translator" and k in _cache:
                del _cache[k]; removed.append(k)
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
            log("Загрузка SDXL-Turbo (diffusers)…")
            try:
                from diffusers import AutoPipelineForText2Image
            except ImportError:
                import subprocess as _sp
                _sp.run([sys.executable, "-m", "pip", "install", "diffusers", "transformers",
                         "accelerate", "-q"], check=False)
                from diffusers import AutoPipelineForText2Image
            dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
            kw = {"torch_dtype": dtype}
            if DEVICE.type == "cuda":
                kw["variant"] = "fp16"
            _draw_pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo", **kw
            ).to(DEVICE)
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
    prompt = request.json.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Пустой промпт"}), 400
    _draw_stop = False
    try:
        pipe   = load_draw()
        steps  = 1 if DEVICE.type == "cuda" else 4

        def _check_stop(pipe, step, timestep, kwargs):
            if _draw_stop:
                raise InterruptedError("Остановлено пользователем")
            return kwargs

        result = pipe(prompt=prompt, num_inference_steps=steps,
                      guidance_scale=0.0, width=512, height=512,
                      callback_on_step_end=_check_stop)
        if _draw_stop:
            return jsonify({"error": "Остановлено"}), 200
        img    = result.images[0]
        buf    = io.BytesIO()
        img.save(buf, format="PNG")
        b64    = base64.b64encode(buf.getvalue()).decode()
        return jsonify({"image": f"data:image/png;base64,{b64}"})
    except InterruptedError:
        return jsonify({"error": "Остановлено"}), 200
    except Exception as e:
        log(f"draw error: {e}", "ERROR")
        return jsonify({"error": str(e)}), 500

# ── Coder Team ──
_coder_v1_proc = None

@app.route("/api/coder_team/run", methods=["POST"])
def coder_team():
    global _coder_v1_proc
    import subprocess as sp
    task = request.json.get("task", "").strip()
    if not task: return jsonify({"error": "Задача не указана"}), 400

    def generate():
        global _coder_v1_proc
        proc = sp.Popen([sys.executable, "-u", str(BASE / "coder_team.py"), task],
                        stdout=sp.PIPE, stderr=sp.STDOUT,
                        text=True, encoding="utf-8", errors="replace", cwd=str(BASE))
        _coder_v1_proc = proc
        for line in proc.stdout:
            yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
        proc.wait()
        _coder_v1_proc = None
        out = BASE / "coder_output.py"
        if out.exists():
            yield f"data: {json.dumps({'done':True,'code':out.read_text(encoding='utf-8')})}\n\n"
        else:
            yield 'data: {"done":true}\n\n'

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/coder_team/stop", methods=["POST"])
def coder_team_stop():
    global _coder_v1_proc
    if _coder_v1_proc:
        _coder_v1_proc.terminate()
        _coder_v1_proc = None
    return jsonify({"ok": True})


@app.route("/api/coder_team_v2/review", methods=["POST"])
def coder_team_v2_review():
    import subprocess as sp, tempfile, os
    code      = (request.json or {}).get("code", "").strip()
    task_desc = (request.json or {}).get("task", "").strip()
    iters_raw = int((request.json or {}).get("iterations", 3))
    iterations = str(999 if iters_raw == 0 else min(max(iters_raw, 1), 16))
    if not code: return jsonify({"error": "Код не передан"}), 400

    # Пишем код во временный файл
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
    tmp.write(code); tmp.close()

    def generate():
        try:
            proc = sp.Popen(
                [sys.executable, "-u", str(BASE / "coder_team_v2.py"),
                 "--review", tmp.name, task_desc, ],
                stdout=sp.PIPE, stderr=sp.STDOUT,
                text=True, encoding="utf-8", errors="replace", cwd=str(BASE),
                env={**os.environ, "PYTHONUNBUFFERED": "1",
                     "CODER_V2_MAX_ITER": iterations}
            )
            for line in proc.stdout:
                yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
            proc.wait()
            out = BASE / "coder_output_v2.py"
            if out.exists():
                yield f"data: {json.dumps({'done':True,'code':out.read_text(encoding='utf-8')})}\n\n"
            else:
                yield 'data: {"done":true}\n\n'
        finally:
            try: os.unlink(tmp.name)
            except: pass

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/coder_v2/control", methods=["POST"])
def coder_v2_control():
    """Записывает команду управления для активного цикла Coder Team v2.
    Тело: {"action": "stop"} или {"action": "comment", "text": "..."}
    """
    data = request.json or {}
    action = data.get("action", "").strip()
    if action not in ("stop", "comment"):
        return jsonify({"error": "action must be 'stop' or 'comment'"}), 400
    ctrl = {"action": action}
    if action == "comment":
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "text required for comment"}), 400
        ctrl["text"] = text
    ctrl_path = BASE / "coder_v2_control.json"
    ctrl_path.write_text(json.dumps(ctrl, ensure_ascii=False), encoding="utf-8")
    return jsonify({"ok": True})


@app.route("/api/coder_team_v2/run", methods=["POST"])
def coder_team_v2():
    import subprocess as sp
    task       = request.json.get("task", "").strip()
    iters_raw  = int(request.json.get("iterations", 4))
    # 0 = без ограничений (большое число), иначе зажимаем 1..16
    iterations = str(999 if iters_raw == 0 else min(max(iters_raw, 1), 16))
    if not task: return jsonify({"error": "Задача не указана"}), 400

    def generate():
        proc = sp.Popen([sys.executable, "-u", str(BASE / "coder_team_v2.py"), task, iterations],
                        stdout=sp.PIPE, stderr=sp.STDOUT,
                        text=True, encoding="utf-8", errors="replace", cwd=str(BASE))
        for line in proc.stdout:
            yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
        proc.wait()
        out = BASE / "coder_output_v2.py"
        if out.exists():
            yield f"data: {json.dumps({'done':True,'code':out.read_text(encoding='utf-8')})}\n\n"
        else:
            yield 'data: {"done":true}\n\n'

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _lm_monitor():
    """Фоновый поток: опрашивает LM Studio каждые 30 сек и пишет статус в lm_log."""
    import time
    prev_models = None
    prev_online = None
    while True:
        try:
            r = _req.get(f"{LM_URL}/models", timeout=3)
            online = r.ok
            models = [m.get("id","?") for m in r.json().get("data",[])] if online else []
        except Exception:
            online = False
            models = []

        if online != prev_online:
            if online:
                lm_log("🟢 LM Studio онлайн")
            else:
                lm_log("🔴 LM Studio офлайн или недоступна", "WARNING")
            prev_online = online

        if online and models != prev_models:
            if models:
                lm_log(f"📦 Загружено моделей: {len(models)}")
                for m in models:
                    lm_log(f"   • {m}")
            else:
                lm_log("📭 Нет загруженных моделей")
            prev_models = models

        time.sleep(30 if prev_online is not None else 1)


def run(port=5050):
    log(f"BASE={BASE} | templates exists: {(BASE / 'templates' / 'app.html').exists()}")
    log(f"Устройство: {DEVICE}")
    log(f"Сервер запущен на порту {port}")
    cuda_info = f"CUDA: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CUDA недоступна, используется CPU"
    log(cuda_info)
    # Проверяем наличие файлов моделей
    model_files = {
        "MiniGPT": BASE / "gpt_multilingual.pth",
        "MNIST CNN": BASE / "mnist_web.pth",
        "Sentiment LSTM": BASE / "sentiment_model.pth",
        "Translator EN→RU": BASE / "translator_bpe_en2ru.pth",
        "Translator RU→EN": BASE / "translator_bpe_ru2en.pth",
        "GAN Generator": BASE / "gan_generator.pth",
        "Price Net":   BASE / "price_model.pth",
        "TimeSeries":  BASE / "timeseries_model.pth",
        "TempNet":     BASE / "temperature_model.pth",
        "SpamLSTM":    BASE / "spam_model.pth",
        "Clustering":  BASE / "clustering_model.pth",
        "AnomalyAE":   BASE / "anomaly_model.pth",
        "DefectNet":   BASE / "defect_model.pth",
        "NERModel":    BASE / "ner_model.pth",
    }
    for name, path in model_files.items():
        status = "найдена" if path.exists() else "НЕ НАЙДЕНА"
        log(f"Модель {name}: {status}", "INFO" if path.exists() else "WARNING")
    t = threading.Thread(target=_lm_monitor, daemon=True)
    t.start()
    app.run(debug=False, host="127.0.0.1", port=port, threaded=True)


if __name__ == "__main__":
    run()
