"""
Neural Network Lab — Web Interface
Запуск: python web_app.py
Открыть: http://localhost:5000
"""
import sys, io, os, json, base64, threading, re, math, warnings, logging
from collections import deque
from datetime import datetime
warnings.filterwarnings('ignore', message='.*triton.*')
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

BASE   = Path(__file__).parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    from flask import Flask, render_template, request, jsonify, Response, stream_with_context, render_template_string
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask', '-q'])
    from flask import Flask, render_template, request, jsonify, Response, stream_with_context, render_template_string

app = Flask(__name__)

# ═══════════════════════════════════════════════════
# ЛОГИРОВАНИЕ
# ═══════════════════════════════════════════════════
_log_buffer = deque(maxlen=500)

class _MemHandler(logging.Handler):
    LEVEL_COLOR = {'DEBUG':'#888','INFO':'#58a6ff','WARNING':'#e3b341','ERROR':'#f85149','CRITICAL':'#ff7b72'}
    def emit(self, record):
        msg = self.format(record)
        _log_buffer.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'level': record.levelname,
            'color': self.LEVEL_COLOR.get(record.levelname, '#ccc'),
            'msg': msg,
        })

_mem_handler = _MemHandler()
_mem_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger('werkzeug').addHandler(_mem_handler)
logging.getLogger('werkzeug').setLevel(logging.INFO)

def log(msg, level='INFO'):
    """Добавить запись в лог вручную"""
    color = _MemHandler.LEVEL_COLOR.get(level, '#ccc')
    _log_buffer.append({'time': datetime.now().strftime('%H:%M:%S'), 'level': level, 'color': color, 'msg': msg})
    print(msg)

_LOGS_HTML = '''<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Logs — Neural Network Lab :5000</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:monospace;font-size:13px}
header{background:#161b22;border-bottom:1px solid #30363d;padding:12px 20px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:10}
h1{font-size:15px;color:#58a6ff}
.badge{background:#21262d;border:1px solid #30363d;border-radius:12px;padding:3px 10px;font-size:11px;color:#8b949e}
button{background:#21262d;border:1px solid #30363d;color:#c9d1d9;border-radius:6px;padding:5px 12px;font-size:12px;cursor:pointer}
button:hover{border-color:#58a6ff;color:#58a6ff}
#log{padding:12px 20px}
.row{display:flex;gap:10px;padding:3px 0;border-bottom:1px solid #161b22;line-height:1.5}
.row:hover{background:#161b22}
.t{color:#484f58;min-width:60px}
.lv{min-width:52px;font-weight:700}
.msg{white-space:pre-wrap;word-break:break-all;flex:1}
#status{font-size:11px;color:#484f58}
</style></head><body>
<header>
  <h1>📋 Логи — Neural Network Lab</h1>
  <span class="badge">:5000</span>
  <span id="status">авто-обновление вкл</span>
  <button onclick="toggleAuto()">⏸ Пауза</button>
  <button onclick="document.getElementById(\'log\').innerHTML=\'\'">🗑 Очистить</button>
  <a href="http://localhost:5001/logs" style="margin-left:auto;color:#58a6ff;font-size:12px;text-decoration:none">→ Smart Assistant логи :5001</a>
</header>
<div id="log"></div>
<script>
let auto=true, lastCount=0;
function toggleAuto(){auto=!auto;document.getElementById('status').textContent=auto?'авто-обновление вкл':'пауза';}
async function poll(){
  if(!auto)return;
  try{
    const r=await fetch('/api/logs').then(r=>r.json());
    if(r.length!==lastCount){
      lastCount=r.length;
      const el=document.getElementById('log');
      el.innerHTML=r.map(e=>`<div class="row"><span class="t">${e.time}</span><span class="lv" style="color:${e.color}">${e.level}</span><span class="msg">${e.msg.replace(/</g,'&lt;')}</span></div>`).join('');
      el.scrollTop=el.scrollHeight;
    }
  }catch(e){}
}
setInterval(poll,1000);poll();
</script></body></html>'''

# ═══════════════════════════════════════════════════
# ARCHITECTURES
# ═══════════════════════════════════════════════════

# ─── MiniGPT ───
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
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / self.head_dim**0.5
        if mask is not None: att = att.masked_fill(mask==0, -1e9)
        att = torch.softmax(att, -1)
        return self.proj((att @ v).transpose(1,2).contiguous().view(B, T, C))

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
    def __init__(self, vocab_size, embed_dim=256, num_heads=8,
                 num_layers=6, ff_dim=1024, context_len=128):
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

# ─── MNIST ResNet ───
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.block(x))

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

# ─── Sentiment LSTM ───
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, n_classes=3):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm    = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                               bidirectional=True, num_layers=2, dropout=0.3)
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(hidden_dim * 2, n_classes)
    def forward(self, x):
        emb = self.dropout(self.embed(x))
        out, (h, _) = self.lstm(emb)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))

# ─── Translator ───
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TranslatorTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=8,
                 num_enc=4, num_dec=4, max_len=30, PAD=0):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        self.pos_enc   = PositionalEncoding(d_model, max_len)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_enc, num_decoder_layers=num_dec,
            dim_feedforward=512, dropout=0.1, batch_first=True)
        self.fc      = nn.Linear(d_model, tgt_vocab)
        self.d_model = d_model
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None):
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.transformer(src_emb, tgt_emb,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_mask=tgt_mask)
        return self.fc(out)
    def translate(self, src_ids, tgt_w2i, tgt_i2w, max_len=30):
        self.eval()
        src = torch.tensor([src_ids], dtype=torch.long).to(DEVICE)
        BOS = tgt_w2i.get('<BOS>', 1)
        EOS = tgt_w2i.get('<EOS>', 2)
        tgt_ids = [BOS]
        with torch.no_grad():
            src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
            mem = self.transformer.encoder(src_emb)
            for _ in range(max_len):
                tgt = torch.tensor([tgt_ids], dtype=torch.long).to(DEVICE)
                tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
                T = tgt.size(1)
                tgt_mask = torch.triu(torch.ones(T, T, device=DEVICE), diagonal=1).bool()
                out = self.transformer.decoder(tgt_emb, mem, tgt_mask=tgt_mask)
                next_id = self.fc(out[:, -1]).argmax(-1).item()
                tgt_ids.append(next_id)
                if next_id == EOS:
                    break
        tokens = [tgt_i2w.get(str(i), tgt_i2w.get(i, '')) for i in tgt_ids[1:] if i != EOS]
        return ' '.join(t for t in tokens if t not in ('<PAD>', '<BOS>', '<EOS>', ''))

# ─── BPE Translator ───
class TranslatorBPE(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=8, num_enc=4, num_dec=4):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len=40)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_enc, num_decoder_layers=num_dec,
            dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(d_model, tgt_vocab)
        self.d  = d_model
    def forward(self, src, tgt, src_pad_mask=None, tgt_mask=None):
        se = self.pos_enc(self.src_embed(src) * math.sqrt(self.d))
        te = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d))
        return self.fc(self.transformer(se, te, src_key_padding_mask=src_pad_mask, tgt_mask=tgt_mask))

# ─── Conditional DCGAN Generator ───
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.net = nn.Sequential(
            nn.Linear(100 + 10, 7 * 7 * 256),
            nn.Unflatten(1, (256, 7, 7)),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, z, labels=None):
        if labels is None:
            labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        emb = self.label_emb(labels)
        return self.net(torch.cat([z, emb], dim=1))

# ═══════════════════════════════════════════════════
# LAZY MODEL CACHE
# ═══════════════════════════════════════════════════
_cache = {}
_lock  = threading.Lock()

def load_gpt():
    with _lock:
        if 'gpt' not in _cache:
            from tokenizers import Tokenizer
            tok  = Tokenizer.from_file(str(BASE / 'bpe_tokenizer.json'))
            VOCAB = tok.get_vocab_size()
            model = MiniGPT(VOCAB).to(DEVICE)
            path  = BASE / 'gpt_multilingual.pth'
            if not path.exists():
                path = BASE / 'gpt_bpe_model.pth'
            model.load_state_dict(torch.load(path, weights_only=True, map_location=DEVICE))
            model.eval()
            _cache['gpt'] = (model, tok)
    return _cache['gpt']

def load_mnist():
    with _lock:
        if 'mnist' not in _cache:
            model = ResNet().to(DEVICE)
            # mnist_web.pth — обучена с аугментацией толстых штрихов (для web canvas)
            path = BASE / 'mnist_web.pth'
            if not path.exists():
                path = BASE / 'best_model_v3.pth'
            model.load_state_dict(torch.load(path, weights_only=True, map_location=DEVICE))
            model.eval()
            _cache['mnist'] = model
    return _cache['mnist']

def load_sentiment():
    with _lock:
        if 'sentiment' not in _cache:
            # Rebuild vocab (same data as sentiment.py)
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
            counter = Counter()
            for t in RAW:
                counter.update(t.lower().split())
            vocab = {'<PAD>': 0, '<UNK>': 1}
            for w, _ in counter.most_common(500):
                vocab[w] = len(vocab)
            model = SentimentLSTM(len(vocab)).to(DEVICE)
            model.load_state_dict(torch.load(BASE / 'sentiment_model.pth', weights_only=True, map_location=DEVICE))
            model.eval()
            _cache['sentiment'] = (model, vocab)
    return _cache['sentiment']

def load_translator(direction='en2ru'):
    key = f'translator_{direction}'
    with _lock:
        if key not in _cache:
            path = BASE / ('translator_model.pth' if direction == 'en2ru' else 'translator_ru2en_model.pth')
            if not path.exists():
                path = BASE / 'translator_model.pth'
            ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
            en_w2i = ckpt['en_w2i']
            ru_w2i = ckpt['ru_w2i']
            en_i2w = ckpt.get('en_i2w', {})
            ru_i2w = ckpt.get('ru_i2w', {})
            if direction == 'en2ru':
                model = TranslatorTransformer(len(en_w2i), len(ru_w2i)).to(DEVICE)
            else:
                model = TranslatorTransformer(len(ru_w2i), len(en_w2i)).to(DEVICE)
            model.load_state_dict(ckpt['model'])
            model.eval()
            _cache[key] = (model, en_w2i, ru_w2i, en_i2w, ru_i2w)
    return _cache[key]

def load_translator_bpe(direction='en2ru'):
    key = f'translator_bpe_{direction}'
    with _lock:
        if key not in _cache:
            from tokenizers import Tokenizer
            en_tok = Tokenizer.from_file(str(BASE / 'bpe_en.json'))
            ru_tok = Tokenizer.from_file(str(BASE / 'bpe_ru.json'))
            if direction == 'en2ru':
                model_path = BASE / 'translator_bpe_en2ru.pth'
                src_tok, tgt_tok = en_tok, ru_tok
            else:
                model_path = BASE / 'translator_bpe_ru2en.pth'
                src_tok, tgt_tok = ru_tok, en_tok
            ckpt = torch.load(model_path, weights_only=False, map_location=DEVICE)
            model = TranslatorBPE(src_tok.get_vocab_size(), tgt_tok.get_vocab_size()).to(DEVICE)
            model.load_state_dict(ckpt['model'])
            model.eval()
            _cache[key] = (model, src_tok, tgt_tok)
    return _cache[key]

def load_gan():
    with _lock:
        if 'gan' not in _cache:
            model = Generator().to(DEVICE)
            model.load_state_dict(torch.load(BASE / 'gan_generator.pth', weights_only=False, map_location=DEVICE))
            model.eval()
            _cache['gan'] = model
    return _cache['gan']

# ═══════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════

@app.before_request
def _log_request():
    log(f'{request.method} {request.path}', 'INFO')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logs')
def logs_page():
    return render_template_string(_LOGS_HTML)

@app.route('/api/logs')
def api_logs():
    return jsonify(list(_log_buffer))

@app.route('/api/status')
def status():
    return jsonify({
        'device': str(DEVICE),
        'version': '2.0',
        'models': {
            'gpt':         (BASE / 'gpt_multilingual.pth').exists() or (BASE / 'gpt_bpe_model.pth').exists(),
            'mnist':       (BASE / 'mnist_web.pth').exists() or (BASE / 'best_model_v3.pth').exists(),
            'sentiment':   (BASE / 'sentiment_model.pth').exists(),
            'translator':  (BASE / 'translator_bpe_en2ru.pth').exists() or (BASE / 'translator_model.pth').exists(),
            'gan':         (BASE / 'gan_generator.pth').exists(),
            'price':       (BASE / 'price_model.pth').exists(),
            'timeseries':  (BASE / 'timeseries_model.pth').exists(),
            'temperature': (BASE / 'temperature_model.pth').exists(),
            'spam':        (BASE / 'spam_model.pth').exists(),
            'ner':         (BASE / 'ner_model.pth').exists(),
            'anomaly':     (BASE / 'anomaly_model.pth').exists(),
            'defect':      (BASE / 'defect_model.pth').exists(),
            'clustering':  (BASE / 'clustering_model.pth').exists(),
            'recommender': (BASE / 'recommender_model.pth').exists(),
        }
    })

# ─── GPT ───
@app.route('/api/gpt/stream', methods=['POST'])
def gpt_stream():
    data        = request.json
    seed        = data.get('seed', '[EN] The')
    length      = min(int(data.get('length', 250)), 500)
    temperature = float(data.get('temperature', 0.8))

    model, tokenizer = load_gpt()
    CTX  = 128
    ids  = list(tokenizer.encode(seed).ids[-CTX:])

    def generate():
        yield f"data: {json.dumps({'text': seed})}\n\n"
        with torch.no_grad():
            for _ in range(length):
                x      = torch.tensor([ids[-CTX:]], dtype=torch.long).to(DEVICE)
                logits = model(x)[0, -1] / temperature
                probs  = torch.softmax(logits, dim=0)
                nid    = torch.multinomial(probs, 1).item()
                ids.append(nid)
                tok = tokenizer.decode([nid])
                yield f"data: {json.dumps({'text': tok})}\n\n"
        yield "data: {\"done\":true}\n\n"

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# ─── MNIST ───
@app.route('/api/mnist/predict', methods=['POST'])
def mnist_predict():
    try:
        import PIL.Image, PIL.ImageOps
        data      = request.json
        img_b64   = data.get('image', '')
        img_bytes = base64.b64decode(img_b64.split(',')[-1])
        img = PIL.Image.open(io.BytesIO(img_bytes)).convert('L')
        arr = np.array(img, dtype=np.uint8)

        # Находим bounding box нарисованного содержимого
        rows = np.any(arr > 20, axis=1)
        cols = np.any(arr > 20, axis=0)
        if rows.any() and cols.any():
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            # Делаем квадратный кроп с отступом
            h, w = r1 - r0, c1 - c0
            side  = max(h, w)
            pad   = max(int(side * 0.3), 10)
            cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
            half  = side // 2 + pad
            r0 = max(0, cr - half); r1 = min(arr.shape[0], cr + half)
            c0 = max(0, cc - half); c1 = min(arr.shape[1], cc + half)
            arr = arr[r0:r1, c0:c1]

        # Обрезаем до содержимого (bounding box)
        if rows.any() and cols.any():
            h = arr.shape[0]
            w = arr.shape[1]
            # arr уже обрезан выше, просто работаем с ним дальше
            pass

        # Вписываем цифру в 20x20, центрируем в 28x28 — точно как MNIST
        h, w = arr.shape
        scale = 20.0 / max(h, w)
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        img = PIL.Image.fromarray(arr).resize((new_w, new_h), PIL.Image.LANCZOS)

        canvas = np.zeros((28, 28), dtype=np.float32)
        y0 = (28 - new_h) // 2
        x0 = (28 - new_w) // 2
        canvas[y0:y0+new_h, x0:x0+new_w] = np.array(img, dtype=np.float32)

        # Сохраняем debug-изображение до нормализации
        debug_arr = (canvas).astype(np.uint8)
        debug_img = PIL.Image.fromarray(debug_arr).resize((112, 112), PIL.Image.NEAREST)
        dbuf = io.BytesIO()
        debug_img.save(dbuf, format='PNG')
        debug_b64 = 'data:image/png;base64,' + base64.b64encode(dbuf.getvalue()).decode()

        arr = canvas / 255.0
        arr = (arr - 0.1307) / 0.3081
        x   = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)

        model = load_mnist()
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].tolist()
        pred = int(np.argmax(probs))
        return jsonify({'digit': pred, 'confidence': round(max(probs)*100, 1),
                        'probs': [round(p*100, 1) for p in probs],
                        'debug': debug_b64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─── Sentiment ───
@app.route('/api/sentiment/analyze', methods=['POST'])
def sentiment_analyze():
    text  = request.json.get('text', '').lower()
    model, vocab = load_sentiment()
    MAX_LEN = 20
    ids = [vocab.get(w, 1) for w in text.split()][:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    x = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].tolist()
    labels = ['негативный', 'нейтральный', 'позитивный']
    emojis = ['😠', '😐', '😊']
    pred   = int(np.argmax(probs))
    return jsonify({
        'label': labels[pred],
        'emoji': emojis[pred],
        'scores': {l: round(p*100, 1) for l, p in zip(labels, probs)}
    })

# ─── Math ───
@app.route('/api/math/solve', methods=['POST'])
def math_solve():
    expr = request.json.get('expression', '')
    try:
        clean = re.sub(r'[^0-9+\-*/().\s]', '', expr)
        if not clean.strip():
            raise ValueError("Пустое выражение")
        result = eval(clean, {"__builtins__": {}})
        return jsonify({'expression': expr, 'result': round(float(result), 6)})
    except ZeroDivisionError:
        return jsonify({'error': 'Деление на ноль'})
    except Exception as e:
        return jsonify({'error': f'Ошибка: {e}'})

# ─── Translator (BPE) ───
@app.route('/api/translate', methods=['POST'])
def translate():
    data      = request.json
    text      = data.get('text', '').strip()
    direction = data.get('direction', 'en2ru')
    try:
        model, src_tok, tgt_tok = load_translator_bpe(direction)
        MAX_LEN = 40
        PAD, SOS, EOS = 0, 1, 2

        # Кодируем входной текст через BPE
        ids = [SOS] + src_tok.encode(text).ids[:MAX_LEN-2] + [EOS]
        ids += [PAD] * (MAX_LEN - len(ids))
        src = torch.tensor([ids[:MAX_LEN]], dtype=torch.long).to(DEVICE)

        # Greedy decode
        model.eval()
        with torch.no_grad():
            se = model.pos_enc(model.src_embed(src) * math.sqrt(model.d))
            mem = model.transformer.encoder(se)
            tgt_ids = [SOS]
            for _ in range(MAX_LEN):
                tgt = torch.tensor([tgt_ids], dtype=torch.long).to(DEVICE)
                te = model.pos_enc(model.tgt_embed(tgt) * math.sqrt(model.d))
                T = tgt.size(1)
                tm = torch.triu(torch.ones(T, T, device=DEVICE), diagonal=1).bool()
                out = model.transformer.decoder(te, mem, tgt_mask=tm)
                nid = model.fc(out[:, -1]).argmax(-1).item()
                if nid == EOS: break
                tgt_ids.append(nid)

        # Декодируем через BPE токенизатор (убирает subword артефакты)
        result = tgt_tok.decode(tgt_ids[1:])
        return jsonify({'translation': result or '(пустой результат)'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─── GAN ───
@app.route('/api/gan/generate', methods=['POST'])
def gan_generate():
    try:
        import PIL.Image
        data  = request.json or {}
        digit = data.get('digit', -1)   # -1 = случайная, 0-9 = конкретная цифра
        count = min(int(data.get('count', 1)), 16)

        model  = load_gan()
        z      = torch.randn(count, 100).to(DEVICE)
        labels = (torch.randint(0, 10, (count,)) if digit == -1
                  else torch.full((count,), int(digit))).to(DEVICE)

        with torch.no_grad():
            imgs = model(z, labels).cpu()

        results = []
        for i in range(count):
            img = ((imgs[i, 0].numpy() + 1) / 2 * 255).astype(np.uint8)
            resample = getattr(PIL.Image, 'Resampling', PIL.Image).BICUBIC
            pil = PIL.Image.fromarray(img).resize((140, 140), resample)
            buf = io.BytesIO()
            pil.save(buf, format='PNG')
            results.append('data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode())

        return jsonify({'images': results, 'digit': digit})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─── Coder Team ───
@app.route('/api/coder_team/run', methods=['POST'])
def coder_team_run():
    import subprocess as sp
    task = request.json.get('task', '').strip()
    if not task:
        return jsonify({'error': 'Задача не указана'}), 400

    script = str(BASE / 'coder_team.py')

    def generate():
        proc = sp.Popen(
            [sys.executable, '-u', script, task],
            stdout=sp.PIPE, stderr=sp.STDOUT,
            text=True, encoding='utf-8', errors='replace',
            cwd=str(BASE)
        )
        for line in proc.stdout:
            yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
        proc.wait()
        # Читаем итоговый файл если есть
        out_file = BASE / 'coder_output.py'
        if out_file.exists():
            code = out_file.read_text(encoding='utf-8')
            yield f"data: {json.dumps({'done': True, 'code': code})}\n\n"
        else:
            yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

if __name__ == '__main__':
    import io as _io
    print("=" * 50)
    print("  Neural Network Lab — Web Interface")
    print(f"  Устройство: {DEVICE}")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
