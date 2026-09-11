import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
from collections import Counter

# ─────────────────────────────────────────
# 1. ДАННЫЕ
# ─────────────────────────────────────────
with open('books_cache.txt', encoding='utf-8') as f:
    raw_text = f.read()

print(f"Текст: {len(raw_text):,} символов")

# Токенизация на слова (не символы как раньше!)
# "Россия была основана" → ["Россия", "была", "основана"]
def tokenize(text):
    # оставляем слова, числа, знаки препинания
    return re.findall(r"[А-Яа-яёЁA-Za-z0-9]+|[.,!?;:\-—«»\(\)\n]", text)

tokens = tokenize(raw_text)
print(f"Токенов всего: {len(tokens):,}")

# Словарь: берём только 8000 самых частых слов
# Редкие слова → <UNK>
MAX_VOCAB = 20000
counts = Counter(tokens)
vocab = ['<UNK>', '<PAD>'] + [w for w, _ in counts.most_common(MAX_VOCAB - 2)]
vocab_size = len(vocab)

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
UNK_IDX = 0

print(f"Словарь: {vocab_size} слов")
print(f"Покрытие: {sum(counts[w] for w in vocab[2:]) / len(tokens) * 100:.1f}% токенов\n")

# Переводим текст в числа
data = torch.tensor(
    [word2idx.get(t, UNK_IDX) for t in tokens],
    dtype=torch.long
)

# Сохраняем словарь
torch.save({'word2idx': word2idx, 'idx2word': idx2word}, 'gpt_vocab.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}")

# ─────────────────────────────────────────
# 2. АРХИТЕКТУРА — Mini GPT
# ─────────────────────────────────────────
#
# Главное отличие от LSTM:
#
# LSTM читает слева направо, слово за словом — как человек читает текст.
# Трансформер видит ВЕСЬ контекст сразу через механизм ATTENTION.
#
# Attention ("внимание") — сеть сама решает, на какие слова обратить
# внимание при генерации каждого следующего слова.
#
# Пример: "Наполеон родился на Корсике. Он стал..."
# При генерации слова после "Он" — attention смотрит на "Наполеон"
# через весь текст, а не только на последние слова.
#
# Архитектура одного блока трансформера:
#
#   Вход
#    │
#    ├─→ [LayerNorm] → [Multi-Head Attention] → (+) ← residual
#    │                                           │
#    └─→─────────────────────────────────────────┘
#    │
#    ├─→ [LayerNorm] → [Feed Forward] → (+) ← residual
#    │                                   │
#    └─→─────────────────────────────────┘
#    │
#   Выход

class SelfAttention(nn.Module):
    """
    Multi-Head Self-Attention.
    Каждый токен "смотрит" на все предыдущие токены и решает,
    что важно для предсказания следующего.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        # Q, K, V — три проекции входа (Query, Key, Value)
        # Аналогия: Q = "что я ищу", K = "что я предлагаю", V = "что я отдаю"
        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence_len, embed_dim

        # Считаем Q, K, V для всех голов сразу
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # каждый: [B, heads, T, head_dim]

        # Attention scores: насколько каждый токен "смотрит" на каждый другой
        scale  = self.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale  # [B, heads, T, T]

        # Causal mask: запрещаем смотреть в будущее (GPT генерирует слева направо)
        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = self.drop(torch.softmax(scores, dim=-1))

        # Взвешенная сумма значений
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = SelfAttention(embed_dim, num_heads)
        self.ff   = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),          # GELU — стандартная функция активации в GPT
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(0.1),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # residual + attention
        x = x + self.ff(self.norm2(x))     # residual + feed-forward
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8,
                 num_layers=6, context_len=128, ff_dim=1024):
        super().__init__()
        self.context_len = context_len

        # Два вида эмбеддингов: для слов и для позиций
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb   = nn.Embedding(context_len, embed_dim)  # позиция в тексте

        self.drop   = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: эмбеддинг и выходной слой делят веса (стандарт в GPT)
        self.head.weight = self.token_emb.weight

        # Инициализация весов как в GPT-2
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)

        # Складываем эмбеддинг слова + эмбеддинг позиции
        x = self.drop(self.token_emb(x) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)  # [B, T, vocab_size]


SEQ_LEN = 128

model = MiniGPT(vocab_size, context_len=SEQ_LEN).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Параметров: {params:,}\n")


# ─────────────────────────────────────────
# 3. ОБУЧЕНИЕ
# ─────────────────────────────────────────
def get_batch(batch_size=64):
    starts = torch.randint(0, len(data) - SEQ_LEN - 1, (batch_size,))
    x = torch.stack([data[s : s + SEQ_LEN]     for s in starts])
    y = torch.stack([data[s + 1 : s + SEQ_LEN + 1] for s in starts])
    return x.to(device), y.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
epochs = 3000
print_every = 100
best_loss = float('inf')

for epoch in range(1, epochs + 1):
    model.train()
    x, y = get_batch(64)
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % print_every == 0:
        print(f"Эпоха {epoch:4d}/{epochs} | Loss: {loss.item():.4f}")
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'gpt_model.pth')


# ─────────────────────────────────────────
# 4. ГЕНЕРАЦИЯ
# ─────────────────────────────────────────
def generate(prompt, max_new_tokens=150, temperature=0.8, top_k=40):
    """
    top_k: на каждом шаге выбираем только из k самых вероятных слов
    (убирает совсем нелепые варианты)
    """
    model.eval()
    tokens_in = [word2idx.get(t, UNK_IDX) for t in tokenize(prompt)]
    ids = torch.tensor([tokens_in], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # берём последние context_len токенов
            ids_crop = ids[:, -SEQ_LEN:]
            logits   = model(ids_crop)[0, -1]  # предсказание для последнего токена

            # top-k фильтрация
            if top_k:
                top_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_vals[-1]] = float('-inf')

            probs   = torch.softmax(logits / temperature, dim=0)
            next_id = torch.multinomial(probs, 1).item()
            ids     = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    result_tokens = [idx2word.get(i, '<UNK>') for i in ids[0].tolist()]
    # Склеиваем слова обратно в текст
    text = ''
    for t in result_tokens:
        if t in '.,!?;:' or text == '':
            text += t
        else:
            text += ' ' + t
    return text


print("\n" + "="*60)
print("ГЕНЕРАЦИЯ (Mini GPT):")
print("="*60)

with open('output_gpt.txt', 'w', encoding='utf-8') as f:
    for prompt in ["Россия является", "История науки", "В начале XX века", "Исследования показали"]:
        result = generate(prompt, max_new_tokens=120, temperature=0.8)
        print(f"\n--- {prompt} ---")
        print(result)
        f.write(f"=== {prompt} ===\n{result}\n\n")

print("\nСохранено в output_gpt.txt")
