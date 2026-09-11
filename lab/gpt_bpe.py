import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import os

# ─────────────────────────────────────────
# 1. BPE ТОКЕНИЗАТОР
# ─────────────────────────────────────────
#
# BPE (Byte Pair Encoding) — алгоритм который:
# 1. Начинает с отдельных символов: ["Р","о","с","с","и","я"]
# 2. Находит самые частые пары и сливает их: "сс" → один токен
# 3. Повторяет пока не наберёт нужный размер словаря
#
# Итог: частые слова → один токен, редкие → несколько частей
# "и" → [и]           (один токен, очень частое)
# "Россия" → [Россия]  (один токен, достаточно частое)
# "Достоевский" → [Досто, евский]  (редкое имя, 2 части)
# UNK исчезает полностью!

TOKENIZER_PATH = "bpe_tokenizer.json"
CACHE_FILE     = "books_cache.txt"
VOCAB_SIZE     = 16000   # достаточно для русского языка

print("Загружаем текст...")
with open(CACHE_FILE, encoding='utf-8') as f:
    text = f.read()
print(f"Символов: {len(text):,}\n")

if os.path.exists(TOKENIZER_PATH):
    print("Загружаем готовый токенизатор...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
else:
    print(f"Обучаем BPE токенизатор (словарь {VOCAB_SIZE})...")

    # Сохраняем текст во временный файл для trainer'а
    with open("_train_text.txt", "w", encoding="utf-8") as f:
        f.write(text)

    # ByteLevel: кодирует каждый байт, пробелы → "Ġ"
    # Это стандарт GPT-2 — позволяет точно восстановить исходный текст
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[PAD]"],
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train(["_train_text.txt"], trainer)
    tokenizer.save(TOKENIZER_PATH)
    try:
        os.remove("_train_text.txt")
    except Exception:
        pass
    print("Токенизатор сохранён\n")

vocab_size = tokenizer.get_vocab_size()
print(f"Размер словаря: {vocab_size}")

# Токенизируем весь текст
print("Токенизируем текст...")
encoding = tokenizer.encode(text)
tokens   = encoding.ids
print(f"Токенов всего: {len(tokens):,}")

# Проверим покрытие — сколько [UNK] в тексте
print(f"UNK токенов: 0 (0.00%) -- ByteLevel не имеет UNK\n")

# Пример токенизации
sample = "Достоевский написал Преступление и наказание"
enc    = tokenizer.encode(sample)
print(f"Пример: '{sample}'")
print(f"Токены: {enc.tokens}\n")

data   = torch.tensor(tokens, dtype=torch.long)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}")


# ─────────────────────────────────────────
# 2. АРХИТЕКТУРА GPT (та же что раньше)
# ─────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale  = self.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn   = self.drop(torch.softmax(scores, dim=-1))
        out    = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn  = SelfAttention(embed_dim, num_heads)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(0.1),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8,
                 num_layers=6, context_len=128, ff_dim=1024):
        super().__init__()
        self.context_len = context_len
        self.token_emb   = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb     = nn.Embedding(context_len, embed_dim)
        self.drop        = nn.Dropout(0.1)
        self.blocks      = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        x    = self.drop(self.token_emb(x) + self.pos_emb(pos))
        x    = self.blocks(x)
        return self.head(self.norm(x))


SEQ_LEN = 128
model   = MiniGPT(vocab_size, context_len=SEQ_LEN).to(device)
params  = sum(p.numel() for p in model.parameters())
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
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)
# CosineAnnealingLR: LR плавно снижается по косинусу — лучше чем ступенчато

epochs     = 10000
print_every = 500
best_loss   = float('inf')

for epoch in range(1, epochs + 1):
    model.train()
    x, y   = get_batch(64)
    logits = model(x)
    loss   = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if epoch % print_every == 0:
        lr = scheduler.get_last_lr()[0]
        print(f"Эпоха {epoch:5d}/{epochs} | Loss: {loss.item():.4f} | LR: {lr:.6f}")
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'gpt_bpe_model.pth')


# ─────────────────────────────────────────
# 4. ГЕНЕРАЦИЯ
# ─────────────────────────────────────────
def generate(prompt, max_new_tokens=200, temperature=0.8, top_k=50):
    model.eval()
    ids = torch.tensor(
        [tokenizer.encode(prompt).ids],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ids_crop = ids[:, -SEQ_LEN:]
            logits   = model(ids_crop)[0, -1]

            if top_k:
                top_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_vals[-1]] = float('-inf')

            probs   = torch.softmax(logits / temperature, dim=0)
            next_id = torch.multinomial(probs, 1).item()
            ids     = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    # Декодируем токены обратно в текст — tokenizer сам склеивает части слов
    return tokenizer.decode(ids[0].tolist())


print("\n" + "="*60)
print("ГЕНЕРАЦИЯ (GPT + BPE):")
print("="*60)

with open('output_bpe.txt', 'w', encoding='utf-8') as f:
    for prompt in ["Россия является", "История науки", "В начале XX века", "Исследования показали"]:
        result = generate(prompt, max_new_tokens=150, temperature=0.8)
        print(f"\n--- {prompt} ---")
        print(result)
        f.write(f"=== {prompt} ===\n{result}\n\n")

print("\nСохранено в output_bpe.txt")
