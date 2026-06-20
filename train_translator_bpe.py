"""
Переводчик EN↔RU с BPE токенизацией
BPE убирает <UNK> — любое слово разбивается на известные части
"""
import sys, io, math, json
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

BASE   = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {DEVICE}")

VOCAB_SIZE = 8000
MAX_LEN    = 40
PAD, SOS, EOS = 0, 1, 2

# ──────────────────────────────────────────────────────
# 1. ЗАГРУЗКА ДАННЫХ
# ──────────────────────────────────────────────────────
print("Загружаю данные...")
en_texts, ru_texts = [], []
with open(BASE / "translator_data.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            en_texts.append(parts[0])
            ru_texts.append(parts[1])

print(f"Пар: {len(en_texts)}")

# ──────────────────────────────────────────────────────
# 2. ОБУЧЕНИЕ BPE ТОКЕНИЗАТОРОВ
# ──────────────────────────────────────────────────────
def train_bpe(texts, save_path, lang):
    path = BASE / save_path
    if path.exists():
        print(f"  Загружаю готовый BPE {lang}...")
        return Tokenizer.from_file(str(path))
    print(f"  Обучаю BPE {lang} (vocab={VOCAB_SIZE})...")
    tok = Tokenizer(BPE(unk_token="<UNK>"))
    tok.pre_tokenizer = ByteLevel()
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"],
        min_frequency=2,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(path))
    print(f"  Сохранён: {path}")
    return tok

print("BPE токенизаторы:")
en_tok = train_bpe(en_texts, "bpe_en.json", "EN")
ru_tok = train_bpe(ru_texts, "bpe_ru.json", "RU")

print(f"EN vocab: {en_tok.get_vocab_size()}, RU vocab: {ru_tok.get_vocab_size()}")

# ──────────────────────────────────────────────────────
# 3. КОДИРОВАНИЕ ДАННЫХ
# ──────────────────────────────────────────────────────
def encode(text, tok, max_len=MAX_LEN):
    ids = [SOS] + tok.encode(text).ids[:max_len-2] + [EOS]
    ids += [PAD] * (max_len - len(ids))
    return ids[:max_len]

print("Кодирую данные...")
EN = np.array([encode(t, en_tok) for t in en_texts], dtype=np.int32)
RU = np.array([encode(t, ru_tok) for t in ru_texts], dtype=np.int32)

split = int(0.92 * len(EN))
EN_tr = torch.tensor(EN[:split], dtype=torch.long)
RU_tr = torch.tensor(RU[:split], dtype=torch.long)
EN_val = torch.tensor(EN[split:], dtype=torch.long)
RU_val = torch.tensor(RU[split:], dtype=torch.long)
print(f"Обучение: {split}, Валидация: {len(EN)-split}")

# ──────────────────────────────────────────────────────
# 4. МОДЕЛЬ
# ──────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TranslatorBPE(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=8, num_enc=4, num_dec=4):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        self.pos_enc   = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_enc, num_decoder_layers=num_dec,
            dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.fc = nn.Linear(d_model, tgt_vocab)
        self.d  = d_model

    def forward(self, src, tgt, src_pad_mask=None, tgt_mask=None):
        se = self.pos_enc(self.src_embed(src) * math.sqrt(self.d))
        te = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d))
        out = self.transformer(se, te, src_key_padding_mask=src_pad_mask, tgt_mask=tgt_mask)
        return self.fc(out)

# ──────────────────────────────────────────────────────
# 5. ОБУЧЕНИЕ EN→RU
# ──────────────────────────────────────────────────────
def train_model(src_data, tgt_data, src_vocab, tgt_vocab, save_name, epochs=150, batch=256):
    model = TranslatorBPE(src_vocab, tgt_vocab).to(DEVICE)
    print(f"\nМодель: {sum(p.numel() for p in model.parameters()):,} параметров")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss = float("inf")
    out_path  = BASE / save_name

    for epoch in range(1, epochs+1):
        model.train()
        idx = torch.randperm(len(src_data))
        total, n = 0.0, 0
        for i in range(0, len(src_data), batch):
            bi  = idx[i:i+batch]
            src = src_data[bi].to(DEVICE)
            tgt = tgt_data[bi].to(DEVICE)
            ti, to_ = tgt[:, :-1], tgt[:, 1:]
            sz = ti.size(1)
            tm = torch.triu(torch.ones(sz, sz, device=DEVICE), diagonal=1).bool()
            sp = (src == PAD)
            out = model(src, ti, src_pad_mask=sp, tgt_mask=tm)
            loss = criterion(out.reshape(-1, tgt_vocab), to_.reshape(-1))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item(); n += 1
        scheduler.step()
        avg = total / n
        if avg < best_loss:
            best_loss = avg
            torch.save({"model": model.state_dict()}, out_path)
        if epoch % 10 == 0:
            print(f"  Эпоха {epoch:3d}/{epochs} | Loss: {avg:.4f} | Best: {best_loss:.4f}")

    print(f"  Сохранено: {out_path} (loss={best_loss:.4f})")
    return model

print("\n=== Обучение EN→RU ===")
train_model(EN_tr, RU_tr, en_tok.get_vocab_size(), ru_tok.get_vocab_size(),
            "translator_bpe_en2ru.pth", epochs=150)

print("\n=== Обучение RU→EN ===")
train_model(RU_tr, EN_tr, ru_tok.get_vocab_size(), en_tok.get_vocab_size(),
            "translator_bpe_ru2en.pth", epochs=150)

print("\nГотово! Оба переводчика обучены с BPE.")
