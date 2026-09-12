"""
Переводчик EN↔RU: трансформер + BPE, корпус Tatoeba.

Запуск из папки lab:  python train_translator_bpe.py
Результат: translator_bpe_en2ru.pth, translator_bpe_ru2en.pth, bpe_en.json, bpe_ru.json.
Чтобы приложение их использовало, скопируй все четыре файла в ../models.

Прошлая версия (post-LN, lr 1e-3 без разогрева) выродилась: декодер научился
генерировать правдоподобные фразы и перестал смотреть на исходный текст.
Здесь pre-LN, разогрев LR, маски паддинга и ранняя остановка по валидации.
"""
import bz2, math, random, sys, tarfile, time
from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

BASE   = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE, MAX_LEN = 8000, 40
PAD, SOS, EOS = 0, 1, 2
CONFIG = {"d_model": 256, "nhead": 8, "num_enc": 4, "num_dec": 4, "ff": 1024, "norm_first": True, "max_len": MAX_LEN}
SAMPLES = {"en2ru": ["Tom is my friend.", "I am very tired.", "Where is the station?", "The weather is nice today."],
           "ru2en": ["Том мой друг.", "Я очень устал.", "Где вокзал?", "Сегодня хорошая погода."]}


def read_sentences(path):
    sents = {}
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                sents[parts[0]] = parts[2]
    return sents


def read_links(path):
    # links.tar.bz2 с сайта Tatoeba внутри содержит links.csv
    with tarfile.open(path, "r:bz2") as tar:
        member = next(m for m in tar.getmembers() if m.isfile())
        for raw in tar.extractfile(member):
            a, _, b = raw.decode().strip().partition("\t")
            yield a, b


def build_pairs():
    print("Корпус Tatoeba: читаю предложения...")
    eng = read_sentences(BASE / "tatoeba_eng.tsv.bz2")
    rus = read_sentences(BASE / "tatoeba_rus.tsv.bz2")
    pairs = {(eng[a], rus[b]) for a, b in read_links(BASE / "tatoeba_links.tsv.bz2") if a in eng and b in rus}
    pairs = sorted(p for p in pairs if len(p[0]) <= 160 and len(p[1]) <= 160)
    random.Random(42).shuffle(pairs)
    print(f"Уникальных пар EN-RU: {len(pairs)}")
    return pairs


def train_bpe(texts, name):
    tok = Tokenizer(BPE(unk_token="<UNK>"))
    tok.pre_tokenizer = ByteLevel()
    tok.decoder = ByteLevelDecoder()
    tok.train_from_iterator(texts, trainer=BpeTrainer(
        vocab_size=VOCAB_SIZE, special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"], min_frequency=2))
    tok.save(str(BASE / name))
    return tok


def encode(text, tok):
    ids = [SOS] + tok.encode(text).ids[:MAX_LEN - 2] + [EOS]
    return ids + [PAD] * (MAX_LEN - len(ids))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TranslatorBPE(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=8, num_enc=4, num_dec=4, ff=1024, norm_first=True, max_len=MAX_LEN):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        self.pos_enc   = PositionalEncoding(d_model, max_len)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_enc,
                                          num_decoder_layers=num_dec, dim_feedforward=ff, dropout=0.1,
                                          batch_first=True, norm_first=norm_first)
        self.fc = nn.Linear(d_model, tgt_vocab)
        self.d  = d_model

    def forward(self, src, tgt):
        T = tgt.size(1)
        causal = torch.triu(torch.ones(T, T, device=src.device, dtype=torch.bool), diagonal=1)
        out = self.transformer(
            self.pos_enc(self.src_embed(src) * math.sqrt(self.d)),
            self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d)),
            tgt_mask=causal, src_key_padding_mask=src == PAD,
            tgt_key_padding_mask=tgt == PAD, memory_key_padding_mask=src == PAD)
        return self.fc(out)


@torch.no_grad()
def translate(model, text, src_tok, tgt_tok):
    model.eval()
    src = torch.tensor([encode(text, src_tok)], device=DEVICE)
    pad = src == PAD
    mem = model.transformer.encoder(model.pos_enc(model.src_embed(src) * math.sqrt(model.d)), src_key_padding_mask=pad)
    ids = [SOS]
    for _ in range(MAX_LEN - 1):
        tgt = torch.tensor([ids], device=DEVICE)
        T = tgt.size(1)
        causal = torch.triu(torch.ones(T, T, device=DEVICE, dtype=torch.bool), diagonal=1)
        out = model.transformer.decoder(model.pos_enc(model.tgt_embed(tgt) * math.sqrt(model.d)), mem,
                                        tgt_mask=causal, memory_key_padding_mask=pad)
        nid = model.fc(out[:, -1]).argmax(-1).item()
        if nid == EOS:
            break
        ids.append(nid)
    return tgt_tok.decode(ids[1:]).strip()


def run_epoch(model, src, tgt, batch, optimizer=None, scheduler=None, criterion=None):
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(src)) if training else torch.arange(len(src))
    total, n = 0.0, 0
    for i in range(0, len(src), batch):
        bi = order[i:i + batch]
        s, t = src[bi].to(DEVICE), tgt[bi].to(DEVICE)
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"), torch.set_grad_enabled(training):
            logits = model(s, t[:, :-1])
            loss = criterion(logits.float().reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1))
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        total += loss.item(); n += 1
    return total / n


def train_direction(name, src_tr, tgt_tr, src_val, tgt_val, src_tok, tgt_tok, epochs=40, batch=256, patience=4):
    model = TranslatorBPE(src_tok.get_vocab_size(), tgt_tok.get_vocab_size(), **CONFIG).to(DEVICE)
    print(f"\n=== {name}: {sum(p.numel() for p in model.parameters()):,} параметров ===")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98), weight_decay=1e-4)
    steps, warmup = epochs * math.ceil(len(src_tr) / batch), 4000
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))
    best, bad = float("inf"), 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, src_tr, tgt_tr, batch, optimizer, scheduler, criterion)
        val = run_epoch(model, src_val, tgt_val, 512, criterion=criterion)
        mark = ""
        if val < best:
            best, bad, mark = val, 0, " *"
            torch.save({"model": model.state_dict(), "config": CONFIG}, BASE / f"translator_bpe_{name}.pth")
        else:
            bad += 1
        print(f"  эпоха {epoch:2d} | train {tr:.3f} | val {val:.3f}{mark} | {time.time() - t0:.0f} с")
        print("    " + " | ".join(f"{s} → {translate(model, s, src_tok, tgt_tok)}" for s in SAMPLES[name]))
        if bad >= patience:
            print("  Валидация перестала улучшаться, останавливаюсь")
            break
    print(f"  Лучший val loss: {best:.3f}")


def main():
    pairs = build_pairs()
    en_texts, ru_texts = [p[0] for p in pairs], [p[1] for p in pairs]
    print("Обучаю BPE...")
    en_tok, ru_tok = train_bpe(en_texts, "bpe_en.json"), train_bpe(ru_texts, "bpe_ru.json")
    EN = torch.tensor([encode(t, en_tok) for t in en_texts])
    RU = torch.tensor([encode(t, ru_tok) for t in ru_texts])
    n_val = 3000
    train_direction("en2ru", EN[n_val:], RU[n_val:], EN[:n_val], RU[:n_val], en_tok, ru_tok)
    train_direction("ru2en", RU[n_val:], EN[n_val:], RU[:n_val], EN[:n_val], ru_tok, en_tok)
    print("\nГотово. Скопируй translator_bpe_*.pth и bpe_en.json, bpe_ru.json в ../models")


if __name__ == "__main__":
    main()
