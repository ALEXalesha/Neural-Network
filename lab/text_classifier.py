"""Классификатор текста (тональность, спам): BiLSTM + max-pooling по словам без паддинга.

Обучение, валидация с ранней остановкой и сохранение в формате, который читает app_server.py.
"""
import time

import torch
import torch.nn as nn

from hf_data import build_vocab, tokenize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def encode(texts, vocab, max_len):
    ids = [[vocab.get(t, 1) for t in tokenize(s.lower())][:max_len] or [1] for s in texts]
    return torch.tensor([x + [0] * (max_len - len(x)) for x in ids])


def accuracy(model, X, Y):
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(X[i:i + 512].to(DEVICE)).argmax(-1).cpu() for i in range(0, len(X), 512)])
    return (pred == Y).float().mean().item(), pred


def train(train_texts, train_y, val_texts, val_y, labels, out_path, max_len=64, vocab_size=30000,
          epochs=15, patience=3, class_weight=None):
    vocab = build_vocab([tokenize(t.lower()) for t in train_texts], vocab_size)
    Xtr, Xval = encode(train_texts, vocab, max_len), encode(val_texts, vocab, max_len)
    Ytr, Yval = torch.tensor(train_y), torch.tensor(val_y)
    cfg = {"embed_dim": 128, "hidden_dim": 128, "n_classes": len(labels)}
    model = TextClassifier(len(vocab), **cfg).to(DEVICE)
    print(f"Словарь {len(vocab)} | обучение {len(Xtr)} | валидация {len(Xval)} | "
          f"{sum(p.numel() for p in model.parameters()):,} параметров")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight.to(DEVICE) if class_weight is not None else None)
    best, bad = -1.0, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        order = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            bi = order[i:i + 128]
            loss = loss_fn(model(Xtr[bi].to(DEVICE)), Ytr[bi].to(DEVICE))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        acc, _ = accuracy(model, Xval, Yval)
        mark = ""
        if acc > best:
            best, bad, mark = acc, 0, " *"
            torch.save({"model_state": model.state_dict(), "vocab": vocab, "labels": labels,
                        "max_len": max_len, "config": cfg}, out_path)
        else:
            bad += 1
        print(f"  эпоха {epoch:2d} | val acc {acc:.3f}{mark} | {time.time() - t0:.0f} с")
        if bad >= patience:
            break
    ck = torch.load(out_path, weights_only=False)
    model.load_state_dict(ck["model_state"])
    return model, vocab
