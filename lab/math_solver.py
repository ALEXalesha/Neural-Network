import torch
import torch.nn as nn
import random

# ─────────────────────────────────────────
# 1. ДАННЫЕ
# ─────────────────────────────────────────
CHARS = "0123456789+-*/=<>^~"  # ~ = знак минуса в ответе
PAD, END, SOS = '<', '>', '^'
NEG = '~'  # отдельный символ для отрицательных чисел
c2i   = {c: i for i, c in enumerate(CHARS)}
i2c   = {i: c for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
MAX_LEN = 16

def make_example():
    op = random.choice(['+', '-', '*'])
    if op == '*':
        a, b = random.randint(0, 99), random.randint(0, 99)
    else:
        a, b = random.randint(0, 999), random.randint(0, 999)

    if op == '+': ans = a + b
    elif op == '-': ans = a - b
    else: ans = a * b

    q = f"{a}{op}{b}="
    # Отрицательные числа: -5 → "~5" перевёрнуто "5~", потом обратно "~5"
    ans_str = str(abs(ans)) + (NEG if ans < 0 else '')
    a = SOS + ans_str + END
    return q, a

def encode(s, length=MAX_LEN):
    ids = [c2i.get(c, 0) for c in s]
    ids += [c2i[PAD]] * (length - len(ids))
    return ids[:length]

def get_batch(bs=512):
    qs, as_ = zip(*[make_example() for _ in range(bs)])
    return (torch.tensor([encode(q) for q in qs], dtype=torch.long),
            torch.tensor([encode(a) for a in as_], dtype=torch.long))

print("Примеры:")
for _ in range(4):
    q, a = make_example()
    print(f"  {q} -> {a}")
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}\n")

# ─────────────────────────────────────────
# 2. АРХИТЕКТУРА — LSTM Seq2Seq с attention
# ─────────────────────────────────────────
#
# Encoder: читает вопрос "123+456=" → скрытое состояние
# Attention: decoder смотрит на все шаги encoder
# Decoder: генерирует ответ "579>" символ за символом

class Encoder(nn.Module):
    def __init__(self, vocab, embed=64, hidden=256, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=c2i[PAD])
        self.lstm  = nn.LSTM(embed, hidden, layers, batch_first=True, dropout=0.1)

    def forward(self, x):
        return self.lstm(self.embed(x))  # out: [B,T,H], hidden: ([L,B,H],[L,B,H])


class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, hidden)
        self.v    = nn.Linear(hidden, 1, bias=False)

    def forward(self, dec_h, enc_out):
        # dec_h: [B, H], enc_out: [B, T, H]
        T = enc_out.size(1)
        dec_h = dec_h.unsqueeze(1).repeat(1, T, 1)       # [B, T, H]
        energy = torch.tanh(self.attn(torch.cat([dec_h, enc_out], dim=2)))
        return torch.softmax(self.v(energy).squeeze(2), dim=1)  # [B, T]


class Decoder(nn.Module):
    def __init__(self, vocab, embed=64, hidden=256, layers=2):
        super().__init__()
        self.embed   = nn.Embedding(vocab, embed, padding_idx=c2i[PAD])
        self.attn    = Attention(hidden)
        self.lstm    = nn.LSTM(embed + hidden, hidden, layers, batch_first=True, dropout=0.1)
        self.fc      = nn.Linear(hidden * 2, vocab)

    def forward(self, token, hidden, enc_out):
        emb = self.embed(token.unsqueeze(1))               # [B, 1, E]
        dec_h = hidden[0][-1]                              # последний слой: [B, H]
        a = self.attn(dec_h, enc_out)                      # [B, T]
        context = (a.unsqueeze(1) @ enc_out)               # [B, 1, H]
        out, hidden = self.lstm(torch.cat([emb, context], dim=2), hidden)
        pred = self.fc(torch.cat([out.squeeze(1), context.squeeze(1)], dim=1))
        return pred, hidden


class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(VOCAB)
        self.decoder = Decoder(VOCAB)

    def forward(self, src, tgt, teacher_forcing=0.5):
        enc_out, hidden = self.encoder(src)
        B, T_out = tgt.shape
        preds = torch.zeros(B, T_out, VOCAB, device=src.device)

        token = tgt[:, 0]  # SOS
        for t in range(1, T_out):
            pred, hidden = self.decoder(token, hidden, enc_out)
            preds[:, t] = pred
            # Teacher forcing: иногда подаём правильный ответ, иногда своё предсказание
            token = tgt[:, t] if random.random() < teacher_forcing else pred.argmax(1)

        return preds


model  = Seq2Seq().to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Параметров: {params:,}\n")

# ─────────────────────────────────────────
# 3. ОБУЧЕНИЕ
# ─────────────────────────────────────────
criterion = nn.CrossEntropyLoss(ignore_index=c2i[PAD])
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

epochs = 30000
for epoch in range(1, epochs + 1):
    model.train()
    src, tgt = get_batch(512)
    src, tgt = src.to(device), tgt.to(device)

    preds = model(src, tgt, teacher_forcing=0.5)
    loss  = criterion(preds[:, 1:].reshape(-1, VOCAB), tgt[:, 1:].reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Эпоха {epoch:4d}/{epochs} | Loss: {loss.item():.4f}")

# ─────────────────────────────────────────
# 4. ГЕНЕРАЦИЯ (без teacher forcing)
# ─────────────────────────────────────────
def solve(question):
    model.eval()
    src = torch.tensor([encode(question)], dtype=torch.long).to(device)
    with torch.no_grad():
        enc_out, hidden = model.encoder(src)
        token  = torch.tensor([c2i[SOS]], device=device)
        result = []
        for _ in range(MAX_LEN):
            pred, hidden = model.decoder(token, hidden, enc_out)
            token = pred.argmax(1)
            ch = i2c[token.item()]
            if ch == END: break
            if ch != PAD: result.append(ch)
    raw = ''.join(result)
    # Если заканчивается на ~, это отрицательное число
    if raw.endswith(NEG):
        return '-' + raw[:-1]
    return raw

# ─────────────────────────────────────────
# 5. ТЕСТ
# ─────────────────────────────────────────
print("\n" + "="*50)
correct, total = 0, 500
for _ in range(total):
    q, _ = make_example()
    pred = solve(q)
    expected = str(eval(q.replace('=', '')))
    if pred == expected:
        correct += 1

print(f"Точность: {correct}/{total} = {correct/total*100:.1f}%\n")

print("Конкретные примеры:")
for q in ["2+2=", "10+15=", "100-37=", "99*99=", "123+456=", "999-1=", "25*4=", "0-5=", "50*50=", "999+999="]:
    pred = solve(q)
    expected = str(eval(q.replace('=', '')))
    ok = "OK" if pred == expected else f"FAIL (правильно: {expected})"
    print(f"  {q:<12} {pred:>6}  {ok}")
