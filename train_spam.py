"""
Обучение SpamLSTM — бинарный классификатор спама (английский текст)
Сохраняет spam_model.pth с весами и словарём
"""
import sys, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

BASE   = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {DEVICE}\n")

# ── Датасет ───────────────────────────────────────────────────────────────────
SPAM = [
    "Congratulations! You are the winner of 1000000 dollars. Click here to claim your prize now!",
    "Free money! Limited time offer. Buy now and earn cash back instantly!",
    "You have been selected for a special lottery. Claim your reward today!",
    "URGENT: Your account has been compromised. Click the link to verify your password.",
    "Make money fast! Work from home and earn thousands per week!",
    "Hot singles in your area want to meet you tonight. Click here!",
    "Win a free iPhone! You are our lucky winner. Act now before it expires!",
    "Cheap medications online. No prescription needed. Order now!",
    "Get rich quick with our proven investment strategy. Guaranteed returns!",
    "Your PayPal account has been suspended. Verify immediately to avoid closure.",
    "Exclusive deal: 90% off designer goods. Limited stock. Buy today!",
    "You won a vacation! Claim your free trip to Las Vegas. Call now!",
    "Earn 5000 dollars per day from home. No experience needed. Join free!",
    "WINNER WINNER! You have been chosen for a special cash reward!",
    "Discount pills delivered to your door. No questions asked. Order online!",
    "Your credit card information needs to be updated. Click to secure your account.",
    "Free casino chips! Sign up now and get 500 bonus spins!",
    "Make extra income online. Thousands already earning from home. Join us!",
    "Special promotion: Buy one get ten free. Offer expires tonight!",
    "You owe taxes! Pay immediately or face legal consequences. Call this number.",
    "Lose weight fast with this secret pill. Doctors hate this trick!",
    "Luxury watches at factory prices. Click to see our full collection!",
    "Alert: unusual activity on your bank account. Log in to secure funds!",
    "Earn commission by referring friends. Join our affiliate program today!",
    "Free gift card worth 500 dollars. Fill out this quick survey now!",
    "Your loan has been pre-approved! Get cash in 24 hours. Apply now!",
    "Hot investment opportunity! Bitcoin doubled last month. Invest today!",
    "Meet wealthy singles near you! Sign up for free dating service now.",
    "Congratulations you have been selected for our exclusive VIP program!",
    "Click to unsubscribe from spam but first collect your reward money!",
    "Nigerian prince needs help transferring funds. Huge reward guaranteed.",
    "You are a winner! Provide your bank account to receive your prize.",
    "Limited time: refinance your home and save thousands. Call now free!",
    "Increase your income by 300 percent guaranteed. No risk investment!",
    "Your subscription is expiring. Click to renew and get bonus months free!",
]

HAM = [
    "Hi, can we schedule a meeting for tomorrow afternoon to discuss the project?",
    "Please find attached the quarterly report. Let me know if you have questions.",
    "Thanks for your help with the presentation. It went really well!",
    "Could you review the pull request when you have a chance? No rush.",
    "The team meeting has been moved to 3pm. Please update your calendar.",
    "I will be out of office next week. Contact my colleague for urgent matters.",
    "Happy birthday! Hope you have a wonderful day with your family.",
    "The server deployment is scheduled for this weekend. All clear on your end?",
    "Can you send me the budget report when it is ready? Thanks in advance.",
    "Just wanted to check in on how the integration tests are coming along.",
    "Great job on the client demo yesterday! The feedback was very positive.",
    "Reminder: expense reports are due by end of this month.",
    "I am running 10 minutes late for our call. Will connect shortly.",
    "Please review the attached document and share your comments by Friday.",
    "The new feature is now deployed to staging. Please test it when possible.",
    "Could we move our 2pm call to 4pm? Something came up on my end.",
    "Thanks for the quick response. The issue has been resolved on our side.",
    "Looking forward to seeing you at the conference next week in Berlin.",
    "The client approved the proposal! We can start onboarding next Monday.",
    "Can you share the login credentials for the staging environment please?",
    "Good morning! Just a reminder about the standup at 9am today.",
    "I reviewed the code and left some comments. Overall looks great though!",
    "Please let me know if you need any additional information for the report.",
    "The bug has been fixed in the latest release. Update when you get a chance.",
    "Lunch at 1pm today? There is a new place near the office we could try.",
    "Your package has been shipped and will arrive in 3 to 5 business days.",
    "Here are the notes from today's meeting. Let me know if I missed anything.",
    "Can we sync up briefly tomorrow morning to align on the roadmap?",
    "The documentation has been updated with the new API changes.",
    "Thank you for attending the webinar. Here is the recording link.",
    "Welcome to the team! Your onboarding session is scheduled for Monday.",
    "I have finished the analysis. Sending the results in a separate email.",
    "Just confirmed the venue for the team event next Thursday evening.",
    "The client called with a few questions. I will handle it and update you.",
    "Feedback on the design mockups: I love the new color scheme!",
]

# Augmentation — небольшие вариации
def augment(texts, label, factor=8):
    import random
    random.seed(42)
    out = [(t, label) for t in texts]
    for _ in range(factor - 1):
        for t in texts:
            words = t.split()
            if len(words) > 4:
                i = random.randint(0, len(words) - 1)
                words[i] = words[i].lower() if random.random() > 0.5 else words[i].upper()
            out.append((" ".join(words), label))
    return out

data = augment(SPAM, 1) + augment(HAM, 0)

import random
random.seed(42)
random.shuffle(data)

texts, labels = zip(*data)

# ── Словарь ───────────────────────────────────────────────────────────────────
cnt = Counter()
for t in texts:
    cnt.update(t.lower().split())

VOCAB = {"<PAD>": 0, "<UNK>": 1}
for w, _ in cnt.most_common(2000):
    VOCAB[w] = len(VOCAB)

VOCAB_SIZE = len(VOCAB)
MAX_LEN    = 40

def encode(text):
    ids = [VOCAB.get(w, 1) for w in text.lower().split()[:MAX_LEN]]
    ids += [0] * (MAX_LEN - len(ids))
    return ids

X_all = torch.tensor([encode(t) for t in texts], dtype=torch.long)
Y_all = torch.tensor(labels, dtype=torch.long)

split   = int(0.8 * len(X_all))
X_train = X_all[:split].to(DEVICE)
Y_train = Y_all[:split].to(DEVICE)
X_test  = X_all[split:].to(DEVICE)
Y_test  = Y_all[split:].to(DEVICE)

print(f"Данных: {len(X_all)} | Спам: {sum(labels)} | Не спам: {len(labels)-sum(labels)}")

# ── Модель ────────────────────────────────────────────────────────────────────
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

model     = SpamLSTM(VOCAB_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# ── Обучение ──────────────────────────────────────────────────────────────────
EPOCHS = 50
BATCH  = 64
print(f"\nОбучение: {EPOCHS} эпох, batch={BATCH}\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = torch.randperm(split)
    total_loss = 0
    for i in range(0, split, BATCH):
        xb = X_train[idx[i:i+BATCH]]
        yb = Y_train[idx[i:i+BATCH]]
        logits = model(xb)
        loss   = criterion(logits, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            logits_t = model(X_test)
            acc = (logits_t.argmax(1) == Y_test).float().mean().item()
        print(f"Эпоха {epoch:3d}/{EPOCHS} | Loss: {total_loss:.3f} | Acc: {acc*100:.1f}%")

# ── Тест ──────────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    logits_t = model(X_test)
    acc = (logits_t.argmax(1) == Y_test).float().mean().item()
print(f"\nТочность на тестовых данных: {acc*100:.1f}%")

# Примеры
examples = [
    ("Congratulations! You won a free prize. Click here now!", 1),
    ("Hi, can we meet tomorrow at 3pm to discuss the project?", 0),
    ("Earn money fast! No experience needed. Join now for free!", 1),
    ("Please review the attached report and let me know your thoughts.", 0),
]
print("\nПроверка на примерах:")
for text, true_label in examples:
    ids  = torch.tensor([encode(text)]).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(ids), dim=1)[0].cpu().numpy()
    pred = probs.argmax()
    mark = "✓" if pred == true_label else "✗"
    print(f"  {mark} [{('HAM','SPAM')[pred]}] {text[:60]}...")

# ── Сохранение ────────────────────────────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "vocab":       VOCAB,
    "vocab_size":  VOCAB_SIZE,
    "max_len":     MAX_LEN,
    "embed_dim":   64,
    "hidden_dim":  128,
}, BASE / "spam_model.pth")

print("\nМодель сохранена: spam_model.pth")
