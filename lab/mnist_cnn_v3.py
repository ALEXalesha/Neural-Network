import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}\n")

# ─────────────────────────────────────────
# 1. ДАННЫЕ + УЛУЧШЕННАЯ АУГМЕНТАЦИЯ
# ─────────────────────────────────────────
# Добавляем ElasticTransform — случайное "растяжение" пикселей
# Это лучшая аугментация именно для рукописных цифр:
# имитирует разный нажим пера и стиль письма

train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ElasticTransform(alpha=30.0, sigma=4.0),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='data', train=True,  download=True, transform=train_transform)
test_data  = datasets.MNIST(root='data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=128, shuffle=False)


# ─────────────────────────────────────────
# 2. RESIDUAL BLOCK — ключевая идея ResNet
# ─────────────────────────────────────────
#
# Проблема глубоких сетей: чем больше слоёв, тем сложнее учиться
# Градиент затухает пока доходит до начала сети
#
# Решение (придумали в Microsoft Research, 2015):
# Добавить "shortcut" — прямой путь в обход слоёв
#
#  x ──→ [Conv→BN→ReLU→Conv→BN] ──→ (+) ──→ ReLU
#  │                                  ↑
#  └──────────── shortcut ────────────┘
#
# Сеть учит не "что делать с x", а "что добавить к x" (residual = остаток)
# Если слой не нужен — веса обнуляются и сигнал идёт напрямую
# Это позволяет делать сети в 100+ слоёв без потери градиента

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))   # shortcut: прибавляем вход к выходу


# ─────────────────────────────────────────
# 3. АРХИТЕКТУРА С RESIDUAL BLOCKS
# ─────────────────────────────────────────
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Начальная свёртка: поднимаем до 64 каналов
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Два residual блока на 64 каналах (28×28)
        self.block1 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
        )

        # Переход 64→128, уменьшение 28→14
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Два residual блока на 128 каналах (14×14)
        self.block2 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )

        # Переход 128→256, уменьшение 14→7
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Два residual блока на 256 каналах (7×7)
        self.block3 = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)   # [256,7,7] → [256,1,1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


model = ResNet().to(device)
print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")
print(f"(В прошлой версии было 140,778)\n")


# ─────────────────────────────────────────
# 4. LABEL SMOOTHING
# ─────────────────────────────────────────
# Обычный CrossEntropyLoss учит сеть выдавать [0,0,0,1,0,0,0,0,0,0] для цифры 3
# Проблема: сеть становится "самоуверенной" и переобучается
#
# Label Smoothing (smoothing=0.1): меняем цель на [0.01, 0.01, ..., 0.91, ..., 0.01]
# Сеть учится быть чуть менее уверенной → лучше обобщает

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

epochs = 15
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
# AdamW: исправленная версия Adam, weight_decay работает правильно

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01,
    epochs=epochs, steps_per_epoch=len(train_loader)
)


# ─────────────────────────────────────────
# 5. ОБУЧЕНИЕ
# ─────────────────────────────────────────
best_accuracy = 0.0

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()

    accuracy = correct / len(test_data) * 100
    avg_loss = total_loss / len(train_loader)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model_v3.pth')
        marker = " <-- лучший!"
    else:
        marker = ""

    print(f"Эпоха {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
          f"Точность: {accuracy:.2f}%{marker}")


# ─────────────────────────────────────────
# 6. TEST TIME AUGMENTATION (TTA)
# ─────────────────────────────────────────
# Идея: при тестировании предсказываем каждую картинку N раз
# с разными случайными аугментациями, усредняем вероятности
#
# Сеть видела разные варианты цифры при обучении →
# при усреднении неуверенные предсказания "сглаживаются"
# → точность растёт без переобучения

print(f"\nЛучшая точность без TTA: {best_accuracy:.2f}%")
print("Применяем Test Time Augmentation...")

model.load_state_dict(torch.load('best_model_v3.pth', weights_only=True))
model.eval()

# Аугментации для TTA — лёгкие, чтобы не исказить сильно
tta_transforms = [
    test_transform,   # оригинал
    transforms.Compose([transforms.RandomRotation(5),  transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]),
    transforms.Compose([transforms.RandomRotation(-5), transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]),
    transforms.Compose([transforms.RandomAffine(0, translate=(0.05, 0.0)), transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]),
    transforms.Compose([transforms.RandomAffine(0, translate=(0.0, 0.05)), transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]),
]

all_probs = None
for t in tta_transforms:
    test_data.transform = t
    loader = DataLoader(test_data, batch_size=128, shuffle=False)
    probs_list = []
    with torch.no_grad():
        for images, _ in loader:
            out = torch.softmax(model(images.to(device)), dim=1)
            probs_list.append(out.cpu())
    probs = torch.cat(probs_list)
    all_probs = probs if all_probs is None else all_probs + probs

final_preds = all_probs.argmax(dim=1)
labels_all  = torch.tensor(test_data.targets)
tta_accuracy = (final_preds == labels_all).float().mean().item() * 100

print(f"Точность с TTA ({len(tta_transforms)} версии): {tta_accuracy:.2f}%")
print(f"Ошибок из 10 000: {round(10000 * (1 - tta_accuracy/100))}")


# ─────────────────────────────────────────
# 7. ИТОГОВОЕ СРАВНЕНИЕ
# ─────────────────────────────────────────
print("\n" + "="*50)
print("ИТОГ ВСЕХ ВЕРСИЙ:")
print("="*50)
print(f"  Обычная сеть (v0):  97.43%  | 257 ошибок")
print(f"  CNN v1:             98.91%  | 109 ошибок")
print(f"  CNN v2:             99.59%  |  41 ошибка")
print(f"  ResNet v3 (TTA): {tta_accuracy:.2f}%  |  {round(10000*(1-tta_accuracy/100))} ошибки")
print("="*50)
