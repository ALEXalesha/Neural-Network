import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}\n")

# ─────────────────────────────────────────
# 1. ДАННЫЕ + АУГМЕНТАЦИЯ
# ─────────────────────────────────────────
# Аугментация = искусственно разнообразим обучающие данные
# Сеть будет видеть каждую цифру немного по-разному каждую эпоху
# → меньше переобучения, лучше обобщение

train_transform = transforms.Compose([
    transforms.RandomRotation(10),          # случайный поворот ±10°
    transforms.RandomAffine(                # случайный сдвиг
        degrees=0, translate=(0.1, 0.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Тестовые данные НЕ аугментируем — хотим честную оценку
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='data', train=True,  download=True, transform=train_transform)
test_data  = datasets.MNIST(root='data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=128, shuffle=False, num_workers=0)


# ─────────────────────────────────────────
# 2. УЛУЧШЕННАЯ АРХИТЕКТУРА
# ─────────────────────────────────────────
# Новое по сравнению с v1:
#
# BatchNorm — нормализует активации внутри батча
#   Без него: каждый слой получает "разъезжающиеся" числа → учится медленно
#   С ним:    числа всегда в нормальном диапазоне → учится быстро и стабильно
#
# Больше фильтров (32→64→128) — сеть замечает больше паттернов
#
# Путь картинки:
#   [1, 28, 28]
#       ↓ Conv(1→32) + BN + ReLU
#   [32, 28, 28]     ← padding=1 сохраняет размер
#       ↓ Conv(32→32) + BN + ReLU + MaxPool
#   [32, 14, 14]
#       ↓ Conv(32→64) + BN + ReLU
#   [64, 14, 14]
#       ↓ Conv(64→64) + BN + ReLU + MaxPool
#   [64, 7, 7]
#       ↓ Conv(64→128) + BN + ReLU
#   [128, 7, 7]
#       ↓ GlobalAveragePooling
#   [128]            ← среднее по каждой карте признаков
#       ↓ Linear
#   [10]

class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=False):
            # Блок: Conv → BatchNorm → ReLU → (MaxPool опционально)
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),   # нормализация
                nn.ReLU(),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(1,   32),          # [1,28,28]  → [32,28,28]
            conv_block(32,  32, pool=True),  # [32,28,28] → [32,14,14]
            nn.Dropout2d(0.1),            # Dropout для свёрточных слоёв

            conv_block(32,  64),          # [32,14,14] → [64,14,14]
            conv_block(64,  64, pool=True),  # [64,14,14] → [64,7,7]
            nn.Dropout2d(0.1),

            conv_block(64, 128),          # [64,7,7]   → [128,7,7]
        )

        # GlobalAveragePooling: вместо Flatten берём среднее по каждой карте
        # [128, 7, 7] → [128] — меньше параметров, меньше переобучения
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


model = ImprovedCNN().to(device)
print(f"Параметров в сети: {sum(p.numel() for p in model.parameters()):,}")
print(f"(В прошлой версии было 225,034)\n")


# ─────────────────────────────────────────
# 3. ОПТИМИЗАТОР + ПЛАНИРОВЩИК LR
# ─────────────────────────────────────────
# OneCycleLR: learning rate сначала растёт, потом плавно падает
# Это позволяет учиться быстро вначале и точно настроиться в конце
# → можно достичь хорошего результата за меньше эпох

epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# weight_decay — L2 регуляризация: штрафует большие веса → меньше переобучения

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=epochs,
    steps_per_epoch=len(train_loader)
)


# ─────────────────────────────────────────
# 4. ОБУЧЕНИЕ
# ─────────────────────────────────────────
best_accuracy = 0.0

for epoch in range(epochs):
    # --- Обучение ---
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        predictions = model(images)
        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()   # обновляем learning rate каждый батч

        total_loss += loss.item()

    # --- Тест ---
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    accuracy = correct / len(test_data) * 100
    avg_loss = total_loss / len(train_loader)
    lr_now = scheduler.get_last_lr()[0]

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        marker = " <-- лучший!"
    else:
        marker = ""

    print(f"Эпоха {epoch+1:2d}/{epochs} | "
          f"Loss: {avg_loss:.4f} | "
          f"Точность: {accuracy:.2f}% | "
          f"LR: {lr_now:.5f}{marker}")


# ─────────────────────────────────────────
# 5. ИТОГ
# ─────────────────────────────────────────
print(f"\nЛучшая точность: {best_accuracy:.2f}%")
print(f"Ошибок из 10 000: {round(10000 * (1 - best_accuracy/100))}")

# Загружаем лучшие веса и смотрим на ошибки
model.load_state_dict(torch.load('best_model.pth', weights_only=True))
model.eval()

# Матрица ошибок: какие цифры путает сеть
confusion = torch.zeros(10, 10, dtype=torch.int)
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predicted = model(images).argmax(dim=1)
        for t, p in zip(labels, predicted):
            confusion[t][p] += 1

print("\nМатрица ошибок (строка=правильная цифра, столбец=что сказала сеть):")
print("     " + "  ".join(str(i) for i in range(10)))
for i in range(10):
    row = "  ".join(
        f"\033[91m{confusion[i][j]:3d}\033[0m" if i != j and confusion[i][j] > 0
        else f"{confusion[i][j]:3d}"
        for j in range(10)
    )
    print(f"  {i}: {row}")
