import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ─────────────────────────────────────────
# 1. ДАННЫЕ
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

# GPU если есть, иначе CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}")


# ─────────────────────────────────────────
# 2. CNN АРХИТЕКТУРА
# ─────────────────────────────────────────
#
# Обычная сеть (прошлый файл):
#   берёт картинку 28×28, разворачивает в 784 числа, теряет пространство
#
# CNN (эта сеть):
#   скользит фильтрами по картинке и ВИДИТ пространственные паттерны
#   (края, углы, кривые) — как глаз человека
#
# Путь картинки через сеть:
#   [1, 28, 28]          — входная картинка (1 канал = чёрно-белая)
#       ↓ Conv2d(1→32)
#   [32, 26, 26]         — 32 карты признаков
#       ↓ MaxPool2d
#   [32, 13, 13]         — уменьшили вдвое
#       ↓ Conv2d(32→64)
#   [64, 11, 11]         — 64 карты признаков
#       ↓ MaxPool2d
#   [64, 5, 5]           — уменьшили вдвое
#       ↓ Flatten
#   [1600]               — разворачиваем в вектор
#       ↓ Linear → Linear
#   [10]                 — ответ: вероятности для 10 цифр

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Свёрточные слои — извлекают признаки из картинки
        self.conv_layers = nn.Sequential(
            # Свёртка 1: 1 входной канал → 32 фильтра, ядро 3×3
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            # MaxPooling: берём максимум в окне 2×2 → уменьшаем картинку в 2 раза
            nn.MaxPool2d(kernel_size=2),

            # Свёртка 2: 32 канала → 64 фильтра
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Полносвязные слои — классифицируют по извлечённым признакам
        self.fc_layers = nn.Sequential(
            nn.Flatten(),           # [64, 5, 5] → [1600]
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Dropout(0.25),       # случайно отключаем 25% нейронов — защита от переобучения
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


model = CNN().to(device)
print(f"Параметров в сети: {sum(p.numel() for p in model.parameters()):,}")

# Покажем форму тензора на каждом слое
dummy = torch.zeros(1, 1, 28, 28).to(device)
print(f"\nФорма тензора по слоям:")
print(f"  Вход:          {list(dummy.shape)}")
dummy = model.conv_layers[0](dummy); print(f"  После Conv1:   {list(dummy.shape)}")
dummy = model.conv_layers[1](dummy)
dummy = model.conv_layers[2](dummy); print(f"  После Pool1:   {list(dummy.shape)}")
dummy = model.conv_layers[3](dummy); print(f"  После Conv2:   {list(dummy.shape)}")
dummy = model.conv_layers[4](dummy)
dummy = model.conv_layers[5](dummy); print(f"  После Pool2:   {list(dummy.shape)}")
dummy = dummy.flatten(1);            print(f"  После Flatten: {list(dummy.shape)}")


# ─────────────────────────────────────────
# 3. ОБУЧЕНИЕ
# ─────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
print()

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        predictions = model(images)
        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Считаем точность на тесте после каждой эпохи
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    accuracy = correct / len(test_data) * 100
    avg_loss = total_loss / len(train_loader)
    print(f"Эпоха {epoch+1}/{epochs} | Ошибка: {avg_loss:.4f} | Точность: {accuracy:.2f}%")


# ─────────────────────────────────────────
# 4. ИТОГ: где ошиблась сеть
# ─────────────────────────────────────────
print("\nПримеры ошибок сети:")
model.eval()
mistakes = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        wrong_mask = predicted != labels
        for i in range(len(labels)):
            if wrong_mask[i] and mistakes < 10:
                print(f"  Правильно: {labels[i].item()} | Сеть решила: {predicted[i].item()}")
                mistakes += 1

        if mistakes >= 10:
            break

print(f"\nВсего ошибок на 10 000 картинок: {10000 - correct}")
