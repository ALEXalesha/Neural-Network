import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ─────────────────────────────────────────
# 1. ДАННЫЕ
# ─────────────────────────────────────────
# Transforms: переводим картинку в тензор и нормализуем пиксели
# было: 0..255  →  стало: -1..1  (сети учиться проще)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Скачиваем датасет автоматически (первый раз ~11 МБ)
train_data = datasets.MNIST(root='data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# DataLoader: подаёт данные батчами по 64 картинки за раз
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

print(f"Обучающих примеров: {len(train_data)}")   # 60 000
print(f"Тестовых примеров:  {len(test_data)}")    # 10 000


# ─────────────────────────────────────────
# 2. АРХИТЕКТУРА СЕТИ
# ─────────────────────────────────────────
# Картинка MNIST: 28×28 пикселей = 784 числа
# Сеть: 784 → 256 → 128 → 10 (по одному нейрону на каждую цифру)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 256),   # полносвязный слой
            nn.ReLU(),             # функция активации (лучше сигмоиды для глубоких сетей)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)     # 10 выходов = 10 цифр
            # Softmax не нужен здесь — он внутри CrossEntropyLoss
        )

    def forward(self, x):
        x = x.view(-1, 784)   # "разворачиваем" картинку 28×28 в вектор 784
        return self.network(x)


model = NeuralNet()
print(f"\nПараметров в сети: {sum(p.numel() for p in model.parameters()):,}")


# ─────────────────────────────────────────
# 3. ФУНКЦИЯ ПОТЕРЬ И ОПТИМИЗАТОР
# ─────────────────────────────────────────
# CrossEntropyLoss: стандартная функция для задач классификации
# Adam: умный вариант gradient descent (сам подбирает learning rate)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ─────────────────────────────────────────
# 4. ОБУЧЕНИЕ
# ─────────────────────────────────────────
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Прямой проход
        predictions = model(images)
        loss = criterion(predictions, labels)

        # Обратный проход
        optimizer.zero_grad()   # сбрасываем старые градиенты
        loss.backward()         # считаем градиенты (backprop)
        optimizer.step()        # обновляем веса

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Эпоха {epoch+1}/{epochs} | Ошибка: {avg_loss:.4f}")


# ─────────────────────────────────────────
# 5. ТЕСТИРОВАНИЕ
# ─────────────────────────────────────────
model.eval()   # отключаем dropout и прочее (у нас нет, но хорошая практика)
correct = 0
total = 0

with torch.no_grad():   # не считаем градиенты — экономим память
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)   # берём цифру с максимальным score
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"\nТочность на тестовых данных: {accuracy:.2f}%")


# ─────────────────────────────────────────
# 6. ПРОВЕРЯЕМ НЕСКОЛЬКО КАРТИНОК ВРУЧНУЮ
# ─────────────────────────────────────────
print("\nПримеры предсказаний:")
examples = list(test_loader)[0]   # первый батч
images, labels = examples

model.eval()
with torch.no_grad():
    outputs = model(images[:10])
    predicted = outputs.argmax(dim=1)

for i in range(10):
    result = "OK" if predicted[i] == labels[i] else "FAIL"
    print(f"  Правильно: {labels[i].item()} | Предсказано: {predicted[i].item()} | {result}")
