"""Переобучение MNIST с аугментацией толстых штрихов для веб-canvas"""
import sys, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch, torch.nn as nn
import torchvision, torchvision.transforms as T
from PIL import Image, ImageFilter
import numpy as np
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
BASE = Path(__file__).parent

class RandomDilate:
    """Утолщает штрихи как при рисовании на canvas"""
    def __call__(self, img):
        if np.random.random() > 0.4:
            size = np.random.choice([3, 5, 7, 9])
            return img.filter(ImageFilter.MaxFilter(size))
        return img

transform_train = T.Compose([
    T.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.7, 1.3), shear=10),
    RandomDilate(),
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,)),
])
transform_test = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

train_ds = torchvision.datasets.MNIST(BASE / 'data', train=True,  download=True, transform=transform_train)
test_ds  = torchvision.datasets.MNIST(BASE / 'data', train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(x + self.block(x))

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.block1 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.down1  = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.block2 = nn.Sequential(ResidualBlock(128), ResidualBlock(128))
        self.down2  = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.block3 = nn.Sequential(ResidualBlock(256), ResidualBlock(256))
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 10))
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x); x = self.down1(x)
        x = self.block2(x); x = self.down2(x)
        x = self.block3(x); x = self.gap(x)
        return self.classifier(x)

model = ResNet().to(device)
# Загружаем предобученные веса и дообучаем
try:
    model.load_state_dict(torch.load(BASE / 'best_model_v3.pth', weights_only=True, map_location=device))
    print("Загружена предобученная модель, дообучаем...")
except: print("Обучаем с нуля...")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
criterion = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(1, 16):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total * 100
    print(f"Epoch {epoch:2d}/15 | Test acc: {acc:.2f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), BASE / 'mnist_web.pth')

print(f"\nЛучшая точность: {best_acc:.2f}%")
print("Сохранено: mnist_web.pth")
