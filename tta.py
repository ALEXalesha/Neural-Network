import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Та же архитектура что в v3
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
        return self.relu(x + self.block(x))

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.block1 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.down1  = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.block2 = nn.Sequential(ResidualBlock(128), ResidualBlock(128))
        self.down2  = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.block3 = nn.Sequential(ResidualBlock(256), ResidualBlock(256))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 10))

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)

model = ResNet().to(device)
model.load_state_dict(torch.load('best_model_v3.pth', weights_only=True))
model.eval()

# ─────────────────────────────────────────
# TEST TIME AUGMENTATION
# ─────────────────────────────────────────
# Каждое изображение предсказываем 7 раз с разными лёгкими трансформациями
# и усредняем вероятности → менее уверенные предсказания становятся точнее

norm = transforms.Normalize((0.5,), (0.5,))

tta_transforms = [
    transforms.Compose([transforms.ToTensor(), norm]),                                              # оригинал
    transforms.Compose([transforms.RandomRotation(5),  transforms.ToTensor(), norm]),               # поворот +5°
    transforms.Compose([transforms.RandomRotation(5),  transforms.ToTensor(), norm]),               # поворот ещё раз
    transforms.Compose([transforms.RandomAffine(0, translate=(0.05, 0.0)), transforms.ToTensor(), norm]),  # сдвиг вправо
    transforms.Compose([transforms.RandomAffine(0, translate=(0.05, 0.0)), transforms.ToTensor(), norm]),  # сдвиг влево
    transforms.Compose([transforms.RandomAffine(0, translate=(0.0, 0.05)), transforms.ToTensor(), norm]),  # сдвиг вниз
    transforms.Compose([transforms.RandomAffine(0, translate=(0.0, 0.05)), transforms.ToTensor(), norm]),  # сдвиг вверх
]

test_data = datasets.MNIST(root='data', train=False, download=False, transform=tta_transforms[0])

print(f"Применяем TTA ({len(tta_transforms)} версий каждой картинки)...")

all_probs = None
for i, t in enumerate(tta_transforms):
    test_data.transform = t
    loader = DataLoader(test_data, batch_size=256, shuffle=False)
    probs_list = []
    with torch.no_grad():
        for images, _ in loader:
            out = torch.softmax(model(images.to(device)), dim=1)
            probs_list.append(out.cpu())
    probs = torch.cat(probs_list)
    all_probs = probs if all_probs is None else all_probs + probs
    print(f"  Версия {i+1}/{len(tta_transforms)} готова")

final_preds  = all_probs.argmax(dim=1)
labels_all   = torch.tensor(test_data.targets)
tta_accuracy = (final_preds == labels_all).float().mean().item() * 100
errors       = (final_preds != labels_all).sum().item()

print(f"\nТочность с TTA: {tta_accuracy:.2f}%")
print(f"Ошибок из 10 000: {errors}")

print("\n" + "="*50)
print("ИТОГ ВСЕХ ВЕРСИЙ:")
print("="*50)
print(f"  Обычная сеть:    97.43%  | 257 ошибок")
print(f"  CNN v1:          98.91%  | 109 ошибок")
print(f"  CNN v2:          99.59%  |  41 ошибка")
print(f"  ResNet (без TTA):99.72%  |  28 ошибок")
print(f"  ResNet + TTA:  {tta_accuracy:.2f}%  |  {errors} ошибок")
print("="*50)
