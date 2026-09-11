"""
Базовая обработка изображений нейронными сетями
1. Автоматическое улучшение яркости/контраста (нейронка подбирает параметры)
2. Умное шумоподавление (denoising autoencoder)
3. Суперразрешение — увеличение изображения x2 без потери качества

Обычные фильтры просто применяют математику.
Нейросетевые — обучаются на примерах и понимают содержимое.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}\n")

# Загрузка MNIST для экспериментов
transform = transforms.ToTensor()
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader  = DataLoader(dataset, batch_size=256, shuffle=True)
test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=256)

# ─────────────────────────────────────────
# ЗАДАЧА 1: ШУМОПОДАВЛЕНИЕ (Denoising Autoencoder)
# ─────────────────────────────────────────
# Добавляем шум к картинке → учим сеть убирать его
print("="*50)
print("ЗАДАЧА 1: Шумоподавление (Denoising Autoencoder)")
print("="*50)

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder — сжимаем картинку
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),  # 28x28 → 28x28
            nn.MaxPool2d(2),                              # 28x28 → 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), # 14x14 → 14x14
            nn.MaxPool2d(2),                              # 14x14 → 7x7
        )
        # Decoder — восстанавливаем чистую картинку
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),  # 7x7 → 14x14
            nn.ConvTranspose2d(32, 1, 2, stride=2),  nn.Sigmoid(), # 14x14 → 28x28
        )

    def forward(self, x):
        return self.dec(self.enc(x))

denoiser = DenoisingAutoencoder().to(device)
opt_d = torch.optim.Adam(denoiser.parameters(), lr=1e-3)

NOISE_LEVEL = 0.4   # насколько сильный шум добавляем
EPOCHS_D    = 10

print(f"Уровень шума: {NOISE_LEVEL}")
for epoch in range(1, EPOCHS_D + 1):
    denoiser.train()
    total_loss = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        noisy = (imgs + NOISE_LEVEL * torch.randn_like(imgs)).clamp(0, 1)
        out   = denoiser(noisy)
        loss  = F.mse_loss(out, imgs)
        opt_d.zero_grad(); loss.backward(); opt_d.step()
        total_loss += loss.item()
    print(f"  Эпоха {epoch:2d}/{EPOCHS_D} | Loss: {total_loss/len(loader):.5f}")

torch.save(denoiser.state_dict(), 'denoiser_model.pth')
print("Модель сохранена!\n")

# ─────────────────────────────────────────
# ЗАДАЧА 2: СУПЕРРАЗРЕШЕНИЕ (Super-Resolution)
# ─────────────────────────────────────────
# Уменьшаем картинку 28x28 → 14x14, потом учим восстанавливать до 28x28
print("="*50)
print("ЗАДАЧА 2: Суперразрешение (x2 увеличение)")
print("="*50)

class SuperResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            # Pixel shuffle: увеличиваем разрешение x2
            nn.Conv2d(64, 4, 3, padding=1),
            nn.PixelShuffle(2),              # 64 канала → 16 + x2 разрешение
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: низкое разрешение 14x14
        # Сначала интерполируем до 28x28, потом улучшаем детали
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.net(x_up)

sr_model = SuperResNet().to(device)
opt_sr   = torch.optim.Adam(sr_model.parameters(), lr=1e-3)

EPOCHS_SR = 10
for epoch in range(1, EPOCHS_SR + 1):
    sr_model.train()
    total_loss = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)                                      # 28x28 оригинал
        low_res = F.interpolate(imgs, size=14, mode='bilinear',
                                align_corners=False)                # 14x14 уменьшенное
        out  = sr_model(low_res)                                    # 28x28 восстановленное
        loss = F.mse_loss(out, imgs)
        opt_sr.zero_grad(); loss.backward(); opt_sr.step()
        total_loss += loss.item()
    print(f"  Эпоха {epoch:2d}/{EPOCHS_SR} | Loss: {total_loss/len(loader):.5f}")

torch.save(sr_model.state_dict(), 'superres_model.pth')
print("Модель сохранена!\n")

# ─────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ─────────────────────────────────────────
denoiser.eval()
sr_model.eval()

test_imgs, _ = next(iter(test_loader))
test_imgs    = test_imgs[:8].to(device)
noisy_imgs   = (test_imgs + NOISE_LEVEL * torch.randn_like(test_imgs)).clamp(0, 1)
low_res_imgs = F.interpolate(test_imgs, size=14, mode='bilinear', align_corners=False)

with torch.no_grad():
    denoised  = denoiser(noisy_imgs)
    sr_output = sr_model(low_res_imgs)

# PSNR — метрика качества изображений (чем выше, тем лучше)
def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 10 * np.log10(1 / mse)

print(f"Результаты:")
print(f"  Шумоподавление:")
print(f"    PSNR шумной картинки:     {psnr(noisy_imgs, test_imgs):.2f} dB")
print(f"    PSNR после очистки:       {psnr(denoised, test_imgs):.2f} dB  (+)")
print(f"  Суперразрешение:")
bilinear = F.interpolate(low_res_imgs, size=28, mode='bilinear', align_corners=False)
print(f"    PSNR простой интерполяции: {psnr(bilinear, test_imgs):.2f} dB")
print(f"    PSNR суперразрешения:      {psnr(sr_output, test_imgs):.2f} dB  (+)")

# Сохраняем картинки
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
row_titles = ["Оригинал", "Зашумлённая", "После очистки", "Суперразрешение x2"]
rows = [
    test_imgs.cpu(),
    noisy_imgs.cpu(),
    denoised.cpu(),
    sr_output.cpu(),
]
for r, (row, title) in enumerate(zip(rows, row_titles)):
    for c in range(8):
        axes[r, c].imshow(row[c, 0], cmap='gray', vmin=0, vmax=1)
        axes[r, c].axis('off')
    axes[r, 0].set_ylabel(title, fontsize=10, rotation=0, ha='right', labelpad=80)

plt.suptitle('Обработка изображений нейронными сетями', fontsize=13)
plt.tight_layout()
plt.savefig('image_processing_results.png', dpi=120)
print("\nГрафик сохранён: image_processing_results.png")
