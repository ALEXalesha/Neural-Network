"""
Conditional DCGAN — генерация цифр по выбору (0-9)

Улучшения по сравнению с базовым GAN:
  - Свёрточная архитектура (DCGAN) — гораздо чётче цифры
  - Условная генерация (cGAN) — выбираешь какую цифру генерировать
  - BatchNorm + LeakyReLU — стабильное обучение
  - Spectral Normalization в дискриминаторе — против mode collapse

Архитектура:
  Generator: шум(100) + метка(10) → 1×28×28
  Discriminator: картинка(1×28×28) + метка → real/fake
"""
import sys, io
if hasattr(sys.stdout, "buffer"): sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm
import os

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}\n")

Z_DIM      = 100   # размер вектора шума
N_CLASSES  = 10    # цифры 0-9
EMBED_DIM  = 10    # размер эмбеддинга метки
IMG_SIZE   = 28

# ─────────────────────────────────────────
# АРХИТЕКТУРЫ
# ─────────────────────────────────────────

class Generator(nn.Module):
    """Шум(100) + метка(0-9) → Картинка(1×28×28)"""
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, EMBED_DIM)
        # Вход: z(100) + label_emb(10) = 110
        self.net = nn.Sequential(
            # 110 → 7×7×256
            nn.Linear(Z_DIM + EMBED_DIM, 7 * 7 * 256),
            nn.Unflatten(1, (256, 7, 7)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7×7 → 14×14
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 14×14 → 28×28
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # финальная свёртка
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        emb = self.label_emb(labels)          # (B, 10)
        inp = torch.cat([z, emb], dim=1)       # (B, 110)
        return self.net(inp)


class Discriminator(nn.Module):
    """Картинка(1×28×28) + метка → вероятность real/fake"""
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, IMG_SIZE * IMG_SIZE)
        # Вход: изображение (1 канал) + метка как карта (1 канал) = 2 канала
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(2, 64, 4, 2, 1)),    # 28→14
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),  # 14→7
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(128, 256, 3, 1, 1)), # 7→7
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            spectral_norm(nn.Linear(256 * 7 * 7, 1)),
        )

    def forward(self, img, labels):
        label_map = self.label_emb(labels).view(-1, 1, IMG_SIZE, IMG_SIZE)
        x = torch.cat([img, label_map], dim=1)  # (B, 2, 28, 28)
        return self.net(x)


G = Generator().to(device)
D = Discriminator().to(device)

g_params = sum(p.numel() for p in G.parameters())
d_params = sum(p.numel() for p in D.parameters())
print(f"Generator:     {g_params:,} параметров")
print(f"Discriminator: {d_params:,} параметров\n")

# ─────────────────────────────────────────
# ДАННЫЕ
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader  = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

# ─────────────────────────────────────────
# ОБУЧЕНИЕ
# ─────────────────────────────────────────
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()

# Фиксированный шум для мониторинга: 10 цифр × 8 вариантов = 80 картинок
fixed_noise  = torch.randn(80, Z_DIM, device=device)
fixed_labels = torch.tensor([i for i in range(10) for _ in range(8)], device=device)

EPOCHS = 50
os.makedirs('gan_progress', exist_ok=True)

print("Обучение Conditional DCGAN...")
print("Эпоха | D_loss | G_loss")
print("-" * 35)

g_losses, d_losses = [], []

for epoch in range(1, EPOCHS + 1):
    g_epoch, d_epoch = 0.0, 0.0

    for real_imgs, real_labels in loader:
        real_imgs   = real_imgs.to(device)
        real_labels = real_labels.to(device)
        B = real_imgs.size(0)

        real_lbl = torch.ones(B, 1, device=device)
        fake_lbl = torch.zeros(B, 1, device=device)

        # ── Шаг 1: Discriminator ──
        z     = torch.randn(B, Z_DIM, device=device)
        fake_labels = torch.randint(0, N_CLASSES, (B,), device=device)
        fakes = G(z, fake_labels).detach()

        d_real = criterion(D(real_imgs, real_labels), real_lbl)
        d_fake = criterion(D(fakes, fake_labels),     fake_lbl)
        d_loss = (d_real + d_fake) / 2

        opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        # ── Шаг 2: Generator ──
        z     = torch.randn(B, Z_DIM, device=device)
        fake_labels = torch.randint(0, N_CLASSES, (B,), device=device)
        fakes = G(z, fake_labels)
        g_loss = criterion(D(fakes, fake_labels), real_lbl)

        opt_G.zero_grad(); g_loss.backward(); opt_G.step()

        g_epoch += g_loss.item()
        d_epoch += d_loss.item()

    n = len(loader)
    g_losses.append(g_epoch / n)
    d_losses.append(d_epoch / n)

    if epoch % 5 == 0:
        print(f"  {epoch:3d}/{EPOCHS} | D: {d_epoch/n:.4f} | G: {g_epoch/n:.4f}")

        # Сохраняем сетку: строки = цифры 0-9, столбцы = 8 вариантов
        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise, fixed_labels).cpu()
        G.train()

        fig, axes = plt.subplots(10, 8, figsize=(10, 12))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i, 0], cmap='gray', vmin=-1, vmax=1)
            ax.axis('off')
        for i in range(10):
            axes[i, 0].set_ylabel(str(i), rotation=0, labelpad=15, fontsize=12)
        plt.suptitle(f'Conditional DCGAN — Эпоха {epoch}', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'gan_progress/epoch_{epoch:03d}.png', dpi=80)
        plt.close()

torch.save(G.state_dict(), 'gan_generator.pth')
torch.save(D.state_dict(), 'gan_discriminator.pth')
print("\nМодели сохранены: gan_generator.pth, gan_discriminator.pth\n")

# ─────────────────────────────────────────
# ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ
# ─────────────────────────────────────────
G.eval()
with torch.no_grad():
    final_samples = G(fixed_noise, fixed_labels).cpu()

fig, axes = plt.subplots(10, 8, figsize=(12, 14))
for i, ax in enumerate(axes.flat):
    ax.imshow(final_samples[i, 0], cmap='gray', vmin=-1, vmax=1)
    ax.axis('off')
for i in range(10):
    axes[i, 0].set_ylabel(str(i), rotation=0, labelpad=15, fontsize=14, fontweight='bold')
plt.suptitle('Conditional DCGAN — Финал (строки = цифры 0-9)', fontsize=14)
plt.tight_layout()
plt.savefig('gan_results.png', dpi=120)
print("Финальные результаты: gan_results.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(g_losses, label='Generator', color='blue')
axes[0].plot(d_losses, label='Discriminator', color='red')
axes[0].axhline(y=0.693, color='gray', linestyle='--', label='Идеал')
axes[0].set_title('Потери'); axes[0].legend(); axes[0].grid(True)
axes[1].imshow(plt.imread('gan_results.png'))
axes[1].axis('off'); axes[1].set_title('Сгенерированные цифры')
plt.tight_layout()
plt.savefig('gan_losses.png', dpi=100)
print("График потерь: gan_losses.png")
print("\nГотово! Теперь модель умеет генерировать конкретные цифры.")
