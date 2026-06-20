"""
MNIST — рисуй цифру мышкой и нейросеть угадает
"""
import sys, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import tkinter as tk
from tkinter import font as tkfont
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

BASE   = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Модель (та же что в web_app.py) ──────────────────
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(self.block(x) + x)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.block1 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.pool1  = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(ResidualBlock(128), ResidualBlock(128))
        self.pool2  = nn.MaxPool2d(2)
        self.block3 = nn.Sequential(ResidualBlock(256), ResidualBlock(256))
        self.pool3  = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 10))
    def forward(self, x):
        x = self.pool1(self.block1(self.stem(x)))
        x = self.pool2(self.block2(nn.functional.interpolate(x, scale_factor=1)))
        return self.classifier(self.pool3(self.block3(x)))

def load_model():
    model = ResNet().to(DEVICE)
    path = BASE / "mnist_web.pth"
    if not path.exists():
        path = BASE / "best_model_v3.pth"
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"Модель загружена: {path.name}")
    return model

def preprocess(canvas_arr):
    """Предобработка нарисованного изображения как в MNIST"""
    from PIL import Image
    arr = canvas_arr.astype(np.uint8)
    rows = np.any(arr > 20, axis=1)
    cols = np.any(arr > 20, axis=0)
    if not rows.any() or not cols.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    h, w = r1 - r0, c1 - c0
    side = max(h, w)
    pad  = max(int(side * 0.3), 10)
    cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
    half = side // 2 + pad
    r0 = max(0, cr - half); r1 = min(arr.shape[0], cr + half)
    c0 = max(0, cc - half); c1 = min(arr.shape[1], cc + half)
    arr = arr[r0:r1, c0:c1]
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    img = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)
    canvas = np.zeros((28, 28), dtype=np.float32)
    y0 = (28 - new_h) // 2
    x0 = (28 - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = np.array(img, dtype=np.float32)
    arr = (canvas / 255.0 - 0.1307) / 0.3081
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)

# ── GUI ──────────────────────────────────────────────
class App:
    CANVAS_SIZE = 280
    BRUSH_SIZE  = 16

    def __init__(self, root):
        self.root  = root
        self.model = load_model()
        root.title("MNIST — нарисуй цифру")
        root.configure(bg="#1a1a2e")
        root.resizable(False, False)

        # Холст для рисования
        self.canvas = tk.Canvas(root, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
                                bg="black", cursor="crosshair",
                                highlightthickness=2, highlightbackground="#444")
        self.canvas.grid(row=0, column=0, padx=20, pady=20, rowspan=6)

        # Массив пикселей
        self.pixels = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE), dtype=np.float32)

        # Правая панель
        big_font  = tkfont.Font(family="Arial", size=72, weight="bold")
        conf_font = tkfont.Font(family="Arial", size=14)
        prob_font = tkfont.Font(family="Arial", size=11)

        tk.Label(root, text="Нейросеть думает:", bg="#1a1a2e", fg="#888",
                 font=conf_font).grid(row=0, column=1, pady=(20,0), sticky="s")

        self.lbl_digit = tk.Label(root, text="?", bg="#1a1a2e", fg="#4fc3f7",
                                   font=big_font, width=3)
        self.lbl_digit.grid(row=1, column=1)

        self.lbl_conf = tk.Label(root, text="", bg="#1a1a2e", fg="#aaa", font=conf_font)
        self.lbl_conf.grid(row=2, column=1)

        # Вероятности
        frame = tk.Frame(root, bg="#1a1a2e")
        frame.grid(row=3, column=1, padx=10, pady=10)
        self.prob_bars = []
        self.prob_lbls = []
        for i in range(10):
            tk.Label(frame, text=str(i), bg="#1a1a2e", fg="#888",
                     font=prob_font, width=2).grid(row=i, column=0, pady=1)
            bar = tk.Canvas(frame, width=120, height=14, bg="#111",
                            highlightthickness=0)
            bar.grid(row=i, column=1, padx=4, pady=1)
            lbl = tk.Label(frame, text="0%", bg="#1a1a2e", fg="#555",
                           font=prob_font, width=5)
            lbl.grid(row=i, column=2)
            self.prob_bars.append(bar)
            self.prob_lbls.append(lbl)

        # Кнопки
        btn_frame = tk.Frame(root, bg="#1a1a2e")
        btn_frame.grid(row=4, column=1, pady=10)
        tk.Button(btn_frame, text="Очистить", command=self.clear,
                  bg="#333", fg="white", relief="flat", padx=12, pady=6).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Распознать", command=self.predict,
                  bg="#4fc3f7", fg="black", relief="flat", padx=12, pady=6,
                  font=tkfont.Font(weight="bold")).pack(side="left", padx=5)

        tk.Label(root, text="Рисуй левой кнопкой мыши", bg="#1a1a2e",
                 fg="#555", font=tkfont.Font(size=10)).grid(row=5, column=1, pady=(0,10))

        # События мыши
        self.canvas.bind("<B1-Motion>",    self.draw)
        self.canvas.bind("<ButtonPress-1>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.predict())
        self.last_x = self.last_y = None

    def draw(self, e):
        x, y = e.x, e.y
        r = self.BRUSH_SIZE // 2
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    fill="white", width=self.BRUSH_SIZE,
                                    capstyle=tk.ROUND, joinstyle=tk.ROUND)
        self.last_x, self.last_y = x, y
        # Рисуем в массиве пикселей
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                px, py = x+dx, y+dy
                if 0 <= px < self.CANVAS_SIZE and 0 <= py < self.CANVAS_SIZE:
                    if dx*dx + dy*dy <= r*r:
                        self.pixels[py, px] = 255.0

    def clear(self):
        self.canvas.delete("all")
        self.pixels[:] = 0
        self.lbl_digit.config(text="?", fg="#4fc3f7")
        self.lbl_conf.config(text="")
        self.last_x = self.last_y = None
        for i in range(10):
            self.prob_bars[i].delete("all")
            self.prob_lbls[i].config(text="0%", fg="#555")

    def predict(self):
        self.last_x = self.last_y = None
        x = preprocess(self.pixels)
        if x is None:
            return
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].tolist()
        pred = int(np.argmax(probs))
        conf = max(probs) * 100
        self.lbl_digit.config(text=str(pred),
                               fg="#4fc3f7" if conf > 70 else "#ff9800" if conf > 40 else "#f44336")
        self.lbl_conf.config(text=f"Уверенность: {conf:.1f}%")
        # Обновляем бары
        for i, p in enumerate(probs):
            bar = self.prob_bars[i]
            bar.delete("all")
            w = int(p * 120)
            color = "#4fc3f7" if i == pred else "#444"
            if w > 0:
                bar.create_rectangle(0, 0, w, 14, fill=color, outline="")
            pct = f"{p*100:.1f}%"
            self.prob_lbls[i].config(text=pct, fg="#4fc3f7" if i == pred else "#666")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app  = App(root)
        root.mainloop()
    except Exception as e:
        import traceback
        print("ОШИБКА:", e)
        traceback.print_exc()
        input("\nНажми Enter для выхода...")
