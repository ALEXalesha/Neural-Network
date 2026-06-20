"""
Предзагрузка всех моделей Smart Assistant
Запусти и жди — всё скачается в кеш Hugging Face
"""
import sys, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import threading
from huggingface_hub import snapshot_download

MODELS = [
    ("deepseek-ai/deepseek-coder-6.7b-instruct", "Coder"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  "Reasoner"),
    ("Qwen/Qwen2.5-7B-Instruct",                 "Writer"),
    ("Qwen/Qwen2-VL-7B-Instruct",                "Vision"),
]

def download(model_id, name):
    print(f"[{name}] Начинаю скачивание {model_id}...")
    try:
        snapshot_download(repo_id=model_id, ignore_patterns=["*.pt", "original/*"])
        print(f"[{name}] ГОТОВО!")
    except Exception as e:
        print(f"[{name}] Ошибка: {e}")

threads = [threading.Thread(target=download, args=(mid, name)) for mid, name in MODELS]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("\nВсе модели скачаны!")
