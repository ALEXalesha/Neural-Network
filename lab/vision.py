"""
Vision — описание изображений через LLaVA-1.5-7B (4-bit квантизация)
Использование:
    python vision.py                    # интерактивный режим
    python vision.py путь/к/картинке    # описать одну картинку
"""
import sys, io
if hasattr(sys.stdout, "buffer"): sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sys
import torch
from pathlib import Path
from PIL import Image
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

print("Загрузка модели LLaVA-1.5-7B (4-bit)...")
print("Первый запуск: скачивается ~4 ГБ, подождите...\n")

# 4-bit квантизация — модель займёт ~4 ГБ вместо 14 ГБ
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("Модель загружена!\n")


def describe(image_path: str, question: str = "Подробно опиши что изображено на этой картинке. Отвечай на русском языке.") -> str:
    image = Image.open(image_path).convert("RGB")

    # Формат чата LLaVA
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # Убираем промпт из ответа
    generated = output[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def main():
    if len(sys.argv) > 1:
        # Режим: python vision.py картинка.jpg
        path = sys.argv[1]
        question = sys.argv[2] if len(sys.argv) > 2 else "Подробно опиши что изображено на этой картинке. Отвечай на русском языке."
        if not Path(path).exists():
            print(f"Файл не найден: {path}")
            return
        print(f"Анализирую: {path}\n")
        result = describe(path, question)
        print("Описание:")
        print(result)
    else:
        # Интерактивный режим
        print("=" * 50)
        print("VISION — Описание изображений")
        print("=" * 50)
        print("Введите путь к картинке (или 'выход' для выхода)")
        print("Можно также задать вопрос по картинке\n")

        while True:
            path = input("Путь к картинке: ").strip().strip('"')
            if path.lower() in ('выход', 'exit', 'quit', 'q'):
                break
            if not Path(path).exists():
                print(f"Файл не найден: {path}\n")
                continue

            question = input("Вопрос (Enter = описать всё): ").strip()
            if not question:
                question = "Подробно опиши что изображено на этой картинке. Отвечай на русском языке."

            print("\nАнализирую...\n")
            result = describe(path, question)
            print("Ответ:")
            print(result)
            print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
