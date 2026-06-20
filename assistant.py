"""
Универсальный ассистент на базе Qwen2-VL-7B (4-bit)
- Описывает картинки
- Пишет код
- Отвечает на вопросы на русском
- Для точной арифметики использует нашу math_solver

Использование:
    python assistant.py
"""
import sys, io
if hasattr(sys.stdout, "buffer"): sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import re
import sys
import torch
from pathlib import Path
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# ─────────────────────────────────────────
# 1. MATH SOLVER — наша обученная сетка
# ─────────────────────────────────────────
import torch.nn as nn

CHARS = "0123456789+-*/=<>^~"
PAD, END, SOS, NEG = '<', '>', '^', '~'
c2i = {c: i for i, c in enumerate(CHARS)}
i2c = {i: c for i, c in enumerate(CHARS)}
VOCAB   = len(CHARS)
MAX_LEN = 16

class Encoder(nn.Module):
    def __init__(self, vocab, embed=64, hidden=256, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=c2i[PAD])
        self.lstm  = nn.LSTM(embed, hidden, layers, batch_first=True, dropout=0.1)
    def forward(self, x):
        return self.lstm(self.embed(x))

class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, hidden)
        self.v    = nn.Linear(hidden, 1, bias=False)
    def forward(self, dec_h, enc_out):
        T     = enc_out.size(1)
        dec_h = dec_h.unsqueeze(1).repeat(1, T, 1)
        energy = torch.tanh(self.attn(torch.cat([dec_h, enc_out], dim=2)))
        return torch.softmax(self.v(energy).squeeze(2), dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab, embed=64, hidden=256, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=c2i[PAD])
        self.attn  = Attention(hidden)
        self.lstm  = nn.LSTM(embed + hidden, hidden, layers, batch_first=True, dropout=0.1)
        self.fc    = nn.Linear(hidden * 2, vocab)
    def forward(self, token, hidden, enc_out):
        emb     = self.embed(token.unsqueeze(1))
        dec_h   = hidden[0][-1]
        a       = self.attn(dec_h, enc_out)
        context = (a.unsqueeze(1) @ enc_out)
        out, hidden = self.lstm(torch.cat([emb, context], dim=2), hidden)
        pred    = self.fc(torch.cat([out.squeeze(1), context.squeeze(1)], dim=1))
        return pred, hidden

class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(VOCAB)
        self.decoder = Decoder(VOCAB)
    def forward(self, src, tgt, teacher_forcing=0.5):
        import random
        enc_out, hidden = self.encoder(src)
        B, T_out = tgt.shape
        preds = torch.zeros(B, T_out, VOCAB, device=src.device)
        token = tgt[:, 0]
        for t in range(1, T_out):
            pred, hidden = self.decoder(token, hidden, enc_out)
            preds[:, t] = pred
            token = tgt[:, t] if random.random() < teacher_forcing else pred.argmax(1)
        return preds

def encode_math(s, length=MAX_LEN):
    ids = [c2i.get(c, 0) for c in s]
    ids += [c2i[PAD]] * (length - len(ids))
    return ids[:length]

def load_math_model():
    model_path = Path(__file__).parent / "math_model.pth"
    if not model_path.exists():
        return None
    m = Seq2Seq().to("cuda")
    m.load_state_dict(torch.load(model_path, weights_only=True))
    m.eval()
    return m

def solve_math(math_model, question: str) -> str:
    """Решает пример через нашу нейронку"""
    src = torch.tensor([encode_math(question)], dtype=torch.long).to("cuda")
    with torch.no_grad():
        enc_out, hidden = math_model.encoder(src)
        token  = torch.tensor([c2i[SOS]], device="cuda")
        result = []
        for _ in range(MAX_LEN):
            pred, hidden = math_model.decoder(token, hidden, enc_out)
            token = pred.argmax(1)
            ch = i2c[token.item()]
            if ch == END: break
            if ch != PAD: result.append(ch)
    raw = ''.join(result)
    if raw.endswith(NEG):
        return '-' + raw[:-1]
    return raw

# Паттерн для поиска арифметических выражений в тексте
MATH_PATTERN = re.compile(r'\b(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*\?')

def extract_and_solve(math_model, text: str) -> tuple[str, bool]:
    """Если в тексте есть пример вида '123 + 456 = ?', решаем нашей сеткой"""
    match = MATH_PATTERN.search(text)
    if match and math_model:
        a, op, b = match.group(1), match.group(2), match.group(3)
        question = f"{a}{op}{b}="
        answer = solve_math(math_model, question)
        return answer, True
    return "", False

# ─────────────────────────────────────────
# 2. ЗАГРУЗКА QWEN2-VL
# ─────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

print("=" * 55)
print("  Универсальный ассистент (Qwen2-VL-7B + MathSolver)")
print("=" * 55)
print("\nЗагрузка модели (первый запуск: ~5 ГБ скачивается)...\n")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Загружаем нашу math_solver
math_model = load_math_model()
if math_model:
    print("Math Solver: загружен ✓")
else:
    print("Math Solver: не найден (запустите math_solver.py сначала)")

print("\nМодель готова! Можно начинать.\n")

# ─────────────────────────────────────────
# 3. ГЕНЕРАЦИЯ ОТВЕТА
# ─────────────────────────────────────────
def chat(messages: list, image_path: str = None) -> str:
    content = []

    if image_path:
        image = Image.open(image_path).convert("RGB")
        content.append({"type": "image", "image": image})

    content.append({"type": "text", "text": messages[-1]["content"]})

    conversation = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    if image_path:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt"
        ).to(model.device)
    else:
        inputs = processor(text=[prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()

# ─────────────────────────────────────────
# 4. ИНТЕРАКТИВНЫЙ ЧАТ
# ─────────────────────────────────────────
print("Команды:")
print("  просто пишите — задайте вопрос")
print("  /фото <путь>   — прикрепить картинку")
print("  /код <задача>  — написать код")
print("  /выход         — выйти")
print("-" * 55 + "\n")

history = []
current_image = None

SYSTEM = "Ты умный ассистент. Отвечай на русском языке. Пиши чётко и по делу."

while True:
    try:
        user_input = input("Вы: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nПока!")
        break

    if not user_input:
        continue

    if user_input.lower() in ('/выход', '/exit', 'выход', 'exit'):
        print("Пока!")
        break

    # Команда /фото
    if user_input.lower().startswith('/фото'):
        parts = user_input.split(maxsplit=1)
        if len(parts) < 2:
            print("Укажите путь: /фото путь/к/картинке\n")
            continue
        path = parts[1].strip().strip('"')
        if not Path(path).exists():
            print(f"Файл не найден: {path}\n")
            continue
        current_image = path
        print(f"Картинка загружена: {path}")
        user_input = "Подробно опиши что изображено на этой картинке."

    # Команда /код
    elif user_input.lower().startswith('/код'):
        parts = user_input.split(maxsplit=1)
        task = parts[1] if len(parts) > 1 else "напиши Hello World"
        user_input = f"Напиши код на Python: {task}. Добавь комментарии на русском языке."

    # Проверяем: может это арифметический пример? (вида "123 + 456 = ?")
    math_answer, is_math = extract_and_solve(math_model, user_input)
    if is_math:
        match = MATH_PATTERN.search(user_input)
        print(f"\nАссистент (MathSolver): {match.group(1)} {match.group(2)} {match.group(3)} = {math_answer}\n")
        current_image = None
        continue

    # Обычный вопрос — отправляем в Qwen2-VL
    history.append({"role": "user", "content": user_input})

    print("\nАссистент: ", end="", flush=True)
    response = chat(history, image_path=current_image)
    print(response)
    print()

    history.append({"role": "assistant", "content": response})
    current_image = None  # картинка используется один раз
