"""
Smart Assistant — терминальная версия (LM Studio backend)
Запуск: python smart_assistant.py

Модели работают через LM Studio (http://localhost:1234).
Настройки — в lm_config.json
"""
import sys, io, re, json
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

BASE = Path(__file__).parent

try:
    import requests
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'requests', '-q'])
    import requests

# ══════════════════════════════════════════════════════
# КОНФИГ
# ══════════════════════════════════════════════════════

def load_config():
    path = BASE / 'lm_config.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {}

CFG      = load_config()
LM_URL   = CFG.get('lm_studio_url', 'http://localhost:1234/v1')
MODELS   = CFG.get('models', {})
KEYWORDS = CFG.get('routing_keywords', {})

# ══════════════════════════════════════════════════════
# РОУТЕР
# ══════════════════════════════════════════════════════

def route(query: str, has_image: bool) -> str:
    if has_image:
        return 'vision'
    q = query.lower()
    scores = {name: sum(1 for kw in kws if kw in q) for name, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get) if scores else 'reasoner'
    return best if scores.get(best, 0) > 0 else 'reasoner'

# ══════════════════════════════════════════════════════
# ЗАПРОС К LM STUDIO (стриминг в терминал)
# ══════════════════════════════════════════════════════

def ask(model_name: str, messages: list, max_tokens: int = 2048) -> str:
    model_cfg = MODELS.get(model_name, MODELS.get('reasoner', {}))
    model_id  = model_cfg.get('model_id', '')

    # Добавляем системный промпт если его нет
    if messages and messages[0]['role'] != 'system' and model_cfg.get('system_prompt'):
        messages = [{'role': 'system', 'content': model_cfg['system_prompt']}] + messages

    try:
        resp = requests.post(
            f'{LM_URL}/chat/completions',
            json={
                'model':       model_id,
                'messages':    messages,
                'temperature': model_cfg.get('temperature', 0.7),
                'max_tokens':  max_tokens,
                'stream':      True,
            },
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()

        full_text = ''
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode('utf-8') if isinstance(line, bytes) else line
            if line.startswith('data: '):
                chunk = line[6:]
                if chunk.strip() == '[DONE]':
                    break
                try:
                    token = json.loads(chunk)['choices'][0].get('delta', {}).get('content', '')
                    if token:
                        # Фильтруем <think> блоки для reasoner — показываем красиво
                        print(token, end='', flush=True)
                        full_text += token
                except Exception:
                    pass

        print()  # новая строка после ответа
        # Форматируем <think> в итоговом тексте
        full_text = re.sub(
            r'<think>(.*?)</think>',
            lambda m: f"\n{'─'*40}\n💭 Размышление:\n{m.group(1).strip()}\n{'─'*40}\n",
            full_text, flags=re.DOTALL
        )
        return full_text.strip()

    except requests.exceptions.ConnectionError:
        print("\n[ОШИБКА] LM Studio не запущена!")
        print("  Открой LM Studio → Local Server → Start Server\n")
        return ''
    except Exception as e:
        print(f"\n[ОШИБКА] {e}\n")
        return ''

# ══════════════════════════════════════════════════════
# ГЛАВНЫЙ ЧАТ
# ══════════════════════════════════════════════════════

def print_header():
    print("=" * 60)
    print("  SMART ASSISTANT  [LM Studio]")
    print(f"  {LM_URL}")
    print("=" * 60)
    print()
    for name, cfg in MODELS.items():
        model_id = cfg.get('model_id', '').split('/')[-1].replace('.gguf', '')
        print(f"  [{name:8}] {model_id}")
    print()
    print("  Команды:")
    print("    /coder      — режим: код и сайты")
    print("    /reasoner   — режим: задачи и рассуждения")
    print("    /writer     — режим: тексты и статьи")
    print("    /auto       — автовыбор модели (по умолчанию)")
    print("    /модель     — показать текущую модель")
    print("    /team <задача> — команда Coder Team")
    print("    /exit       — выйти")
    print("─" * 60)
    print()

def main():
    print_header()

    forced_model = None
    current_model = None

    while True:
        try:
            line = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break

        if not line:
            continue

        # ── Команды ──────────────────────────────────
        if line.lower() in ('/exit', 'quit', 'q', 'exit'):
            print("Пока!")
            break

        if line.lower() in ('/coder', '/code', '/код'):
            forced_model = 'coder'
            print(f"[Режим: coder — {MODELS.get('coder',{}).get('model_id','').split('/')[-1]}]\n")
            continue

        if line.lower() in ('/reasoner', '/reason', '/think'):
            forced_model = 'reasoner'
            print(f"[Режим: reasoner — {MODELS.get('reasoner',{}).get('model_id','').split('/')[-1]}]\n")
            continue

        if line.lower() in ('/writer', '/write'):
            forced_model = 'writer'
            print(f"[Режим: writer — {MODELS.get('writer',{}).get('model_id','').split('/')[-1]}]\n")
            continue

        if line.lower() == '/auto':
            forced_model = None
            print("[Режим: авто — роутер выберет модель]\n")
            continue

        if line.lower() == '/модель':
            print(f"Текущая модель: {current_model or 'не выбрана (авто)'}\n")
            continue

        if line.lower().startswith('/team'):
            task = line[5:].strip()
            if not task:
                task = input("Задача для Coder Team: ").strip()
            if task:
                import subprocess as sp
                sp.run([sys.executable, str(BASE / 'coder_team.py'), task])
            continue

        if line.lower() in ('/lab', '/menu'):
            import subprocess as sp
            sp.run([sys.executable, str(BASE / 'menu.py')])
            continue

        # ── Роутинг ──────────────────────────────────
        chosen = forced_model if forced_model else route(line, False)

        if chosen != current_model:
            model_short = MODELS.get(chosen, {}).get('model_id', chosen).split('/')[-1].replace('.gguf', '')
            print(f"[→ {chosen}: {model_short}]")
            current_model = chosen

        # ── Генерация ─────────────────────────────────
        print(f"\nАссистент ({chosen}):\n")
        ask(chosen, [{'role': 'user', 'content': line}])
        print()

if __name__ == "__main__":
    main()
