"""
Coder Team — команда из 4 нейросетей для написания кода
Backend: LM Studio (http://localhost:1234)

Конвейер:
  [Writer]   DeepSeek-Coder   — пишет первичный код
  [Reviewer] Qwen2.5          — находит баги и проблемы
  [Fixer]    DeepSeek-Coder   — исправляет по замечаниям
  [Refiner]  Qwen2.5          — полирует, документирует, улучшает

Настройки моделей — в lm_config.json
"""
import sys, re, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

from paths import DATA_DIR, lm_config_path, lm_studio_url

# ──────────────────────────────────────────────────────
# КОНФИГ
# ──────────────────────────────────────────────────────
def load_config():
    path = lm_config_path()
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {}

CFG    = load_config()
LM_URL = lm_studio_url(CFG)
MODELS = CFG.get('models', {})

# model_id для каждого агента
AGENT_MODELS = {
    'writer':   MODELS.get('coder',    {}).get('model_id', 'deepseek-coder-6.7b-instruct'),
    'reviewer': MODELS.get('writer',   {}).get('model_id', 'qwen2.5-7b-instruct'),
    'fixer':    MODELS.get('coder',    {}).get('model_id', 'deepseek-coder-6.7b-instruct'),
    'refiner':  MODELS.get('writer',   {}).get('model_id', 'qwen2.5-7b-instruct'),
}

# ──────────────────────────────────────────────────────
# ЗАПРОС К LM STUDIO
# ──────────────────────────────────────────────────────
def ask(agent_name: str, system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> str:
    model_id = AGENT_MODELS[agent_name]
    try:
        resp = requests.post(
            f'{LM_URL}/chat/completions',
            json={
                'model':       model_id,
                'messages':    [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': user_prompt},
                ],
                'temperature': 0.3,
                'max_tokens':  max_tokens,
                'stream':      False,
            },
            timeout=300,
        )
        resp.raise_for_status()
        text = resp.json()['choices'][0]['message'].get('content') or ''
        # Убираем <think>...</think> блоки (DeepSeek-R1)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return text
    except requests.exceptions.ConnectionError:
        print("\n[ОШИБКА] LM Studio не запущена!")
        print("  Открой LM Studio → Local Server → Start Server\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ОШИБКА] {e}\n")
        return f"[Ошибка: {e}]"

# ──────────────────────────────────────────────────────
# АГЕНТЫ
# ──────────────────────────────────────────────────────
def agent_writer(task: str) -> str:
    print("\n" + "="*60)
    print(f"АГЕНТ 1 — WRITER  [{AGENT_MODELS['writer']}]")
    print("="*60, flush=True)
    result = ask('writer',
        system_prompt=(
            "You are an expert programmer. Write clean, working Python code. "
            "Include all imports. Add brief comments. Output only the code."
        ),
        user_prompt=f"Task: {task}\n\nWrite complete, working code:",
    )
    print(result)
    return result

def agent_reviewer(task: str, code: str) -> str:
    print("\n" + "="*60)
    print(f"АГЕНТ 2 — REVIEWER  [{AGENT_MODELS['reviewer']}]")
    print("="*60, flush=True)
    result = ask('reviewer',
        system_prompt=(
            "You are a senior code reviewer. Find bugs, edge cases, security issues. "
            "Be specific and actionable. Number each issue."
        ),
        user_prompt=(
            f"Task: {task}\n\nCode:\n```python\n{code}\n```\n\n"
            "List all issues (bugs, edge cases, improvements needed):"
        ),
        max_tokens=1024,
    )
    print(result)
    return result

def agent_fixer(task: str, code: str, review: str) -> str:
    print("\n" + "="*60)
    print(f"АГЕНТ 3 — FIXER  [{AGENT_MODELS['fixer']}]")
    print("="*60, flush=True)
    result = ask('fixer',
        system_prompt=(
            "You are an expert programmer. Fix all issues from the review. "
            "Output the complete corrected code only."
        ),
        user_prompt=(
            f"Task: {task}\n\nOriginal code:\n```python\n{code}\n```\n\n"
            f"Issues to fix:\n{review}\n\nWrite complete fixed code:"
        ),
    )
    print(result)
    return result

def agent_refiner(task: str, code: str) -> str:
    print("\n" + "="*60)
    print(f"АГЕНТ 4 — REFINER  [{AGENT_MODELS['refiner']}]")
    print("="*60, flush=True)
    result = ask('refiner',
        system_prompt=(
            "You are a software architect. Improve the code: add docstrings, "
            "type hints, PEP8 formatting. Output final polished code only."
        ),
        user_prompt=f"Code:\n```python\n{code}\n```\n\nPolished final code:",
    )
    print(result)
    return result

# ──────────────────────────────────────────────────────
# КОНВЕЙЕР
# ──────────────────────────────────────────────────────
def run_pipeline(task: str, save: bool = True) -> str:
    print(f"\n{'='*60}")
    print(f"ЗАДАЧА: {task}")
    print(f"LM Studio: {LM_URL}")
    print(f"{'='*60}\n")

    code_v1    = agent_writer(task)
    review     = agent_reviewer(task, code_v1)
    code_v2    = agent_fixer(task, code_v1, review)
    code_final = agent_refiner(task, code_v2)

    if save:
        out_file = DATA_DIR / "coder_output.py"
        match = re.search(r'```[a-zA-Z0-9_+-]*\n(.*?)```', code_final, re.DOTALL)
        save_code = match.group(1) if match else code_final
        out_file.write_text(f"# Task: {task}\n# Generated by Coder Team\n\n{save_code}", encoding='utf-8')
        print(f"\n[Сохранено: {out_file}]")

    print(f"\n{'='*60}")
    print("КОНВЕЙЕР ЗАВЕРШЁН")
    print(f"{'='*60}")
    return code_final

# ──────────────────────────────────────────────────────
# ИНТЕРАКТИВНЫЙ РЕЖИМ
# ──────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║              CODER TEAM — 4 агента                  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Writer   → пишет код        (DeepSeek-Coder)       ║")
    print("║  Reviewer → находит баги     (Qwen2.5)              ║")
    print("║  Fixer    → исправляет       (DeepSeek-Coder)       ║")
    print("║  Refiner  → полирует         (Qwen2.5)              ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Backend: {LM_URL:<42}║")
    print("║  /exit — выход   /save — вкл/выкл сохранение        ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    save_output = True

    while True:
        try:
            task = input("Задача: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not task:
            continue
        if task == "/exit":
            print("Выход.")
            break
        if task == "/save":
            save_output = not save_output
            print(f"Сохранение: {'вкл' if save_output else 'выкл'}\n")
            continue

        run_pipeline(task, save=save_output)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_pipeline(' '.join(sys.argv[1:]), save=True)
    else:
        main()
