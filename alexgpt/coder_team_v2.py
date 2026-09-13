"""
Coder Team v2 — автономная команда из 4 нейросетей с итеративным циклом
Backend: LM Studio (http://localhost:1234)

Цикл:
  [Архитектор] — один раз, создаёт план
  [Кодер]      — пишет / переписывает код
  [Ревьюер]    — проверяет логику, баги, стиль
  [Тестировщик]— генерирует тесты, проверяет покрытие
        ↓
  Оба одобрили? → готово
  Есть замечания? → Кодер получает объединённый фидбек и итерирует
        ↑___________________________________|
                  (до MAX_ITERATIONS раз)
"""
import sys, re, json, time, os, textwrap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import requests

from paths import DATA_DIR, FROZEN, lm_config_path, lm_studio_url, pyfile_cmd

_iter_arg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
MAX_ITERATIONS = int(_iter_arg or os.environ.get("CODER_V2_MAX_ITER", 4))

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

AGENT_MODELS = {
    'architect': MODELS.get('reasoner', MODELS.get('writer',   {})).get('model_id', 'qwen2.5-7b-instruct'),
    'coder':     MODELS.get('coder',    {}).get('model_id', 'deepseek-coder-6.7b-instruct'),
    'reviewer':  MODELS.get('writer',   {}).get('model_id', 'qwen2.5-7b-instruct'),
    'tester':    MODELS.get('coder',    {}).get('model_id', 'deepseek-coder-6.7b-instruct'),
}

# ──────────────────────────────────────────────────────
# ЗАПРОС К LM STUDIO
# ──────────────────────────────────────────────────────
LM_ERROR_PREFIX = "[LM_ERROR]"

def ask(agent: str, system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> str:
    model_id = AGENT_MODELS[agent]
    last_err = None
    for attempt in range(3):  # до 3 попыток
        try:
            resp = requests.post(
                f'{LM_URL}/chat/completions',
                json={
                    'model':       model_id,
                    'messages':    [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user',   'content': user_prompt},
                    ],
                    'temperature': 0.25,
                    'max_tokens':  max_tokens,
                    'stream':      False,
                },
                timeout=300,
            )
            resp.raise_for_status()
            # content бывает None, если рассуждения R1 съели весь лимит токенов
            text = resp.json()['choices'][0]['message'].get('content') or ''
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            return text
        except requests.exceptions.ConnectionError:
            print("\n[ОШИБКА] LM Studio не запущена! Открой LM Studio → Local Server → Start Server\n")
            sys.exit(1)
        except requests.exceptions.HTTPError as e:
            last_err = e
            status = e.response.status_code if e.response is not None else 0
            if status >= 500:
                # Серверная ошибка — ретрай с паузой
                if attempt < 2:
                    wait = 8 * (attempt + 1)
                    print(f"\n[ПРЕДУПРЕЖДЕНИЕ] {agent}: {e} — повтор через {wait}с (попытка {attempt+1}/3)\n", flush=True)
                    time.sleep(wait)
                    continue
            # 4xx или исчерпаны ретраи
            print(f"\n[ОШИБКА] {agent}: {e}\n")
            return f"{LM_ERROR_PREFIX} {e}"
        except Exception as e:
            last_err = e
            print(f"\n[ОШИБКА] {agent}: {e}\n")
            return f"{LM_ERROR_PREFIX} {e}"
    print(f"\n[ОШИБКА] {agent}: все попытки исчерпаны — {last_err}\n")
    return f"{LM_ERROR_PREFIX} {last_err}"

def extract_code(text: str) -> str:
    """Извлекает код из markdown-блока (любой язык)."""
    match = re.search(r'```[a-zA-Z0-9_+-]*\n(.*?)```', text, re.DOTALL)
    return match.group(1).strip() if match else ''

_NEGATED = re.compile(r"\b(not|isn't|cannot|can't|won't)\b[^.\n]{0,20}\b(approv|lgtm)|\b(un|dis)approv|\bне\s+одобр")
_STRONG_APPROVE = re.compile(r"\ball tests pass|\bapproved\b|\blgtm\b|\bno issues found|\bодобрен|"
                             r"все тесты проходят|замечаний нет")

def is_approved(text: str) -> bool:
    """Проверяет, одобрил ли агент код."""
    lower = text.strip().lower()

    # "NOT APPROVED" и "не одобрено" содержат слово-одобрение, поэтому отказ проверяем первым
    if _NEGATED.search(lower):
        return False
    if _STRONG_APPROVE.search(lower):
        return True

    # Явный отказ в начале ответа
    EXPLICIT_REJECT = ('issue', 'bug', 'error', 'problem', 'fail',
                       'ошибка', 'баг', 'проблема', '1.', '1)')
    for kw in EXPLICIT_REJECT:
        if lower.startswith(kw):
            return False

    # Взвешенный подсчёт
    positive_phrases = [
        'no bugs', 'looks good', 'no critical', 'ready for production',
        'всё корректно', 'код готов', 'нет замечаний', 'no problems found',
        'code is correct', 'works correctly', 'well written',
        'if all these tests pass', 'these tests should pass',
        'working correctly', 'tests will pass',
        'means all tests passed', 'all tests passed',
    ]
    negative_phrases = [
        'there is a bug', 'there is an error', 'missing error handling',
        'does not handle', "doesn't handle", 'will fail', 'incorrect',
        'баг в', 'ошибка в', 'не обрабатывает', 'не хватает', 'упадёт',
        'should fix', 'needs to be fixed', 'needs fixing',
        'assertion error', 'assertionerror',
    ]
    pos = sum(1 for p in positive_phrases if p in lower)
    neg = sum(1 for p in negative_phrases if p in lower)
    return pos > 0 and neg == 0

# ──────────────────────────────────────────────────────
# АГЕНТЫ
# ──────────────────────────────────────────────────────
def agent_architect(task: str) -> str:
    print(f"\n{'━'*60}")
    print(f"  АРХИТЕКТОР  [{AGENT_MODELS['architect']}]")
    print(f"{'━'*60}", flush=True)
    result = ask('architect',
        system_prompt=(
            "You are a software architect. Analyze the task and create a clear implementation plan. "
            "List: required functions, data structures, edge cases to handle, and expected behavior. "
            "Be concise and precise."
        ),
        user_prompt=f"Task: {task}\n\nCreate an implementation plan:",
        # Архитектор — DeepSeek R1: сначала рассуждает (reasoning_content), потом пишет план
        max_tokens=4096,
    )
    print(result, flush=True)
    return result

def agent_coder(task: str, plan: str, feedback: str = "") -> str:
    print(f"\n{'━'*60}")
    iteration_note = f"  КОДЕР (итерация с фидбеком)  [{AGENT_MODELS['coder']}]" if feedback else f"  КОДЕР  [{AGENT_MODELS['coder']}]"
    print(iteration_note)
    print(f"{'━'*60}", flush=True)

    if feedback:
        user_prompt = (
            f"Task: {task}\n\nArchitect plan:\n{plan}\n\n"
            f"Feedback from team (fix ALL these issues):\n{feedback}\n\n"
            "Write the complete corrected Python code:"
        )
    else:
        user_prompt = (
            f"Task: {task}\n\nArchitect plan:\n{plan}\n\n"
            "Write complete, working Python code with all imports and error handling:"
        )

    result = ask('coder',
        system_prompt=(
            "You are an expert Python developer. Write clean, efficient, production-ready code. "
            "Include all imports. Handle edge cases. Output ONLY the implementation code in a ```python block. "
            "NEVER include test code, unittest, pytest, or mock objects in your implementation. "
            "Tests are separate — your job is to write only the working program code."
        ),
        user_prompt=user_prompt,
    )
    print(result, flush=True)
    return result

def agent_reviewer(task: str, plan: str, code: str) -> tuple[str, bool]:
    print(f"\n{'━'*60}")
    print(f"  РЕВЬЮЕР  [{AGENT_MODELS['reviewer']}]")
    print(f"{'━'*60}", flush=True)
    result = ask('reviewer',
        system_prompt=(
            "You are a senior code reviewer. Carefully check the code for:\n"
            "1. Logic bugs and edge cases\n2. Missing error handling\n"
            "3. Performance issues\n4. Security problems\n5. Compliance with the plan\n\n"
            "If the code is correct and complete, start your response with 'APPROVED'.\n"
            "Otherwise list each issue clearly numbered."
        ),
        user_prompt=(
            f"Task: {task}\n\nPlan:\n{plan}\n\n"
            f"Code to review:\n```python\n{code}\n```\n\n"
            "Review:"
        ),
        max_tokens=1024,
    )
    print(result, flush=True)
    if result.startswith(LM_ERROR_PREFIX):
        print("\n  → Ревьюер: ⚠ ПРОПУЩЕН (ошибка LM Studio)", flush=True)
        return result, None   # None = inconclusive
    approved = is_approved(result)
    status = "✓ ОДОБРЕНО" if approved else "✗ ЕСТЬ ЗАМЕЧАНИЯ"
    print(f"\n  → Ревьюер: {status}", flush=True)
    return result, approved

def _extract_defs_only(code: str) -> str:
    """Убирает опасный топ-левел код (while/input/if __name__), оставляет всё остальное."""
    DANGEROUS = ('while ', 'while\t', 'if __name__', 'input(', 'input (')
    lines = textwrap.dedent(code).splitlines()
    result_lines = []
    skip_block = False

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if not stripped:
            # Пустая строка — добавляем если не внутри пропускаемого блока
            if not skip_block:
                result_lines.append(line)
            continue

        if indent == 0:
            if stripped.startswith(DANGEROUS):
                skip_block = True
                continue
            elif stripped.startswith(('input(', 'input (')):
                continue  # голый input() на верхнем уровне
            else:
                skip_block = False
                result_lines.append(line)
        else:
            if skip_block:
                continue  # тело опасного блока
            result_lines.append(line)

    return '\n'.join(result_lines).strip('\n')


_SAFE_MODULES = {
    'unittest', 'pytest', 'sys', 'os', 're', 'json', 'mock', 'io',
    'time', 'datetime', 'math', 'random', 'pathlib', 'typing',
    'collections', 'functools', 'itertools', 'string', 'traceback',
    'inspect', 'abc', 'copy', 'tempfile', 'subprocess', 'hashlib',
    'base64', 'uuid', 'decimal', 'asyncio', 'threading', 'socket',
    'http', 'urllib', 'requests', 'flask', 'numpy', 'pandas',
    'PIL', 'cv2', 'builtins', 'contextlib', 'dataclasses', 'enum',
}

def _normalize_imports(test_code: str) -> str:
    """Заменяет импорты основного модуля на 'solution' в тест-коде.
    Не трогает стандартные библиотеки (unittest, pytest и т.д.)."""

    def _replace_from(m):
        mod = m.group(1)
        if mod in _SAFE_MODULES or mod.startswith('unittest') or mod.startswith('pytest'):
            return m.group(0)
        return 'from solution import'

    def _replace_import(m):
        mod = m.group(1)
        alias = m.group(2) or ''
        if mod in _SAFE_MODULES or mod.startswith('unittest') or mod.startswith('pytest'):
            return m.group(0)
        return f'import solution{alias}'

    # from weather_api import X  →  from solution import X
    test_code = re.sub(
        r'^from[ \t]+(\w+)[ \t]+import\b',
        _replace_from,
        test_code,
        flags=re.MULTILINE,
    )
    # import weather_api         →  import solution
    # import weather_api as api  →  import solution as api
    # Используем ^ чтобы не матчить "import X" внутри "from solution import X"
    test_code = re.sub(
        r'^import[ \t]+(\w+)([ \t]+as[ \t]+\w+)?',
        _replace_import,
        test_code,
        flags=re.MULTILINE,
    )
    return test_code


def agent_tester(task: str, code: str) -> tuple[str, bool]:
    import tempfile, subprocess as sp, os

    print(f"\n{'━'*60}")
    print(f"  ТЕСТИРОВЩИК  [{AGENT_MODELS['tester']}]")
    print(f"{'━'*60}", flush=True)

    # Шаг 1: просим модель написать тест-код
    result = ask('tester',
        system_prompt=(
            "You are a QA engineer. Write Python test functions using assert statements.\n"
            "The code under test is importable as module 'solution'.\n"
            "Import it like: from solution import <function_name>\n"
            "Test the functions directly (do NOT call main() or use input()).\n"
            "Cover: normal cases, edge cases, error cases.\n"
            "Output ONLY the test code in a ```python block.\n"
            "At the end of the block, call each test function and print('ALL TESTS PASS')."
        ),
        user_prompt=(
            f"Task: {task}\n\n"
            f"Code to test (importable as 'solution'):\n```python\n{code}\n```\n\n"
            "Write test functions (use: from solution import ...):"
        ),
        max_tokens=1024,
    )
    print(result, flush=True)

    if result.startswith(LM_ERROR_PREFIX):
        print("\n  → Тестировщик: ⚠ ПРОПУЩЕН (ошибка LM Studio)", flush=True)
        return result, None

    # Шаг 2: извлекаем тест-код и исправляем импорты
    test_code = extract_code(result)
    if not test_code:
        print("\n  → Тестировщик: ⚠ ПРОПУЩЕН (не найден код в ответе)", flush=True)
        return result, None
    test_code = _normalize_imports(test_code)

    # Оставляем ТОЛЬКО def/class из основного кода — убираем топ-левел код
    # который может вызвать input() или while True: и заблокировать процесс
    code_for_test = _extract_defs_only(code)

    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp()
        solution_path = os.path.join(tmp_dir, 'solution.py')
        test_path     = os.path.join(tmp_dir, 'test_runner.py')

        with open(solution_path, 'w', encoding='utf-8') as f:
            f.write(code_for_test)
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_code)

        # Auto-install pytest если нужен (в собранной версии pip нет)
        if not FROZEN and ('import pytest' in test_code or 'from pytest' in test_code):
            sp.run([sys.executable, '-m', 'pip', 'install', 'pytest', '-q'],
                   capture_output=True, timeout=60)

        # stdin=DEVNULL: input() в сгенерированном коде падает сразу, а не висит до таймаута
        run = sp.run(
            pyfile_cmd(test_path),
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15,
            cwd=tmp_dir, stdin=sp.DEVNULL, creationflags=getattr(sp, 'CREATE_NO_WINDOW', 0),
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
        )

        if run.returncode == 0:
            out = (run.stdout or "").strip()
            print("\n  ✓ Тесты выполнены успешно", flush=True)
            if out:
                print(f"  {out}", flush=True)
            status, approved = "✓ ТЕСТЫ ПРОШЛИ", True
            final = f"ALL TESTS PASS\n{out}"
        else:
            err = (run.stderr or run.stdout or "неизвестная ошибка").strip()
            print(f"\n  ✗ Тесты провалились:\n{err}", flush=True)
            status, approved = "✗ ТЕСТЫ ПРОВАЛИЛИСЬ", False
            final = f"TESTS FAILED:\n{err}\n\nTest code:\n```python\n{test_code}\n```"

    except sp.TimeoutExpired:
        print("\n  ✗ Таймаут (15с) — возможно бесконечный цикл или input()", flush=True)
        status, approved = "✗ ТАЙМАУТ", False
        final = "TESTS FAILED: timeout 15s (infinite loop or blocking input?)"
    except Exception as e:
        print(f"\n  ✗ Ошибка запуска тестов: {e}", flush=True)
        status, approved = "✗ ОШИБКА ЗАПУСКА", False
        final = f"TESTS FAILED: {e}"
    finally:
        if tmp_dir:
            import shutil
            try: shutil.rmtree(tmp_dir, ignore_errors=True)
            except: pass

    print(f"\n  → Тестировщик: {status}", flush=True)
    return final, approved

# ──────────────────────────────────────────────────────
# УПРАВЛЕНИЕ ИТЕРАЦИЯМИ (контрол-файл)
# ──────────────────────────────────────────────────────
CONTROL_FILE = DATA_DIR / "coder_v2_control.json"

def _check_control() -> dict:
    """Читает и удаляет контрол-файл. Возвращает {} если файла нет."""
    if CONTROL_FILE.exists():
        try:
            data = json.loads(CONTROL_FILE.read_text(encoding='utf-8'))
            CONTROL_FILE.unlink(missing_ok=True)
            return data
        except Exception:
            pass
    return {}

def _apply_control(ctrl: dict, feedback_combined: str) -> tuple[bool, str]:
    """Применяет команду управления. Возвращает (stop, обновлённый_фидбек)."""
    action = ctrl.get('action', '')
    if action == 'stop':
        print("\n  ⏹ ОСТАНОВЛЕНО ПОЛЬЗОВАТЕЛЕМ — сохраняю текущий код\n", flush=True)
        return True, feedback_combined
    if action == 'comment':
        comment = ctrl.get('text', '').strip()
        if comment:
            print(f"\n  💬 Комментарий пользователя: {comment}\n", flush=True)
            extra = f"=== КОММЕНТАРИЙ ПОЛЬЗОВАТЕЛЯ ===\n{comment}"
            feedback_combined = (feedback_combined + "\n\n" + extra).strip()
    return False, feedback_combined


# ──────────────────────────────────────────────────────
# ГЛАВНЫЙ ЦИКЛ
# ──────────────────────────────────────────────────────
def run_pipeline(task: str, save: bool = True) -> str:
    print(f"\n{'═'*60}")
    print("  CODER TEAM v2 — АВТОНОМНЫЙ ЦИКЛ")
    print(f"  Задача: {task}")
    print(f"  LM Studio: {LM_URL}")
    print(f"  Макс. итераций: {MAX_ITERATIONS}")
    print(f"{'═'*60}")

    # Шаг 1: Архитектор (один раз)
    plan = agent_architect(task)

    code_raw = ""
    feedback_combined = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'═'*60}")
        print(f"  ИТЕРАЦИЯ {iteration} / {MAX_ITERATIONS}")
        print(f"{'═'*60}")

        # Шаг 2: Кодер
        code_raw = agent_coder(task, plan, feedback_combined)
        code = extract_code(code_raw)

        # Шаг 3: Ревьюер и Тестировщик последовательно
        # (LM Studio плохо справляется с параллельными запросами к разным моделям)
        print(f"\n{'━'*60}")
        print("  ПРОВЕРКА (ревьюер + тестировщик)")
        print(f"{'━'*60}", flush=True)

        review_result, review_ok = agent_reviewer(task, plan, code)
        test_result,   test_ok   = agent_tester(task, code)

        # None = агент не смог ответить (ошибка LM Studio) — не считаем как провал
        # Одобрено если: оба одобрили, ИЛИ один одобрил а второй не смог ответить
        def vote_status(ok):
            if ok is True:   return "✓ ОДОБРЕНО"
            if ok is False:  return "✗ ЗАМЕЧАНИЯ"
            return "⚠ ПРОПУЩЕН"

        votes = [v for v in [review_ok, test_ok] if v is not None]
        both_approved = len(votes) > 0 and all(votes)

        print(f"\n{'━'*60}")
        print(f"  ИТОГ ИТЕРАЦИИ {iteration}:")
        print(f"  Ревьюер:      {vote_status(review_ok)}")
        print(f"  Тестировщик:  {vote_status(test_ok)}")
        if any(v is None for v in [review_ok, test_ok]):
            print("  (Агент пропущен — ошибка LM Studio, его вето не учитывается)")
        print(f"{'━'*60}", flush=True)

        if both_approved:
            who = "ОБА АГЕНТА" if all(v is not None for v in [review_ok, test_ok]) else "АКТИВНЫЕ АГЕНТЫ"
            print(f"\n  ✓ {who} ОДОБРИЛИ — КОД ГОТОВ (итерация {iteration})")
            break

        # Проверяем контрол-файл (остановка / комментарий от пользователя)
        ctrl = _check_control()
        if ctrl:
            stop, feedback_combined = _apply_control(ctrl, feedback_combined)
            if stop:
                break

        if iteration < MAX_ITERATIONS:
            # Объединяем фидбек только от агентов с реальными замечаниями
            parts = []
            if review_ok is False:
                parts.append(f"=== РЕВЬЮЕР ===\n{review_result}")
            if test_ok is False:
                parts.append(f"=== ТЕСТИРОВЩИК ===\n{test_result}")
                # Если ошибка — неизвестный модуль, добавляем явный запрет
                if "ModuleNotFoundError" in test_result or "No module named" in test_result:
                    import re as _re
                    bad_mods = _re.findall(r"No module named '([^']+)'", test_result)
                    bad_str = ", ".join(f"'{m}'" for m in bad_mods) if bad_mods else "unknown"
                    parts.append(
                        f"=== ВАЖНО ===\n"
                        f"Модуль(и) {bad_str} не установлен(ы) и недоступен(ы).\n"
                        f"Используй ТОЛЬКО стандартные библиотеки Python и `requests`.\n"
                        f"НЕ используй: openweathermap, geocodex, tkinter для GUI, или любые другие сторонние библиотеки кроме requests.\n"
                        f"Для работы с API погоды используй только `import requests`."
                    )
            auto_feedback = "\n\n".join(parts)
            # Комментарий пользователя добавляется к авто-фидбеку
            if feedback_combined and not auto_feedback:
                pass  # только пользовательский комментарий
            elif auto_feedback:
                feedback_combined = auto_feedback + (
                    ("\n\n" + feedback_combined) if feedback_combined else ""
                )
            if auto_feedback or feedback_combined:
                print(f"\n  → Передаю замечания кодеру, итерация {iteration + 1}...", flush=True)
            else:
                print(f"\n  → Нет замечаний (только ошибки LM Studio), повторяю итерацию {iteration + 1}...", flush=True)
                feedback_combined = ""
        else:
            print(f"\n  ! Достигнут лимит итераций ({MAX_ITERATIONS}). Сохраняю лучший вариант.")

    # Финальный код
    final_code = extract_code(code_raw)

    if save:
        out_file = DATA_DIR / "coder_output_v2.py"
        out_file.write_text(
            f"# Task: {task}\n# Generated by Coder Team v2\n\n{final_code}",
            encoding='utf-8'
        )
        print(f"\n  [Сохранено: {out_file}]")

    print(f"\n{'═'*60}")
    print("  CODER TEAM v2 — ЗАВЕРШЕНО")
    print(f"{'═'*60}\n")
    return code_raw

# ──────────────────────────────────────────────────────
# РЕЖИМ ПРОВЕРКИ ГОТОВОГО КОДА
# ──────────────────────────────────────────────────────
def run_review(code: str, task_desc: str = "", save: bool = True) -> str:
    """Пропускает архитектора и кодера — сразу проверяет готовый код ревьюером и тестировщиком.
    Если есть замечания — кодер исправляет, цикл повторяется до MAX_ITERATIONS раз.
    """
    task = task_desc or "Review and improve the provided code"
    plan = "(provided code, no architect plan)"

    print(f"\n{'═'*60}")
    print("  CODER TEAM v2 — ПРОВЕРКА КОДА")
    print(f"  LM Studio: {LM_URL}")
    print(f"  Макс. итераций: {MAX_ITERATIONS}")
    print(f"{'═'*60}")

    code_raw = f"```python\n{code}\n```"
    feedback_combined = ""

    def vote_status(ok):
        if ok is True:  return "✓ ОДОБРЕНО"
        if ok is False: return "✗ ЗАМЕЧАНИЯ"
        return "⚠ ПРОПУЩЕН"

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'═'*60}")
        print(f"  ИТЕРАЦИЯ {iteration} / {MAX_ITERATIONS}")
        print(f"{'═'*60}")

        current_code = extract_code(code_raw)

        # На первой итерации берём исходный код, далее — код после правок кодером
        if iteration > 1:
            code_raw = agent_coder(task, plan, feedback_combined)
            current_code = extract_code(code_raw)

        print(f"\n{'━'*60}")
        print("  ПРОВЕРКА (ревьюер + тестировщик)")
        print(f"{'━'*60}", flush=True)

        review_result, review_ok = agent_reviewer(task, plan, current_code)
        test_result,   test_ok   = agent_tester(task, current_code)

        votes = [v for v in [review_ok, test_ok] if v is not None]
        both_approved = len(votes) > 0 and all(votes)

        print(f"\n{'━'*60}")
        print(f"  ИТОГ ИТЕРАЦИИ {iteration}:")
        print(f"  Ревьюер:      {vote_status(review_ok)}")
        print(f"  Тестировщик:  {vote_status(test_ok)}")
        print(f"{'━'*60}", flush=True)

        if both_approved:
            who = "ОБА АГЕНТА" if all(v is not None for v in [review_ok, test_ok]) else "АКТИВНЫЕ АГЕНТЫ"
            print(f"\n  ✓ {who} ОДОБРИЛИ — КОД ГОТОВ (итерация {iteration})")
            break

        # Проверяем контрол-файл (остановка / комментарий от пользователя)
        ctrl = _check_control()
        if ctrl:
            stop, feedback_combined = _apply_control(ctrl, feedback_combined)
            if stop:
                break

        if iteration < MAX_ITERATIONS:
            parts = []
            if review_ok is False:
                parts.append(f"=== РЕВЬЮЕР ===\n{review_result}")
            if test_ok is False:
                parts.append(f"=== ТЕСТИРОВЩИК ===\n{test_result}")
            auto_feedback = "\n\n".join(parts)
            if auto_feedback:
                feedback_combined = auto_feedback + (
                    ("\n\n" + feedback_combined) if feedback_combined else ""
                )
            if auto_feedback or feedback_combined:
                print(f"\n  → Передаю замечания кодеру, итерация {iteration + 1}...", flush=True)
            else:
                print("\n  → Нет замечаний (только ошибки LM Studio), повторяю...", flush=True)
                feedback_combined = ""
        else:
            print(f"\n  ! Достигнут лимит итераций ({MAX_ITERATIONS}). Сохраняю лучший вариант.")

    final_code = extract_code(code_raw)

    if save:
        out_file = DATA_DIR / "coder_output_v2.py"
        out_file.write_text(
            f"# Review task: {task}\n# Reviewed by Coder Team v2\n\n{final_code}",
            encoding='utf-8'
        )
        print(f"\n  [Сохранено: {out_file}]")

    print(f"\n{'═'*60}")
    print("  CODER TEAM v2 — ЗАВЕРШЕНО")
    print(f"{'═'*60}\n")
    return code_raw


# ──────────────────────────────────────────────────────
# ИНТЕРАКТИВНЫЙ РЕЖИМ
# ──────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║           CODER TEAM v2 — АВТОНОМНЫЙ ЦИКЛ           ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Архитектор  — планирует решение                    ║")
    print("║  Кодер       — пишет и исправляет код               ║")
    print("║  Ревьюер     — проверяет баги и логику              ║")
    print("║  Тестировщик — генерирует и прогоняет тесты         ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Backend: {LM_URL:<42}║")
    print(f"║  Макс. итераций: {MAX_ITERATIONS:<37}║")
    print("║  /exit — выход                                       ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    while True:
        try:
            task = input("Задача: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break
        if not task: continue
        if task == "/exit": break
        run_pipeline(task)

if __name__ == "__main__":
    # --review <code_file> [task_desc]
    if len(sys.argv) > 1 and sys.argv[1] == "--review":
        code_file = sys.argv[2] if len(sys.argv) > 2 else None
        task_desc = sys.argv[3] if len(sys.argv) > 3 else ""
        if code_file and Path(code_file).exists():
            run_review(Path(code_file).read_text(encoding="utf-8"), task_desc)
        else:
            print("Использование: coder_team_v2.py --review <code_file> [task_desc]")
    elif len(sys.argv) > 1:
        run_pipeline(sys.argv[1], )
    else:
        main()
