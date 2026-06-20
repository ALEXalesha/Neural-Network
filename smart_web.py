"""
Smart Assistant — Web Interface (LM Studio backend)
Запуск: python smart_web.py
Открыть: http://localhost:5001

Модели запускаются через LM Studio (http://localhost:1234).
Настройки моделей и промпты — в lm_config.json
"""
import sys, io, os, json, base64, uuid, re, warnings, logging
import threading
from collections import deque
from datetime import datetime
warnings.filterwarnings('ignore', message='.*triton.*')
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

BASE = Path(__file__).parent

try:
    from flask import Flask, render_template, request, jsonify, Response, stream_with_context, render_template_string
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask', '-q'])
    from flask import Flask, render_template, request, jsonify, Response, stream_with_context, render_template_string

try:
    import requests as _requests
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'requests', '-q'])
    import requests as _requests

app = Flask(__name__, template_folder='templates')

# Глобальное состояние для остановки генерации
_stop_event      = threading.Event()
_active_response = None  # активное соединение к LM Studio

# ═══════════════════════════════════════════════════
# ЛОГИРОВАНИЕ
# ═══════════════════════════════════════════════════
_log_buffer = deque(maxlen=500)

class _MemHandler(logging.Handler):
    LEVEL_COLOR = {'DEBUG':'#888','INFO':'#58a6ff','WARNING':'#e3b341','ERROR':'#f85149','CRITICAL':'#ff7b72'}
    def emit(self, record):
        _log_buffer.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'level': record.levelname,
            'color': self.LEVEL_COLOR.get(record.levelname, '#ccc'),
            'msg': self.format(record),
        })

_mem_handler = _MemHandler()
_mem_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger('werkzeug').addHandler(_mem_handler)
logging.getLogger('werkzeug').setLevel(logging.INFO)

def log(msg, level='INFO'):
    color = _MemHandler.LEVEL_COLOR.get(level, '#ccc')
    _log_buffer.append({'time': datetime.now().strftime('%H:%M:%S'), 'level': level, 'color': color, 'msg': msg})
    print(msg)

_LOGS_HTML = '''<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Logs — Smart Assistant :5001</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:monospace;font-size:13px}
header{background:#161b22;border-bottom:1px solid #30363d;padding:12px 20px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:10}
h1{font-size:15px;color:#58a6ff}
.badge{background:#21262d;border:1px solid #30363d;border-radius:12px;padding:3px 10px;font-size:11px;color:#8b949e}
button{background:#21262d;border:1px solid #30363d;color:#c9d1d9;border-radius:6px;padding:5px 12px;font-size:12px;cursor:pointer}
button:hover{border-color:#58a6ff;color:#58a6ff}
#log{padding:12px 20px}
.row{display:flex;gap:10px;padding:3px 0;border-bottom:1px solid #161b22;line-height:1.5}
.row:hover{background:#161b22}
.t{color:#484f58;min-width:60px}
.lv{min-width:52px;font-weight:700}
.msg{white-space:pre-wrap;word-break:break-all;flex:1}
#status{font-size:11px;color:#484f58}
</style></head><body>
<header>
  <h1>📋 Логи — Smart Assistant</h1>
  <span class="badge">:5001</span>
  <span id="status">авто-обновление вкл</span>
  <button onclick="toggleAuto()">⏸ Пауза</button>
  <button onclick="document.getElementById(\'log\').innerHTML=\'\'">🗑 Очистить</button>
  <a href="http://localhost:5000/logs" style="margin-left:auto;color:#58a6ff;font-size:12px;text-decoration:none">→ Neural Network логи :5000</a>
</header>
<div id="log"></div>
<script>
let auto=true, lastCount=0;
function toggleAuto(){auto=!auto;document.getElementById('status').textContent=auto?'авто-обновление вкл':'пауза';}
async function poll(){
  if(!auto)return;
  try{
    const r=await fetch('/api/logs').then(r=>r.json());
    if(r.length!==lastCount){
      lastCount=r.length;
      const el=document.getElementById('log');
      el.innerHTML=r.map(e=>`<div class="row"><span class="t">${e.time}</span><span class="lv" style="color:${e.color}">${e.level}</span><span class="msg">${e.msg.replace(/</g,'&lt;')}</span></div>`).join('');
      el.scrollTop=el.scrollHeight;
    }
  }catch(e){}
}
setInterval(poll,1000);poll();
</script></body></html>'''

# ═══════════════════════════════════════════════════
# КОНФИГ из lm_config.json
# ═══════════════════════════════════════════════════

def load_config():
    path = BASE / 'lm_config.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {}

CFG = load_config()
LM_URL   = CFG.get('lm_studio_url', 'http://localhost:1234/v1')
MODELS   = CFG.get('models', {})
KEYWORDS = CFG.get('routing_keywords', {})

# ═══════════════════════════════════════════════════
# РОУТЕР
# ═══════════════════════════════════════════════════

def route(query: str, has_image: bool) -> str:
    if has_image:
        return 'vision'
    q = query.lower()
    scores = {name: sum(1 for kw in kws if kw in q) for name, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get) if scores else 'reasoner'
    return best if scores.get(best, 0) > 0 else 'reasoner'

# ═══════════════════════════════════════════════════
# МАРШРУТЫ
# ═══════════════════════════════════════════════════

@app.before_request
def _log_request():
    if not request.path.startswith('/api/logs'):
        log(f'{request.method} {request.path}', 'INFO')

@app.route('/')
def index():
    return render_template('smart.html')

@app.route('/logs')
def logs_page():
    return render_template_string(_LOGS_HTML)

@app.route('/api/logs')
def api_logs():
    return jsonify(list(_log_buffer))

@app.route('/api/status')
def status():
    # Проверяем что LM Studio запущена
    lm_ok = False
    lm_model = None
    try:
        r = _requests.get(f'{LM_URL}/models', timeout=2)
        if r.ok:
            data = r.json()
            models_list = data.get('data', [])
            lm_ok = True
            if models_list:
                lm_model = models_list[0].get('id', '')
    except Exception:
        pass

    models_info = {
        name: {
            'model_id': cfg.get('model_id', ''),
            'model_short': cfg.get('model_id', '').split('/')[-1].replace('.gguf', ''),
            'temperature': cfg.get('temperature', 0.7),
            'max_tokens': cfg.get('max_tokens', 2048),
        }
        for name, cfg in MODELS.items()
    }

    return jsonify({
        'device': 'LM Studio',
        'current_model': lm_model,
        'lm_studio_ok': lm_ok,
        'lm_studio_url': LM_URL,
        'loading': False,
        'cached': {name: True for name in MODELS},
        'models_info': models_info,
    })

@app.route('/api/reload_config', methods=['POST'])
def reload_config():
    """Перезагрузить lm_config.json без перезапуска сервера"""
    global CFG, LM_URL, MODELS, KEYWORDS
    CFG      = load_config()
    LM_URL   = CFG.get('lm_studio_url', 'http://localhost:1234/v1')
    MODELS   = CFG.get('models', {})
    KEYWORDS = CFG.get('routing_keywords', {})
    return jsonify({'ok': True, 'url': LM_URL, 'models': list(MODELS.keys())})

@app.route('/api/chat', methods=['POST'])
def chat():
    data         = request.json
    message      = data.get('message', '').strip()
    forced_model = data.get('model', 'auto')
    image_b64    = data.get('image', None)

    if not message:
        return jsonify({'error': 'Пустое сообщение'}), 400

    # Сохранить изображение если есть
    image_path = None
    if image_b64:
        img_bytes  = base64.b64decode(image_b64.split(',')[-1])
        image_path = str(BASE / '_upload_temp.jpg')
        with open(image_path, 'wb') as f:
            f.write(img_bytes)
        forced_model = 'vision'

    chosen = forced_model if forced_model != 'auto' else route(message, image_path is not None)
    log(f'[CHAT] модель={chosen} | сообщение: {message[:80]}{"…" if len(message)>80 else ""}')
    model_cfg = MODELS.get(chosen, MODELS.get('reasoner', {}))
    model_id  = model_cfg.get('model_id', '')

    def generate():
        try:
            # Проверяем какая модель сейчас загружена в LM Studio
            try:
                r = _requests.get(f'{LM_URL}/models', timeout=2)
                loaded_models = [m.get('id', '') for m in r.json().get('data', [])] if r.ok else []
                model_short = model_id.split('/')[-1].replace('.gguf', '')
                already_loaded = any(model_short.lower() in m.lower() for m in loaded_models)
                if not already_loaded:
                    log(f'[LM Studio] загружаю модель: {model_short}', 'WARNING')
                    yield f"data: {json.dumps({'type': 'status', 'text': f'⏳ Загружаю модель {model_short} в LM Studio... (30-60 сек)'})}\n\n"
                else:
                    log(f'[LM Studio] запрос к модели: {model_short}')
                    yield f"data: {json.dumps({'type': 'status', 'text': f'Отправляю запрос → {model_short}...'})}\n\n"
            except Exception:
                yield f"data: {json.dumps({'type': 'status', 'text': f'Отправляю запрос → {chosen}...'})}\n\n"

            yield f"data: {json.dumps({'type': 'model', 'text': chosen})}\n\n"

            # Строим сообщения
            system_prompt = model_cfg.get('system_prompt', '')
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})

            if image_path and chosen == 'vision':
                # Vision: base64 в контенте
                with open(image_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
                messages.append({'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                    {'type': 'text', 'text': message},
                ]})
            else:
                messages.append({'role': 'user', 'content': message})

            payload = {
                'model':       model_id,
                'messages':    messages,
                'stream':      True,
                'temperature': model_cfg.get('temperature', 0.7),
                'max_tokens':  model_cfg.get('max_tokens', 2048),
            }

            global _active_response
            _stop_event.clear()

            # Vision требует больше времени на обработку
            req_timeout = 300 if chosen == 'vision' else 120

            resp = _requests.post(
                f'{LM_URL}/chat/completions',
                json=payload,
                stream=True,
                timeout=req_timeout,
            )
            _active_response = resp

            if not resp.ok:
                err = resp.text[:500]
                yield f"data: {json.dumps({'type': 'error', 'text': f'LM Studio ошибка {resp.status_code}: {err}'})}\n\n"
                return

            # Читаем SSE поток от LM Studio
            got_tokens = False
            for line in resp.iter_lines():
                if _stop_event.is_set():
                    resp.close()
                    break
                if not line:
                    continue
                line = line.decode('utf-8') if isinstance(line, bytes) else line
                if line.startswith('data: '):
                    chunk = line[6:]
                    if chunk.strip() == '[DONE]':
                        break
                    try:
                        obj = json.loads(chunk)
                        # Проверяем на ошибку в самом ответе
                        if 'error' in obj:
                            yield f"data: {json.dumps({'type': 'error', 'text': obj['error'].get('message', str(obj['error']))})}\n\n"
                            break
                        delta = obj['choices'][0].get('delta', {})
                        token = delta.get('content', '')
                        if token:
                            got_tokens = True
                            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                    except Exception:
                        pass

            if not got_tokens and not _stop_event.is_set():
                log('[LM Studio] модель не вернула токены', 'ERROR')
                yield f"data: {json.dumps({'type': 'error', 'text': 'Модель не вернула ответ. Убедись что в LM Studio загружена vision-модель (Qwen2-VL) и сервер запущен.'})}\n\n"
            else:
                log(f'[LM Studio] ответ получен ✓')

            _active_response = None

            if image_path and Path(image_path).exists():
                os.remove(image_path)

            yield "data: {\"type\":\"done\"}\n\n"

        except _requests.exceptions.ConnectionError:
            log('[LM Studio] соединение отклонено — сервер не запущен', 'ERROR')
            yield f"data: {json.dumps({'type': 'error', 'text': 'LM Studio не запущена. Открой LM Studio, загрузи модель и запусти сервер (Local Server → Start).'})}\n\n"
        except Exception as e:
            log(f'[CHAT] ошибка: {e}', 'ERROR')
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/draw', methods=['POST'])
def draw():
    prompt = request.json.get('prompt', 'beautiful landscape')

    def stream():
        try:
            yield f"data: {json.dumps({'type': 'status', 'text': 'Загружаю SDXL-Turbo (~6 ГБ при первом запуске)...'})}\n\n"

            import torch
            from pathlib import Path as _Path
            from diffusers import AutoPipelineForText2Image
            import time

            pipe = AutoPipelineForText2Image.from_pretrained(
                'stabilityai/sdxl-turbo',
                torch_dtype=torch.float16,
                variant='fp16',
            ).to('cuda' if torch.cuda.is_available() else 'cpu')

            yield f"data: {json.dumps({'type': 'status', 'text': 'Генерирую изображение...'})}\n\n"

            image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]

            output_path = str(BASE / f'generated_{int(time.time())}.png')
            image.save(output_path)

            del pipe
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with open(output_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()

            yield f"data: {json.dumps({'type': 'image', 'image': 'data:image/png;base64,' + b64, 'path': output_path})}\n\n"
            yield "data: {\"type\":\"done\"}\n\n"

        except ImportError:
            yield f"data: {json.dumps({'type': 'error', 'text': 'Установи diffusers: pip install diffusers accelerate'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return Response(stream_with_context(stream()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/stop', methods=['POST'])
def stop_generation():
    global _active_response
    _stop_event.set()
    if _active_response:
        try:
            _active_response.close()
        except Exception:
            pass
        _active_response = None
    return jsonify({'ok': True})

@app.route('/api/unload', methods=['POST'])
def unload():
    return jsonify({'ok': True})

# ─── Coder Team ───
@app.route('/api/coder_team/run', methods=['POST'])
def coder_team_run():
    import subprocess as sp
    task = request.json.get('task', '').strip()
    if not task:
        return jsonify({'error': 'Задача не указана'}), 400

    script = str(BASE / 'coder_team.py')

    def generate():
        proc = sp.Popen(
            [sys.executable, '-u', script, task],
            stdout=sp.PIPE, stderr=sp.STDOUT,
            text=True, encoding='utf-8', errors='replace',
            cwd=str(BASE)
        )
        for line in proc.stdout:
            yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
        proc.wait()
        out_file = BASE / 'coder_output.py'
        if out_file.exists():
            code = out_file.read_text(encoding='utf-8')
            yield f"data: {json.dumps({'done': True, 'code': code})}\n\n"
        else:
            yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# ═══════════════════════════════════════════════════
# ИСТОРИЯ ЧАТОВ
# ═══════════════════════════════════════════════════
CHATS_DIR = BASE / 'chats'
CHATS_DIR.mkdir(exist_ok=True)

@app.route('/api/chats')
def chats_list():
    chats = []
    for f in sorted(CHATS_DIR.glob('*.json'), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            chats.append({
                'id':      f.stem,
                'title':   data.get('title', 'Чат'),
                'count':   len([m for m in data.get('messages', []) if m.get('role') != 'bot_pending']),
                'created': data.get('created', ''),
            })
        except Exception:
            pass
    return jsonify(chats)

@app.route('/api/chats/save', methods=['POST'])
def chats_save():
    data    = request.json
    chat_id = data.get('id') or str(uuid.uuid4())[:8]
    path    = CHATS_DIR / f'{chat_id}.json'
    path.write_text(json.dumps({
        'id':       chat_id,
        'title':    data.get('title', 'Чат'),
        'created':  data.get('created', ''),
        'messages': data.get('messages', []),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return jsonify({'id': chat_id})

@app.route('/api/chats/<chat_id>')
def chats_get(chat_id):
    path = CHATS_DIR / f'{chat_id}.json'
    if not path.exists():
        return jsonify({'error': 'Не найден'}), 404
    return jsonify(json.loads(path.read_text(encoding='utf-8')))

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def chats_delete(chat_id):
    path = CHATS_DIR / f'{chat_id}.json'
    if path.exists():
        path.unlink()
    return jsonify({'ok': True})

@app.route('/api/chats/<chat_id>/rename', methods=['POST'])
def chats_rename(chat_id):
    path = CHATS_DIR / f'{chat_id}.json'
    if not path.exists():
        return jsonify({'error': 'Не найден'}), 404
    data = json.loads(path.read_text(encoding='utf-8'))
    data['title'] = request.json.get('title', data.get('title', ''))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return jsonify({'ok': True})

if __name__ == '__main__':
    print("=" * 50)
    print("  Smart Assistant — Web Interface")
    print(f"  Backend: LM Studio ({LM_URL})")
    print("  http://localhost:5001")
    print("=" * 50)
    print()
    print("  Настройки моделей: lm_config.json")
    print("  Убедись что LM Studio запущена и")
    print("  сервер включён (Local Server → Start)")
    print()
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
