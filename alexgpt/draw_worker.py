"""Рисование по описанию (SDXL-Turbo) в отдельном процессе.

Запускается Python-ом «модуля рисования» (свой torch + diffusers в папке данных) или, при запуске
из исходников, текущим Python, если в нём есть diffusers. Сервер AlexGPT общается с процессом строками JSON:
  первая строка вывода — {"ready": true, "device": "cuda"} после загрузки модели (или {"error": ...});
  на вход {"prompt": "...", "steps": 1} → на выход {"image": "data:image/png;base64,..."} или {"error": ...}.
Остановка генерации — появление файла ALEXGPT_DRAW_STOP. Процесс завершается, когда закрыт stdin.
"""
import base64
import io
import json
import os
import sys
from pathlib import Path

STOP = Path(os.environ.get("ALEXGPT_DRAW_STOP", "draw.stop"))
MODEL = os.environ.get("ALEXGPT_DRAW_MODEL", "stabilityai/sdxl-turbo")

# stdout — только для ответов; всё, что печатают библиотеки, уходит в stderr (лог)
OUT = sys.stdout
sys.stdout = sys.stderr


def send(obj):
    OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    OUT.flush()


def main():
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        cuda = torch.cuda.is_available()
        # В кэше лежат только fp16-веса; на CPU они загружаются и приводятся к float32
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL, variant="fp16", torch_dtype=torch.float16 if cuda else torch.float32)
        pipe = pipe.to("cuda" if cuda else "cpu")
        pipe.set_progress_bar_config(disable=True)
    except Exception as e:
        send({"error": f"Не удалось загрузить модель рисования: {e}"})
        return
    send({"ready": True, "device": "cuda" if cuda else "cpu"})

    def check_stop(_pipe, _step, _timestep, kwargs):
        if STOP.exists():
            raise InterruptedError
        return kwargs

    for line in sys.stdin:
        try:
            req = json.loads(line)
            steps = max(1, min(int(req.get("steps") or (1 if cuda else 4)), 8))
            out = pipe(prompt=str(req["prompt"]), num_inference_steps=steps, guidance_scale=0.0,
                       width=512, height=512, callback_on_step_end=check_stop)
            buf = io.BytesIO()
            out.images[0].save(buf, format="PNG")
            send({"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()})
        except InterruptedError:
            send({"error": "Остановлено"})
        except Exception as e:
            send({"error": f"Ошибка генерации: {e}"})


if __name__ == "__main__":
    main()
