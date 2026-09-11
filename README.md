# Neural Network Lab / AlexGPT

Учебный проект по нейросетям на PyTorch. Внутри два слоя:

- `alexgpt/` - готовое десктоп-приложение AlexGPT: 16 локальных нейросетей (распознавание цифр, тональность, спам, NER, перевод EN↔RU, своя маленькая GPT, GAN, прогнозы, кластеризация, рекомендации и т.д.) плюс чат, "Команда кодеров" и решатель тестов через LM Studio.
- `lab/` - скрипты, на которых эти модели обучались, и учебные эксперименты (MNIST, GAN, GPT с нуля и другие).

## Скачать

Готовые сборки лежат в [релизах](http://gitea.local:3000/ALEXaloysha/neuralnetwork/releases):

- `AlexGPT-<версия>-setup.zip` - установщик. Ставится без прав администратора в `%LOCALAPPDATA%\Programs\AlexGPT`, создаёт ярлыки.
- `AlexGPT-<версия>-portable.zip` - распаковать куда угодно и запустить `AlexGPT.exe`. Настройки и чаты хранятся в папке `data` рядом с exe.

Python на целевом компьютере не нужен, интернет тоже. Работает на CPU, видеокарта не обязательна.

Чат, "Решить тест" и "Команда кодеров" работают через [LM Studio](https://lmstudio.ai): запусти в нём Local Server (порт 1234) и загрузи модели из `alexgpt/lm_config.json`. Остальные разделы работают без него.

## Запуск из исходников

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python alexgpt\main.py
```

Веса моделей лежат в `models/` через Git LFS, поэтому перед клонированием нужен `git lfs install`.

Чтобы свои настройки LM Studio не терялись при обновлениях, положи изменённый `lm_config.json` в папку данных: `.appdata/` при запуске из исходников, `%LOCALAPPDATA%\AlexGPT` в установленной версии, `data/` в portable.

## Тесты

```bash
.venv\Scripts\python -m pytest tests
```

Тесты проверяют инварианты, а не отдельные примеры: hypothesis генерирует сотни случайных входов для каждого API и проверяет, что сервер никогда не падает с 500, отвечает валидным JSON без NaN, вероятности суммируются в 100%, метка совпадает с максимальной вероятностью, калькулятор совпадает с Python и так далее. Для 1000 примеров на тест: `HYPOTHESIS_PROFILE=thorough`.

## Сборка exe, portable и установщика

```bash
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

Скрипт сам создаст `packaging/.buildvenv` с torch CPU, соберёт PyInstaller-версию и положит в `packaging/dist/` portable-zip и установщик (нужен NSIS: `winget install NSIS.NSIS`).

## Структура

| Путь | Что там |
|------|---------|
| `alexgpt/main.py` | Точка входа: окно, сервер и дочерние процессы (в сборке это один exe с ключами `--server`, `--run`, `--pyfile`) |
| `alexgpt/gui.py` | Окно на PyQt6 + QtWebEngine |
| `alexgpt/app_server.py` | Flask-сервер со всеми моделями и API |
| `alexgpt/coder_team.py`, `coder_team_v2.py` | Команды LLM-агентов для написания и проверки кода |
| `alexgpt/paths.py` | Где лежат ресурсы, модели и пользовательские данные |
| `alexgpt/templates/app.html` | Весь интерфейс |
| `models/` | Веса и токенайзеры, которые использует приложение |
| `lab/` | Обучение моделей и учебные скрипты, см. ниже |
| `tests/` | pytest + hypothesis |
| `packaging/` | PyInstaller spec, NSIS-скрипт, иконка, `build.ps1` |

## Лаборатория

Скрипты в `lab/` запускаются из этой папки (`cd lab`), зависимости в `lab/requirements.txt`. Датасеты и результаты обучения (`*.pth`, `*.png`, тексты) остаются локально и в git не попадают. Чтобы приложение взяло новую модель, скопируй её `.pth` в `models/`.

- `mnist.py` → `mnist_cnn*.py` → `retrain_mnist_web.py` - от полносвязной сети до ResNet для рисованных цифр
- `mini_gpt.py`, `gpt_bpe.py`, `train_multilingual.py` - GPT с нуля, BPE-токенизация, многоязычная модель
- `translator.py`, `train_translator_bpe.py` - seq2seq-трансформер на корпусе Tatoeba
- `gan.py`, `image_processing.py` - условный GAN, денойзинг и суперразрешение
- `train_*.py` - остальные модели приложения (спам, NER, аномалии, дефекты, кластеры, рекомендации, температура)
- `regression.py`, `timeseries.py`, `sentiment.py`, `clustering.py`, `anomaly.py` - первые версии тех же задач с графиками
