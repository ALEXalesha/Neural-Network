# -*- mode: python ; coding: utf-8 -*-
"""
AlexGPT — onedir-сборка со всем внутри: GUI, Flask-сервер, torch CPU, модели.
Собирается через build.ps1 (он же делает portable-zip и NSIS-установщик).
Целевой машине нужны только Windows 10/11 x64: без Python и без интернета.
"""
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent
APP  = ROOT / 'alexgpt'

datas, binaries, hidden = [], [], []
for pkg in ('torch', 'tokenizers', 'numpy', 'PIL'):
    d, b, h = collect_all(pkg, exclude_datas=['include/**/*', 'test/**/*', '**/*.h', '**/*.cuh'])
    datas += d; binaries += b; hidden += h

datas += [
    (str(APP / 'templates'), 'templates'),
    (str(APP / 'static'), 'static'),
    (str(APP / 'lm_config.json'), '.'),
    # Рисование работает в отдельном Python «модуля рисования», сам скрипт едет с программой
    (str(APP / 'draw_worker.py'), '.'),
]
datas += [(str(p), 'models') for p in sorted((ROOT / 'models').iterdir()) if p.suffix in ('.pth', '.json')]

# Модули, которые запускаются через `AlexGPT.exe --run`, и stdlib для тестов,
# которые пишет Coder Team v2 (они выполняются внутри exe через --pyfile)
hidden += ['app_server', 'gui', 'paths', 'coder_team', 'coder_team_v2', 'sketch_segment']
hidden += ['unittest', 'unittest.mock', 'doctest', 'dataclasses', 'decimal', 'fractions', 'statistics',
           'heapq', 'bisect', 'csv', 'sqlite3', 'string', 'textwrap', 'calendar', 'uuid', 'hmac',
           'secrets', 'pprint', 'queue', 'argparse', 'difflib', 'fnmatch', 'glob', 'shutil', 'zipfile',
           'gzip', 'xml.etree.ElementTree', 'html.parser', 'asyncio', 'concurrent.futures']

a = Analysis(
    [str(APP / 'main.py')],
    pathex=[str(APP)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=[
        'diffusers', 'transformers', 'accelerate', 'torchvision', 'torchaudio',
        'matplotlib', 'IPython', 'jupyter', 'tkinter', '_tkinter', 'test',
        'scipy', 'sklearn', 'pandas', 'torch.utils.tensorboard',
    ],
    noarchive=False,
    optimize=1,
)

# ── Обрезка Qt: модули и данные, которые QtWebEngine не нужны ───────────────
_QT_BIN_KILL = (
    'qt6quick3d', 'qt63dcore', 'qt63drender', 'qt63dinput', 'qt63danimation', 'qt63dextras',
    'qt63dlogic', 'qt63dquick', 'qt6remoteobjects', 'qt6texttospeech', 'qt6serialport',
    'qt6bluetooth', 'qt6nfc', 'qt6charts', 'qt6datavisualization', 'qt6statemachine', 'qt6test',
    'qt6sql', 'qt6help', 'qt6designer', 'qt6quickcontrols2imagine', 'qt6quickcontrols2material',
    'qt6quickcontrols2universal', 'qt6quickparticles', 'qt6quicktimeline', 'qt6quickvectorimage',
)
_QML_KILL = ('qtquick3d', 'qtmultimedia', 'qttest', 'qtsensors', 'qtremoteobjects', 'qttexttospeech',
             'qtcharts', 'qtdatavisualization')


def _keep_binary(entry):
    low = entry[0].lower().replace('\\', '/')
    return not any(k in low for k in _QT_BIN_KILL)


def _keep_data(entry):
    dest = entry[0].lower().replace('\\', '/')
    if any(f'/qml/{k}' in dest for k in _QML_KILL):
        return False
    if '/translations/' in dest and not any(k in dest.split('/')[-1] for k in ('_ru.', '_en.', 'en_us')):
        return False
    if '/plugins/' in dest and any(k in dest for k in ('geometryloaders', 'sqldrivers', 'sceneparsers', 'designer')):
        return False
    return '/torch/include/' not in dest and '/torch/test/' not in dest


a.binaries = [b for b in a.binaries if _keep_binary(b)]
a.datas = [d for d in a.datas if _keep_data(d)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AlexGPT',
    console=False,
    upx=False,
    icon=str(Path(SPECPATH) / 'alexgpt.ico'),
)

coll = COLLECT(exe, a.binaries, a.datas, upx=False, name='AlexGPT')
