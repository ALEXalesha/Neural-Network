@echo off
chcp 65001 > nul
echo ============================================
echo  Нейронная Лаборатория — пересборка всех EXE
echo ============================================
echo.

cd /d "c:\iCloudDrive\Алексей\Neural network"

echo [1/5] Проверка PyInstaller...
python -m pip install pyinstaller --quiet
if errorlevel 1 (
    echo ОШИБКА: не удалось установить PyInstaller
    pause & exit /b 1
)

echo.
echo [2/5] Menu.exe ...
python -m PyInstaller build\Menu.spec --distpath . --workpath build\_work_menu --noconfirm --clean
if errorlevel 1 ( echo ОШИБКА при сборке Menu.exe ) else ( echo OK: Menu.exe )

echo.
echo [3/5] Smart Assistant.exe ...
python -m PyInstaller build\Smart Assistant.spec --distpath . --workpath build\_work_sa --noconfirm --clean
if errorlevel 1 ( echo ОШИБКА при сборке Smart Assistant.exe ) else ( echo OK: Smart Assistant.exe )

echo.
echo [4/5] Smart Assistant Web.exe ...
python -m PyInstaller "build\Smart Assistant Web.spec" --distpath . --workpath build\_work_web --noconfirm --clean
if errorlevel 1 ( echo ОШИБКА при сборке Smart Assistant Web.exe ) else ( echo OK: Smart Assistant Web.exe )

echo.
echo [5/5] Neural Network Lab.exe ...
python -m PyInstaller "build\Neural Network Lab.spec" --distpath . --workpath build\_work_lab --noconfirm --clean
if errorlevel 1 ( echo ОШИБКА при сборке Neural Network Lab.exe ) else ( echo OK: Neural Network Lab.exe )

echo.
echo ============================================
echo  Готово! Все EXE пересобраны.
echo ============================================
echo.
pause
