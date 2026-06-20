@echo off
title Smart Assistant Web — http://localhost:5001
cd /d "%~dp0"
echo ================================================
echo   Smart Assistant Web Interface
echo   http://localhost:5001
echo ================================================
echo.
start /B "" cmd /c "timeout /t 6 /nobreak > nul && start http://localhost:5001"
python smart_web.py
pause
