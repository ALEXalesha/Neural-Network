@echo off
title Neural Network Lab — http://localhost:5000
cd /d "%~dp0"
echo ================================================
echo   Neural Network Lab
echo   http://localhost:5000
echo ================================================
echo.
start /B "" cmd /c "timeout /t 5 /nobreak > nul && start http://localhost:5000"
python web_app.py
pause
