@echo off
echo ============================================
echo   MT5 Trading Bot - Starting...
echo ============================================
echo.

:: Check MT5 is running
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if errorlevel 1 (
    echo [WARNING] MetaTrader 5 does not appear to be running.
    echo Please open MT5 and log in before starting the bot.
    echo.
    choice /C YN /M "Continue anyway?"
    if errorlevel 2 exit /b 0
)

:: Check .env exists
if not exist .env (
    echo [ERROR] .env file not found. Please run setup.bat first.
    pause
    exit /b 1
)

echo [OK] Starting bot...
echo [OK] Press Ctrl+C to stop the bot
echo.
python bot.py

echo.
echo Bot stopped.
pause
