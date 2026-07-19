@echo off
echo ============================================
echo   MT5 Trading Bot - Setup
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please download Python from https://python.org/downloads
    echo Make sure to tick "Add Python to PATH" during install!
    echo.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

:: Create logs folder
if not exist logs mkdir logs
echo [OK] logs folder ready

:: Create .env if it does not exist
if exist .env (
    echo [OK] .env already exists - skipping credential setup
    goto install_deps
)

echo.
echo ============================================
echo   Enter your credentials
echo ============================================
echo.

set /p MT5_LOGIN=MT5 Account Number:
set /p MT5_PASSWORD=MT5 Password (press Enter if none):
set /p MT5_SERVER=MT5 Server (e.g. Axi-US50-Demo):
set /p TG_TOKEN=Telegram Bot Token:
set /p TG_CHAT=Your Telegram Chat ID:
set /p TG_CHAT2=Second Telegram Chat ID (optional, press Enter to skip):
set /p SYMBOLS=Symbols to trade (e.g. XAUUSD,BTCUSD):
echo.
echo Fixed lot sizes (leave blank to use automatic risk-based sizing):
set /p XAUUSD_LOT=XAUUSD fixed lot size (e.g. 0.01, press Enter to skip):
set /p BTCUSD_LOT=BTCUSD fixed lot size (e.g. 0.03, press Enter to skip):

:: Build chat ID list
set TG_CHATS=%TG_CHAT%
if not "%TG_CHAT2%"=="" set TG_CHATS=%TG_CHAT%,%TG_CHAT2%

echo.
echo [..] Creating .env file...

(
    echo MT5_LOGIN=%MT5_LOGIN%
    echo MT5_PASSWORD=%MT5_PASSWORD%
    echo MT5_SERVER=%MT5_SERVER%
    echo MT5_PATH=
    echo.
    echo TELEGRAM_BOT_TOKEN=%TG_TOKEN%
    echo TELEGRAM_ALLOWED_CHAT_IDS=%TG_CHATS%
    echo.
    echo SYMBOLS=%SYMBOLS%
    echo TIMEZONE=UTC
    echo MAX_RISK_PERCENT=2.0
    echo.
    echo XAUUSD_LOT=%XAUUSD_LOT%
    echo BTCUSD_LOT=%BTCUSD_LOT%
    echo.
    echo SL_ATR_MULTIPLIER=1.0
    echo RR_RATIO=3.0
    echo.
    echo SCALP_MODE=true
    echo SCAN_INTERVAL_SECONDS=60
    echo MAX_POSITIONS_PER_SYMBOL=3
    echo MAX_TRADE_AGE_MINUTES=15
    echo.
    echo EMA_FAST=20
    echo EMA_SLOW=50
    echo EMA_100=100
    echo SMA_200=200
    echo RSI_PERIOD=14
    echo MACD_FAST=12
    echo MACD_SLOW=26
    echo MACD_SIGNAL=9
    echo ATR_PERIOD=14
    echo ADX_PERIOD=14
    echo SR_LOOKBACK=50
    echo.
    echo LOG_LEVEL=INFO
    echo LOG_FILE=logs/trading_bot.log
) > .env

echo [OK] .env file created

:install_deps
echo.
echo [..] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency install failed.
    echo Try running manually: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo Next steps:
echo  1. Make sure MetaTrader 5 is open and logged in
echo  2. Double-click run.bat to start the bot
echo  3. Watch for a Telegram message confirming it is live
echo.
pause
