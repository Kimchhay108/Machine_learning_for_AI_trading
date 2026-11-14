@echo off
TITLE Clean Up AI Trading Data Cache
echo ======================================================================
echo   XAUUSD AI Trading Platform - Data Cleanup Utility
echo ======================================================================
echo.
echo Cleaning temporary download cache and finished training labels...
echo (Preserving your active AI models, candle history, and paper balance)
echo.

cd /d "%~dp0"
.venv\Scripts\python.exe -m src.cli.pipeline_cli clean

pause
