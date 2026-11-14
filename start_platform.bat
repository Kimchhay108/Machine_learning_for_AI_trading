@echo off
TITLE XAUUSD AI Trading Platform Launcher
echo ======================================================================
echo   XAUUSD Hybrid AI Trading Platform - Local Launcher
echo ======================================================================
echo.

cd /d "%~dp0"

echo [1/3] Starting FastAPI AI Backend (Port 8000)...
start "XAUUSD-Backend" /min cmd /c ".venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"

echo [2/3] Starting Next.js Trading Dashboard (Port 3000)...
cd frontend
start "XAUUSD-Frontend" /min cmd /c "npm run start"
cd ..

echo [3/3] Opening Trading Dashboard in your browser...
timeout /t 3 /nobreak >nul
start http://localhost:3000

echo.
echo ======================================================================
echo   Platform is running locally!
echo   Dashboard:  http://localhost:3000
echo   API Docs:   http://localhost:8000/docs
echo.
echo   To stop the platform anytime, run: stop_platform.bat
echo ======================================================================
pause
