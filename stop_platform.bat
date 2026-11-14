@echo off
TITLE Stop XAUUSD AI Trading Platform
echo ======================================================================
echo   Stopping Local AI Trading Platform Servers...
echo ======================================================================

for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do (
    if not "%%a"=="" if not "%%a"=="0" (
        echo Stopping Backend Process PID: %%a
        taskkill /F /PID %%a >nul 2>&1
    )
)

for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000') do (
    if not "%%a"=="" if not "%%a"=="0" (
        echo Stopping Frontend Process PID: %%a
        taskkill /F /PID %%a >nul 2>&1
    )
)

echo.
echo All platform services have been stopped.
timeout /t 2 >nul
