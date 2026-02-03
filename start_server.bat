@echo off
REM MIRA × Phoenix - Server Startup Script (Windows Batch)
REM Run this script to start the backend server

echo ========================================
echo MIRA Voice Commerce - Starting Server
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "app\main.py" (
    echo Error: app\main.py not found!
    echo Please run this script from the backend directory.
    pause
    exit /b 1
)

REM Check Python
echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found!
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

REM Start server
echo.
echo Starting server...
echo Server will be available at: http://localhost:8002
echo API Documentation: http://localhost:8002/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8002
