# MIRA × Phoenix - Server Startup Script
# Run this script to start the backend server

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MIRA Voice Commerce - Starting Server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "app\main.py")) {
    Write-Host "Error: app\main.py not found!" -ForegroundColor Red
    Write-Host "Please run this script from the backend directory." -ForegroundColor Yellow
    exit 1
}

# Check Python
Write-Host "Checking Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Python not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.9+ from python.org" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green

# Check if dependencies are installed
Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Yellow

# Ensure dependencies are installed
if (Test-Path "requirements.txt") {
    Write-Host "Installing/updating dependencies from requirements.txt..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to install dependencies!" -ForegroundColor Red
        Write-Host "Try manually: python -m pip install -r requirements.txt" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "Error: requirements.txt not found!" -ForegroundColor Red
    exit 1
}

# Start server
Write-Host ""
Write-Host "Starting server..." -ForegroundColor Yellow
Write-Host "Server will be available at: http://localhost:8002" -ForegroundColor Cyan
Write-Host "API Documentation: http://localhost:8002/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8002
