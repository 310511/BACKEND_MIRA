# MIRA × Phoenix - Install Dependencies Script
# Run this if you get "ModuleNotFoundError"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MIRA × Phoenix - Installing Dependencies" -ForegroundColor Cyan
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

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow

if (Test-Path "requirements.txt") {
    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Error: Failed to install from requirements.txt!" -ForegroundColor Red
        Write-Host "Installing critical dependencies manually..." -ForegroundColor Yellow
        python -m pip install uvicorn fastapi gql[all] pydantic pydantic-settings python-multipart python-dotenv httpx aiofiles
    } else {
        Write-Host ""
        Write-Host "✓ All dependencies installed successfully!" -ForegroundColor Green
    }
} else {
    Write-Host "requirements.txt not found. Installing critical dependencies..." -ForegroundColor Yellow
    python -m pip install uvicorn fastapi gql[all] pydantic pydantic-settings python-multipart python-dotenv httpx aiofiles
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now start the server with:" -ForegroundColor Yellow
Write-Host "  .\start_server.ps1" -ForegroundColor Cyan
Write-Host ""
