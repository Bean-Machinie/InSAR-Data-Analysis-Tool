# Start backend and frontend dev servers in parallel
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$env:PATH = "$env:USERPROFILE\.local\bin;C:\Program Files\nodejs;$env:PATH"

# Backend
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$root\backend'; uv run uvicorn 'insar_viewer.app:create_app' --factory --reload --port 8050 --host 127.0.0.1"

# Frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$root\frontend'; npm run dev"

Write-Host "Dev servers started. Backend: http://127.0.0.1:8050  Frontend: http://localhost:5173"
