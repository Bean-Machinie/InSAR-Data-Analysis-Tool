# InSAR Viewer install script
# Run once from any PowerShell: Set-Location insar-viewer; .\scripts\install.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$env:PATH = "$env:USERPROFILE\.local\bin;C:\Program Files\nodejs;$env:PATH"

Write-Host "=== InSAR Viewer Install ==="

# 1. Check WebView2
$wv2 = Get-ItemProperty "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" -ErrorAction SilentlyContinue
if (-not $wv2) {
    $wv2 = Get-ItemProperty "HKCU:\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" -ErrorAction SilentlyContinue
}
if ($wv2) {
    Write-Host "WebView2 runtime found: $($wv2.pv)"
} else {
    Write-Warning "WebView2 runtime not detected. On Windows 11 it should be pre-installed. If the app fails to open, download it from https://developer.microsoft.com/en-us/microsoft-edge/webview2/"
}

# 2. Backend dependencies
Write-Host "`n--- Installing Python dependencies ---"
Set-Location "$root\backend"
uv sync
Write-Host "Backend OK."

# 3. Frontend
Write-Host "`n--- Installing frontend dependencies ---"
Set-Location "$root\frontend"
npm install
Write-Host "`n--- Building frontend ---"
npm run build
Write-Host "Frontend OK."

# 4. Create Desktop shortcut
Write-Host "`n--- Creating desktop shortcut ---"
$pythonExe = & "$env:USERPROFILE\.local\bin\uv" run --project "$root\backend" python -c "import sys; print(sys.executable)" 2>$null
if (-not $pythonExe) {
    # Fall back to pythonw from PATH
    $pythonExe = (Get-Command pythonw.exe -ErrorAction SilentlyContinue)?.Source
}
$launchScript = "$root\scripts\launch.pyw"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut("$env:USERPROFILE\Desktop\InSAR Viewer.lnk")
$shortcut.TargetPath = $pythonExe
$shortcut.Arguments = "`"$launchScript`""
$shortcut.WorkingDirectory = $root
$shortcut.Description = "InSAR Deformation Viewer"
$shortcut.Save()
Write-Host "Shortcut created: $env:USERPROFILE\Desktop\InSAR Viewer.lnk"

Set-Location $root
Write-Host "`n=== Install complete ==="
Write-Host "Launch: double-click the 'InSAR Viewer' shortcut on your Desktop."
Write-Host "Dev mode: .\scripts\dev.ps1"
