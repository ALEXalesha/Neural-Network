# Сборка AlexGPT: PyInstaller (onedir) -> portable zip + NSIS-установщик.
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$root = Split-Path $PSScriptRoot
$version = [regex]::Match((Get-Content "$root\alexgpt\app_server.py" -Raw), 'VERSION = "([^"]+)"').Groups[1].Value
$py = "$PSScriptRoot\.buildvenv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "[init] Создаю окружение сборки (torch CPU, около 300 МБ)..."
    python -m venv .buildvenv
    & $py -m pip install --upgrade pip
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    & $py -m pip install -r "$root\requirements.txt" pyinstaller
}

Write-Host "[1/3] PyInstaller, версия $version"
& $py -m PyInstaller alexgpt.spec --clean --noconfirm --distpath dist --workpath build
if ($LASTEXITCODE) { throw "PyInstaller завершился с ошибкой" }

Write-Host "[2/3] Portable zip"
$portable = "dist\portable\AlexGPT"
Remove-Item dist\portable -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $portable | Out-Null
Copy-Item dist\AlexGPT\* $portable -Recurse
Set-Content "$portable\portable.txt" "Portable mode: settings, chats and logs are stored in the data folder next to AlexGPT.exe" -Encoding utf8
$zip = "dist\AlexGPT-$version-portable.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
tar.exe -a -cf $zip -C dist\portable AlexGPT
if ($LASTEXITCODE) { throw "Не удалось упаковать zip" }

Write-Host "[3/3] NSIS installer"
$makensis = "${env:ProgramFiles(x86)}\NSIS\makensis.exe"
if (Test-Path $makensis) {
    & $makensis /V2 "/DAPP_VERSION=$version" installer.nsi
    if ($LASTEXITCODE) { throw "NSIS завершился с ошибкой" }
} else {
    Write-Warning "NSIS не найден, установщик пропущен. Поставить: winget install NSIS.NSIS"
}

Get-ChildItem dist -File | Format-Table Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB) } }
