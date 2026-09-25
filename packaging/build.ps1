# AlexGPT build: PyInstaller (onedir) -> portable zip + NSIS installer.
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
# Messages are ASCII on purpose: Windows PowerShell 5.1 reads BOM-less scripts as ANSI.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$root = Split-Path $PSScriptRoot
$version = [regex]::Match((Get-Content "$root\alexgpt\app_server.py" -Raw), 'VERSION = "([^"]+)"').Groups[1].Value
$py = "$PSScriptRoot\.buildvenv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "[init] Creating build venv (torch CPU, about 300 MB)..."
    python -m venv .buildvenv
    & $py -m pip install --upgrade pip
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    & $py -m pip install -r "$root\requirements.txt" pyinstaller
}

Write-Host "[1/3] PyInstaller, version $version"
& $py -m PyInstaller alexgpt.spec --clean --noconfirm --distpath dist --workpath build
if ($LASTEXITCODE) { throw "PyInstaller failed" }

Write-Host "[2/3] Portable zip"
$portable = "dist\portable\AlexGPT"
Remove-Item dist\portable -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $portable | Out-Null
Copy-Item dist\AlexGPT\* $portable -Recurse
Set-Content "$portable\portable.txt" "Portable mode: settings, chats and logs are stored in the data folder next to AlexGPT.exe" -Encoding ascii
$zip = "dist\AlexGPT-$version-portable.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
# Windows tar (bsdtar) by full path: from Git Bash PATH finds GNU tar first, which ignores -a
# and writes a plain tar under the .zip name (1.4.0 build caught it: 1.1 GB, not a zip).
& "$env:SystemRoot\System32\tar.exe" -a -cf $zip -C dist\portable AlexGPT
if ($LASTEXITCODE) { throw "zip failed" }
# The staging copy is only needed for the zip; it duplicates dist\AlexGPT (~1.2 GB)
Remove-Item dist\portable -Recurse -Force

Write-Host "[3/3] NSIS installer"
$makensis = "${env:ProgramFiles(x86)}\NSIS\makensis.exe"
if (Test-Path $makensis) {
    & $makensis /V2 "/DAPP_VERSION=$version" installer.nsi
    if ($LASTEXITCODE) { throw "NSIS failed" }
} else {
    Write-Warning "NSIS not found, installer skipped. Install it with: winget install NSIS.NSIS"
}

# PyInstaller work files are rebuilt with --clean anyway
Remove-Item build -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem dist -File | Format-Table Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB) } }
