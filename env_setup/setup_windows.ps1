# One-click setup of EdiVal + its GUI on Windows 10/11 with an NVIDIA GPU (24 GB+ VRAM).
#
# How to run (from any folder):
#   1. Open PowerShell as Administrator (Start menu -> type "PowerShell" -> "Run as administrator")
#   2. Paste:
#        Set-ExecutionPolicy -Scope Process Bypass -Force; irm https://raw.githubusercontent.com/lol2232/EdiVal/add-gradio-gui/env_setup/setup_windows.ps1 -OutFile $env:TEMP\setup_edival.ps1; & $env:TEMP\setup_edival.ps1
#
# The evaluator relies on vLLM, which does not run natively on Windows, so everything is
# installed inside WSL2 (Ubuntu 22.04). The GUI still opens in your normal Windows browser.
# Safe to re-run: if a reboot is needed, run the same command again afterwards.

$ErrorActionPreference = "Stop"
$Distro = "Ubuntu-22.04"
$RepoUrl = "https://github.com/lol2232/EdiVal.git"
$Branch = "add-gradio-gui"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`n$msg" -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { Fail "Please run PowerShell as Administrator (right-click PowerShell -> Run as administrator)." }

# --- 1. GPU check -------------------------------------------------------------------------------
Step "Checking NVIDIA GPU"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Fail "No NVIDIA driver found. Install the latest driver from https://www.nvidia.com/Download/index.aspx, reboot, and run this again."
}
$gpu = (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | Select-Object -First 1).Split(",")
$gpuName = $gpu[0].Trim()
$vramGB = [math]::Round([double]$gpu[1].Trim() / 1024, 1)
Write-Host "Found $gpuName with $vramGB GB VRAM"
if ($vramGB -lt 23) {
    Write-Host "EdiVal's judge model (Qwen2-VL-7B) needs about 24 GB of GPU memory; with $vramGB GB it will most likely fail to load." -ForegroundColor Yellow
    if ((Read-Host "Continue anyway? (y/N)") -ne "y") { exit 1 }
}
# Qwen2-VL-7B needs ~18 GB reserved; the other models use the rest.
$gpuMem = ([math]::Round([math]::Min(0.8, [math]::Max(0.5, 18 / $vramGB)), 2)).ToString([Globalization.CultureInfo]::InvariantCulture)

# --- 2. WSL2 + Ubuntu ---------------------------------------------------------------------------
Step "Checking WSL2 / $Distro"
$distros = @()
try { $distros = (wsl.exe --list --quiet) -replace "`0", "" | ForEach-Object { $_.Trim() } | Where-Object { $_ } } catch {}
if (-not ($distros -contains $Distro)) {
    Write-Host "Installing WSL2 and $Distro..."
    wsl.exe --install -d $Distro
    Write-Host @"

WSL is being installed. Next:
  1. Reboot if Windows asks you to.
  2. An Ubuntu window will open and ask for a NEW Linux username and password - pick anything
     and remember the password (you will need it during setup).
  3. Close that Ubuntu window and run this same setup command again.
"@ -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 0
}
wsl.exe --update | Out-Null
wsl.exe --set-default-version 2 | Out-Null

# --- 3. Clone repo + run Linux setup -------------------------------------------------------------
Step "Getting the EdiVal code inside Ubuntu"
wsl.exe -d $Distro -- bash -lc "command -v git >/dev/null || (sudo apt-get update -y && sudo apt-get install -y git); if [ -d ~/EdiVal/.git ]; then cd ~/EdiVal && git fetch origin $Branch && git checkout $Branch && git pull --ff-only origin $Branch; else git clone -b $Branch $RepoUrl ~/EdiVal; fi"
if ($LASTEXITCODE -ne 0) { Fail "Could not download the repository." }

Step "Installing everything inside Ubuntu (30-60 min, mostly downloads)"
wsl.exe -d $Distro -- bash -lc "bash ~/EdiVal/env_setup/setup_wsl.sh"
if ($LASTEXITCODE -ne 0) { Fail "Linux setup failed - scroll up for the first error and send it to Claude." }

# --- 4. Desktop launcher ------------------------------------------------------------------------
Step "Creating desktop launcher"
$launcher = Join-Path ([Environment]::GetFolderPath("Desktop")) "Start EdiVal GUI.bat"
@"
@echo off
title EdiVal GUI
echo Loading models - the first start downloads ~20 GB and can take a while.
echo Your browser will open automatically when the GUI is ready. Close this window to stop the GUI.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "while (`$true) { try { Invoke-WebRequest http://localhost:7860 -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process http://localhost:7860; break } catch { Start-Sleep 5 } }"
wsl.exe -d $Distro -- bash -lc "source ~/miniforge3/bin/activate edival && cd ~/EdiVal && python app.py --tp 1 --gpu-mem $gpuMem"
pause
"@ | Set-Content -Path $launcher -Encoding ASCII
Write-Host "Created: $launcher"

Step "All done"
Write-Host "Double-click 'Start EdiVal GUI' on your desktop. The browser opens at http://localhost:7860 once the models are loaded." -ForegroundColor Green
Read-Host "Press Enter to close"
