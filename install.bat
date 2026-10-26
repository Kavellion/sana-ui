@echo off
REM Sana UI one-click installer (Windows).
REM
REM What it does:
REM   1. Creates a Python virtual environment (venv) in this folder.
REM   2. Installs PyTorch with CUDA support (needs an NVIDIA GPU + a
REM      recent driver) and the rest of the requirements.
REM   3. Creates images_out\ for finished pictures.
REM
REM What it does NOT do:
REM   - Download the Sana model weights. Diffusers downloads the
REM     model (about 10 GB of bf16 weights) from Hugging Face on the
REM     first Generate, into the Hugging Face cache on your user
REM     profile -- not into this folder.
REM
REM Requirements: Windows, Python 3.10 or newer on your PATH, an NVIDIA
REM GPU (12 GB recommended), about 20 GB free disk. Everything here is
REM free -- no account, no payment, no server, no token: the model
REM repo is public and not gated.

setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo Python was not found. Install Python 3.10 or newer from
    echo python.org, tick "Add python.exe to PATH", then run this again.
    echo.
    pause
    exit /b 1
)

echo Python found:
python --version

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo Installing PyTorch with CUDA support (this is the big download)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo PyTorch install failed. Check your internet connection and
    echo that your GPU driver is up to date, then run this again.
    pause
    exit /b 1
)

echo Installing the remaining packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo Package install failed. Run this again -- pip resumes
    echo downloads it already finished.
    pause
    exit /b 1
)

if not exist images_out mkdir images_out

echo.
echo Done. Double-click run.bat to start Sana UI.
echo.
pause
