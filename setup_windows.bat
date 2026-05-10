@echo off
chcp 65001 >nul
echo === Deepfake Speech Detection - Windows Setup ===
echo.

:: Check Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9~3.11.
    pause
    exit /b 1
)

:: Check ffmpeg
where ffmpeg >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARN] ffmpeg not installed.
    echo.
    echo Install ffmpeg using one of:
    echo   1. winget install ffmpeg
    echo   2. chocolatey: choco install ffmpeg
    echo   3. Download from: https://ffmpeg.org/download.html
    echo.
    pause
)

:: Create venv
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install torch CPU (avoid CUDA ~2GB)
echo Installing torch CPU...
pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install torch.
    pause
    exit /b 1
)

:: Install remaining packages
echo Installing remaining packages...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install packages. Trying audio_classification_models separately...
    pip install git+https://github.com/awsaf49/audio_classification_models
)

echo.
echo === Setup complete! ===
echo.
echo Run app:   .venv\Scripts\python app.py
echo Run CLI:   .venv\Scripts\python predict.py --audio path/to/file.wav

pause
