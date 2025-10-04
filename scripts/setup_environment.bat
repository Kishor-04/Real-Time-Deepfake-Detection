@echo off
REM Video Deepfake Detection - Environment Setup Script (Windows)
REM Author: Kishor-04
REM Date: 2025-01-04

echo ======================================================================
echo Video Deepfake Detection - Environment Setup
echo    Author: Kishor-04
echo ======================================================================

echo.
echo Creating virtual environment...
python -m venv venv
echo    Done: Virtual environment created

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
pip install --upgrade pip --quiet

echo.
echo Installing dependencies...
pip install -r requirements.txt --quiet
echo    Done: Dependencies installed

echo.
echo Checking GPU availability...
python -c "import torch; print(f'   CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo.
echo Creating directory structure...
mkdir data\raw\real 2>nul
mkdir data\raw\fake 2>nul
mkdir data\processed\frames 2>nul
mkdir data\processed\faces 2>nul
mkdir models\pretrained 2>nul
mkdir models\checkpoints 2>nul
mkdir results 2>nul
mkdir runs 2>nul
echo    Done: Directories created

echo.
echo ======================================================================
echo Setup completed successfully!
echo ======================================================================
echo.
echo Next steps:
echo    1. Place your videos in data\raw\real\ and data\raw\fake\
echo    2. Run: python main.py --mode all
echo.
echo ======================================================================
pause