@echo off
setlocal

echo ==============================
echo Starting project setup...
echo ==============================

REM Go to the folder where this .bat file exists
cd /d "%~dp0"

REM Check Python 3.11
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.11 is not installed.
    echo Install Python 3.11, then run this file again.
    pause
    exit /b 1
)

echo Using Python:
py -3.11 --version

REM Create virtual environment if missing
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3.11 -m venv venv
) else (
    echo Virtual environment already exists.
)

REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
if exist "requirements.txt" (
    echo Installing requirements...
    pip install -r requirements.txt

    if errorlevel 1 (
        echo.
        echo Dependency installation failed. Project will not run.
        pause
        exit /b 1
    )
) else (
    echo No requirements.txt found. Skipping dependency install.
)

echo ==============================
echo Running project...
echo ==============================

python app.py

pause