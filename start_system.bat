@echo off
setlocal enabledelayedexpansion

REM USB Autorun - Facial Recognition System
REM This script runs automatically when USB is inserted

title School Attendance System

REM Start HTTP server in background
if exist "C:\attend_env\Scripts\python.exe" (
    echo Starting HTTP server...
    start "" "C:\attend_env\Scripts\python.exe" -m http.server 8000 --bind 127.0.0.1
    timeout /t 2 /nobreak
    echo Starting points sync server...
    start "" "C:\attend_env\Scripts\python.exe" points_server.py
    timeout /t 1 /nobreak
) else if exist "venv\Scripts\python.exe" (
    echo Starting HTTP server...
    start "" "venv\Scripts\python.exe" -m http.server 8000 --bind 127.0.0.1
    timeout /t 2 /nobreak
    echo Starting points sync server...
    start "" "venv\Scripts\python.exe" points_server.py
    timeout /t 1 /nobreak
)

REM Sync attendance data from Excel file
if exist "sync_attendance.py" (
    if exist "venv\Scripts\python.exe" (
        echo Syncing attendance data...
        venv\Scripts\python.exe sync_attendance.py
    ) else if exist "C:\attend_env\Scripts\python.exe" (
        C:\attend_env\Scripts\python.exe sync_attendance.py
    )
)

REM Open classroom dashboard in browser via localhost
if exist "classroom_dashboard.html" (
    echo Opening classroom dashboard...
    timeout /t 1 /nobreak
    start "" "http://localhost:8000/classroom_dashboard.html"
)

REM Prefer packaged EXE if present (portable, no Python required)
if exist "AttendanceApp.exe" (
    echo Detected portable executable. Launching AttendanceApp.exe ...
    start "" "%CD%\AttendanceApp.exe"
    exit /b 0
)

REM Check if venv exists
if not exist "venv\" (
    echo ================================================
    echo   First Time Setup Required
    echo ================================================
    echo.
    echo Setting up system environment...
    echo This will take 10-15 minutes
    echo.
    
    REM Create virtual environment
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create environment
        echo Make sure Python 3.8+ is installed
        pause
        exit /b 1
    )
    
    REM Activate and install
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install face-recognition opencv-python numpy pandas openpyxl pillow
    
    echo.
    echo Setup complete! System is ready.
    echo.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the system
python facial_recognition_v2.py

REM If user closes, offer menu
:menu
cls
echo.
echo /
echo   School Attendance System
echo /
echo.
echo  1. Run Again
echo  2. Run Diagnostics
echo  3. Exit
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    python facial_recognition_v2.py
    goto menu
) else if "%choice%"=="2" (
    python diagnostics.py
    pause
    goto menu
) else (
    exit /b 0
)
