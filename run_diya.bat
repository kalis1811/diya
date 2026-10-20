@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo Could not find uv. Install uv first, then run this file again.
    pause
    exit /b 1
)

set "FLUTTER_CMD="
if defined FLUTTER_ROOT if exist "%FLUTTER_ROOT%\bin\flutter.bat" set "FLUTTER_CMD=%FLUTTER_ROOT%\bin\flutter.bat"
if not defined FLUTTER_CMD if exist "C:\flutter_windows_3.38.7-stable\flutter\bin\flutter.bat" set "FLUTTER_CMD=C:\flutter_windows_3.38.7-stable\flutter\bin\flutter.bat"
if not defined FLUTTER_CMD (
    where flutter >nul 2>&1
    if not errorlevel 1 set "FLUTTER_CMD=flutter"
)
if not defined FLUTTER_CMD (
    echo Could not find Flutter. Set FLUTTER_ROOT or add Flutter to PATH, then run this file again.
    pause
    exit /b 1
)

echo Building the Diya Flutter app...
pushd "%~dp0flutter_frontend"
call "%FLUTTER_CMD%" build windows --release
if errorlevel 1 (
    popd
    echo Flutter build failed. Check the message above and try again.
    pause
    exit /b 1
)
popd

powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-NetConnection 127.0.0.1 -Port 8400 -InformationLevel Quiet) { exit 1 }"
if not errorlevel 1 goto port_is_free
echo Port 8400 is already in use. Stop the running Diya/API window with Ctrl+C, then run this file again.
pause
exit /b 1

:port_is_free
echo Preparing Diya's Python environment...
uv sync
if errorlevel 1 (
    echo Dependency setup failed. Check the message above and try again.
    pause
    exit /b 1
)

echo Checking the local admin login...
uv run python -c "from brain.config import get_settings; import sys; sys.exit(0 if len(get_settings().admin_password) >= 10 else 1)"
if errorlevel 1 (
    echo First run: set up the private owner login.
    call uv run python scripts/setup_admin.py
    if errorlevel 1 (
        echo Admin setup did not complete.
        pause
        exit /b 1
    )
)

echo Checking Ollama and required models...
call uv run python scripts/check_models.py
if errorlevel 1 (
    pause
    exit /b 1
)

echo Updating the local document index...
uv run python -m brain.ingest
if errorlevel 1 (
    echo Indexing failed. Check that Ollama is running and the embedding model is installed.
    pause
    exit /b 1
)

where docker >nul 2>&1
if errorlevel 1 goto check_mongodb
echo Starting MongoDB with Docker...
docker compose up -d mongodb
if errorlevel 1 echo Docker could not start MongoDB. Checking for an existing MongoDB server...
if errorlevel 1 goto check_mongodb
goto mongodb_ready

:check_mongodb
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-NetConnection 127.0.0.1 -Port 27017 -InformationLevel Quiet) { exit 0 } else { exit 1 }"
if errorlevel 1 (
    echo MongoDB is not installed or running. Diya will open with local fallback storage.
    echo Conversations will not be saved to MongoDB until you install or start MongoDB.
) else (
    echo Using the MongoDB server already running on this computer.
)

:mongodb_ready
echo Starting Diya API in a new window...
start "Diya API" /D "%~dp0" cmd /k "uv run python -m brain.api"

echo Waiting for Diya to become ready...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ready=$false; for($i=0;$i -lt 60;$i++){try{$r=Invoke-RestMethod -Uri 'http://127.0.0.1:8400/health' -TimeoutSec 2; if($r){$ready=$true; break}}catch{}; Start-Sleep -Seconds 1}; if(-not $ready){exit 1}"
if errorlevel 1 (
    echo Diya did not respond within 60 seconds. Check the Diya API window for errors.
    pause
    exit /b 1
)

set "FLUTTER_DIR=%~dp0flutter_frontend\build\windows\x64\runner\Release"
set "FLUTTER_APP=%FLUTTER_DIR%\flutter_frontend.exe"
if not exist "%FLUTTER_APP%" (
    echo Flutter app was not found at "%FLUTTER_APP%".
    echo Build the Windows app from the flutter_frontend folder, then run this file again.
    pause
    exit /b 1
)

echo Launching Diya Flutter app...
start "Diya" /D "%FLUTTER_DIR%" "%FLUTTER_APP%"
if errorlevel 1 (
    echo Could not launch the Diya Flutter app.
    pause
    exit /b 1
)

echo Diya is open. Keep the Diya API window running while you use it.
endlocal
