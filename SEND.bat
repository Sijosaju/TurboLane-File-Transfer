@echo off
REM TurboLane Send Script
REM Usage: SEND.bat "path\to\file.mkv"

set PYTHONDONTWRITEBYTECODE=1

if "%~1"=="" (
    echo Usage: SEND.bat "path\to\file"
    pause
    exit /b 1
)

turbolane-server send %1 --host 192.168.1.12 --port 9000 --streams 4 --min-streams 4 --max-streams 8
