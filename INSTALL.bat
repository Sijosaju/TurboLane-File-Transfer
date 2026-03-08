@echo off
echo ============================================================
echo  TurboLane - Clean Install Script
echo ============================================================
echo.

REM Step 1: Set PYTHONDONTWRITEBYTECODE permanently (current user)
echo [1/5] Disabling Python bytecode cache permanently...
setx PYTHONDONTWRITEBYTECODE 1
set PYTHONDONTWRITEBYTECODE=1
echo      Done.

REM Step 2: Delete all pycache in project folder
echo [2/5] Clearing all __pycache__ folders...
for /d /r "%~dp0" %%d in (__pycache__) do (
    if exist "%%d" rd /s /q "%%d"
)
echo      Done.

REM Step 3: Delete stale egg-info
echo [3/5] Removing stale egg-info...
if exist "%~dp0turbolane_server.egg-info" rd /s /q "%~dp0turbolane_server.egg-info"
echo      Done.

REM Step 4: Delete Q-table (poisoned from failed runs)
echo [4/5] Clearing RL Q-table...
if exist "%~dp0models\dci\q_table.json" del /f /q "%~dp0models\dci\q_table.json"
echo      Done.

REM Step 5: Reinstall the package
echo [5/5] Reinstalling TurboLane package...
cd /d "%~dp0"
C:\Users\sijos\AppData\Local\Programs\Python\Python311\python.exe -m pip install -e . --force-reinstall --no-cache-dir
echo      Done.

echo.
echo ============================================================
echo  Verifying installation...
echo ============================================================
C:\Users\sijos\AppData\Local\Programs\Python\Python311\python.exe -c "from turbolane_server.protocol import CHUNK_SIZE; print('  CHUNK_SIZE =', CHUNK_SIZE, '  (should be 262144)')"
C:\Users\sijos\AppData\Local\Programs\Python\Python311\python.exe -c "from turbolane_server.transfer import ChunkQueue; print('  ChunkQueue import: OK')"

echo.
echo ============================================================
echo  Installation complete!
echo  CHUNK_SIZE should show 262144 above.
echo  You can now run: turbolane-server send ...
echo ============================================================
echo.
pause
