@echo off
REM ============================================================
REM  Emergent Garden - repeated runs for statistics
REM
REM  Runs each config RUNS times and saves each run under:
REM      logs\<config>\<config>_run01 ... _run10
REM  Each run folder holds rounds.csv, migrations.csv,
REM  cluster_sizes.csv, events.jsonl, sim_log.txt and the .png plots.
REM
REM  Runs sequentially (one window at a time). Each run stops
REM  automatically after "max_rounds" set in config.json.
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] venv Python not found at "%PY%".
    echo Create it:  python -m venv .venv  ^&  .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

REM --- Edit these to choose what to run ------------------------
REM Config names = files in configs\ WITHOUT the path or .json extension
set "CONFIGS=cfl_on__emergent_on cfl_on__emergent_off cfl_off__emergent_on cfl_off__emergent_off"
REM Repetitions per config
set "RUNS=10"
REM Uncomment the next line to run with NO window (unattended):
REM set "SDL_VIDEODRIVER=dummy"
REM ------------------------------------------------------------

for %%C in (%CONFIGS%) do (
    for /L %%N in (1,1,%RUNS%) do (
        set "NN=0%%N"
        set "NN=!NN:~-2!"
        echo === %%C : run !NN! / %RUNS% ===
        "%PY%" main.py --config "configs\%%C.json" --log-dir "logs\%%C" --run-name "%%C_run!NN!"
    )
)

echo.
echo All runs complete. See logs\^<config^>\^<config^>_runNN\
pause
endlocal
