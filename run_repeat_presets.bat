@echo off
REM ============================================================
REM  Emergent Garden - repeated runs over EMERGENT PRESETS
REM
REM  For each preset, runs both CFL OFF and CFL ON (attraction kept
REM  ON so the preset is visible), RUNS times each, saved under:
REM      logs\<preset>\cfl_off__emergent_on_run01 ... _run10
REM      logs\<preset>\cfl_on__emergent_on_run01  ... _run10
REM
REM  --config sets the cfl/attraction toggles, --preset sets the forces.
REM
REM  Runs sequentially; each run stops after "max_rounds".
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] venv Python not found at "%PY%".
    pause
    exit /b 1
)

REM --- Edit these to choose what to run ------------------------
REM Preset names = files in configs\emergent\ WITHOUT the .json extension
set "PRESETS=default cells swarm orbits gas chains"
REM CFL variants: config files in configs\ that set the toggles. Both keep
REM emergent (attraction) ON so the preset shows; only CFL differs.
set "CFLSET=cfl_off__emergent_on cfl_on__emergent_on"
REM Repetitions per preset and CFL variant
set "RUNS=10"
REM Uncomment the next line to run with NO window (unattended):
REM set "SDL_VIDEODRIVER=dummy"
REM ------------------------------------------------------------

for %%P in (%PRESETS%) do (
    for %%F in (%CFLSET%) do (
        for /L %%N in (1,1,%RUNS%) do (
            set "NN=0%%N"
            set "NN=!NN:~-2!"
            echo === preset %%P : %%F : run !NN! / %RUNS% ===
            "%PY%" main.py --config "configs\%%F.json" --preset %%P --log-dir "logs\%%P" --run-name "%%F_run!NN!"
        )
    )
)

echo.
echo All preset runs complete. See logs\^<preset^>\^<cfl_variant^>_runNN\
pause
endlocal
