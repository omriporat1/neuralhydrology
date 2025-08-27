@echo off
setlocal enabledelayedexpansion

rem ---- User-configurable defaults (for reproducibility) ----
set JOB_SUB_ID=14
set JOB_ID=0
set RUN_SUFFIX=cmd
set OVERWRITE=1
set DEBUG=1

set BATCH_SIZE=
set LEARNING_RATE=0.00005
set SAVE_EVERY=1
set NUM_WORKERS=0

set DATA_DIR=C:\PhD\Data\Caravan
set TEMPLATE=%~dp0local_debug_template.yml

set TRAIN_BASIN_FILE=
set VAL_BASIN_FILE=
set TEST_BASIN_FILE=
rem ----------------------------------------------------------

pushd "%~dp0"
if not exist "logs" mkdir "logs"
if not exist "results" mkdir "results"

rem Force CPU for PyTorch
set CUDA_VISIBLE_DEVICES=

rem Export reproducibility env flags
if defined BATCH_SIZE set BATCH_SIZE=%BATCH_SIZE%
set RUN_SUFFIX=%RUN_SUFFIX%
if "%OVERWRITE%"=="1" ( set ALLOW_OVERWRITE=1 ) else ( set ALLOW_OVERWRITE=0 )
set PYTHONUNBUFFERED=1

set PY=python
set ARGS=".\train_single_model_with_av_rain_local_debug.py" ^
 --job-sub-id %JOB_SUB_ID% ^
 --job-id %JOB_ID% ^
 --save-every %SAVE_EVERY% ^
 --learning-rate %LEARNING_RATE% ^
 --data-dir "%DATA_DIR%" ^
 --cpu-only ^
 --num-workers %NUM_WORKERS% ^
 --template "%TEMPLATE%"

if "%DEBUG%"=="1" set ARGS=%ARGS% --debug
if not "%RUN_SUFFIX%"=="" set ARGS=%ARGS% --run-suffix "%RUN_SUFFIX%"
if not "%TRAIN_BASIN_FILE%"=="" set ARGS=%ARGS% --train-basin-file "%TRAIN_BASIN_FILE%"
if not "%VAL_BASIN_FILE%"=="" set ARGS=%ARGS% --val-basin-file "%VAL_BASIN_FILE%"
if not "%TEST_BASIN_FILE%"=="" set ARGS=%ARGS% --test-basin-file "%TEST_BASIN_FILE%"

for /f "tokens=1-3 delims=/:. " %%a in ("%DATE% %TIME%") do (
  set STAMP=%%a-%%b-%%c_%%d-%%e-%%f
)
set OUTLOG=logs\output_local_%STAMP%.log
set ERRLOG=logs\error_local_%STAMP%.log

echo Running: %PY% %ARGS%
%PY% %ARGS% 1>> "%OUTLOG%" 2>> "%ERRLOG%"
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
  echo Training failed with exit code %RC%. See "%ERRLOG%".
  popd
  exit /b %RC%
)

echo Training completed. Logs: "%OUTLOG%"
popd