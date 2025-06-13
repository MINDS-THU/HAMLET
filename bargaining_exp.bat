@echo off
REM ────────────────────────────────────────────────────────────────
REM  run_all_pairs.bat  –  kicks off the single-executor experiment
REM ────────────────────────────────────────────────────────────────

SETLOCAL EnableDelayedExpansion

echo.
echo ==============================================================
echo   Starting bargaining job …
echo ==============================================================

REM ── 1) Activate Conda environment
CALL conda activate base

REM ── 2) Change to your project root
cd /d "C:\Files\Project\COOPA"

REM ── 3) User-configurable parameters
SET data_split=validation
SET random_seed=30
SET num_workers=12
SET log_dir=logs
SET mode=uniform

REM Optional: restrict which models participate.
REM Example → SET model_filter=gpt Qwen
SET model_filter= 

REM ── 4) Ensure log directory exists
IF NOT EXIST "%log_dir%" mkdir "%log_dir%"

REM ── 5) Build optional --model_filter argument
SET "model_filter_arg="
IF NOT "%model_filter%"=="" (
    SET "model_filter_arg=--model_filter %model_filter%"
)

REM ── 6) Launch Python (single run, handles all pairs)
python -m apps.bargaining.run ^
        --data_split %data_split% ^
        --mode %mode% ^
        --random_seed %random_seed% ^
        --num_workers %num_workers% ^
        --log_dir "%log_dir%" ^
        %model_filter_arg%

echo.
echo All experiments completed!
PAUSE
