@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [1/4] Check Python
py -3 --version
if errorlevel 1 (echo Python 3 is required. Install it from python.org & pause & exit /b 1)
echo [2/4] Install dependencies
py -3 -m pip install duckdb numpy pandas scipy scikit-learn matplotlib
if errorlevel 1 (echo Dependency installation failed. & pause & exit /b 1)
echo [3/4] Download missing project source modules from GitHub (first time only)
py -3 prepare_vd_sources.py
if errorlevel 1 (echo Source download failed. Check internet access. & pause & exit /b 1)
echo [4/4] Run long-gap validation on 6, 24, 72 and 151.5 hours
py -3 run_vd_long_gap_validation.py "VD_N3_N_2024_2025_LITE.parquet"
echo.
if errorlevel 1 (echo Experiment did not complete. Read the error messages above.) else (echo DONE: check outputs\vd_long_gap_validation)
pause
