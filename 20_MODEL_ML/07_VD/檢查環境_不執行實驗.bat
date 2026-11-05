@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Testing long gap runner without actually running algorithms...
py -3 run_vd_long_gap_validation.py "VD_N3_N_2024_2025_LITE.parquet" --dry-run
pause
