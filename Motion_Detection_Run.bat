@echo off
set SCRIPT_PATH=%~dp0app.py
env\Scripts\python.exe -m streamlit run %SCRIPT_PATH%
pause