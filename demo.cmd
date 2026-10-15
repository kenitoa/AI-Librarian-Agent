@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
title AI Librarian Agent - CMD Demo
mode con cols=112 lines=38
color 0F
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
".venv\Scripts\python.exe" validate_live.py --demo %*
set "DEMO_EXIT_CODE=%errorlevel%"
pause
exit /b %DEMO_EXIT_CODE%
