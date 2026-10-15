@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import langchain, langchain_ollama, langgraph" >nul 2>&1
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" "%~dp005_library_agent.py" %*
if errorlevel 1 goto failed
pause
exit /b 0
:failed
echo Setup or execution failed. Please check Python, Ollama, and requirements.
pause
exit /b 1
