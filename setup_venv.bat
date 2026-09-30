@echo off
REM One-time setup: creates .venv and installs dependencies.
cd /d "%~dp0"
python -m venv .venv || goto :err
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :err
echo.
echo Done. In VS Code: Ctrl+Shift+P ^> "Python: Select Interpreter" ^> .venv,
echo then Run and Debug (Ctrl+Shift+D) ^> "Profiler - test video" ^> F5.
pause
exit /b 0
:err
echo Setup failed.
pause
exit /b 1
