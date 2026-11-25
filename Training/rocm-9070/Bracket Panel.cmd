@echo off
where pyw.exe >nul 2>&1
if errorlevel 1 goto console
start "" pyw.exe -3.12 "%~dp0bracket_panel.py" %*
exit /b
:console
py.exe -3.12 "%~dp0bracket_panel.py" %*
if errorlevel 1 pause
