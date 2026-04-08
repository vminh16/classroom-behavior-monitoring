@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0run_demo.ps1" %*
exit /b %ERRORLEVEL%
