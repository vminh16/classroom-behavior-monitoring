@echo off
setlocal
call "%~dp0run_demo.cmd" %*
exit /b %ERRORLEVEL%
