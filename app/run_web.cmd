@echo off
setlocal EnableExtensions

pushd "%~dp0\.." >nul

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found on PATH.
  echo Activate your runtime environment first, for example:
  echo   conda activate paddle_det
  popd >nul
  exit /b 1
)

python -c "import fastapi, uvicorn, jinja2, pydantic" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Missing web dependencies for the app UI.
  echo Install them in the active environment with:
  echo   pip install -r app\requirements.txt
  popd >nul
  exit /b 1
)

if "%MEDIAMTX_BIN%"=="" (
  where mediamtx >nul 2>nul
  if errorlevel 1 (
    echo [WARN] MediaMTX was not found on PATH.
    echo [WARN] WebRTC preview needs MediaMTX. Install it or set MEDIAMTX_BIN.
  )
)

if "%APP_HOST%"=="" set "APP_HOST=127.0.0.1"
if "%APP_PORT%"=="" set "APP_PORT=8000"

echo Starting web UI on http://%APP_HOST%:%APP_PORT%
python -m app
set "EXIT_CODE=%ERRORLEVEL%"

popd >nul
exit /b %EXIT_CODE%
