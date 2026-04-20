@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "RUNNER=%SCRIPT_DIR%run_demo.cmd"

if not exist "%RUNNER%" (
    echo ERROR: Cannot find %RUNNER%
    exit /b 1
)

echo === Classroom Monitoring Demo Menu ===
echo.
choice /c VCR /m "Select source type: [V]ideo, [C]amera, [R]TSP"
if errorlevel 3 (
    set "SOURCE_TYPE=rtsp"
) else if errorlevel 2 (
    set "SOURCE_TYPE=camera"
) else (
    set "SOURCE_TYPE=video"
)

set "VIDEO_FILE="
set "CAMERA_ID="
set "RTSP_URL="
set "OUTPUT_DIR="
set "CONFIG="
set "ENV_NAME="
set "SECRETS_FILE="
set "CONDA_EXE="

if /I "%SOURCE_TYPE%"=="video" (
    echo.
    set /p "VIDEO_FILE=Video file path [default: project test_data\data.mp4]: "
)

if /I "%SOURCE_TYPE%"=="camera" (
    echo.
    set /p "CAMERA_ID=Camera ID [default: 0]: "
)

if /I "%SOURCE_TYPE%"=="rtsp" (
    call :prompt_rtsp
)

echo.
set /p "OUTPUT_DIR=Output directory [blank = auto timestamped output\demo_...]: "
set /p "CONFIG=Config path [blank = deploy_bundle\config\demo_final.yml]: "
set /p "ENV_NAME=Conda env name [blank = paddle_det]: "
set /p "CONDA_EXE=Full path to conda.exe [blank = auto detect]: "

choice /c YN /m "Enable Telegram alerts?"
if errorlevel 2 (
    set "USE_TELEGRAM=0"
) else (
    set "USE_TELEGRAM=1"
    set /p "SECRETS_FILE=Telegram env file [blank = deploy_bundle\secrets\telegram.env]: "
)

choice /c YN /m "Enable local preview window?"
if errorlevel 2 (
    set "PREVIEW_FLAG="
) else (
    set "PREVIEW_FLAG=--preview-local"
)

set /p "PREVIEW_MAX_FPS=Preview max FPS [blank = config/default]: "
if defined PREVIEW_MAX_FPS (
    set PREVIEW_FPS_ARG=--preview-max-fps "%PREVIEW_MAX_FPS%"
) else (
    set "PREVIEW_FPS_ARG="
)

choice /c YN /m "Save rendered video to disk?"
if errorlevel 2 (
    set "SAVE_FLAG=--no-save-video"
) else (
    set "SAVE_FLAG="
)

choice /c GN /m "Use GPU? [G]=GPU [N]=CPU"
if errorlevel 2 (
    set "CPU_FLAG=--cpu"
) else (
    set "CPU_FLAG="
)

choice /c YN /m "Dry-run only?"
if errorlevel 2 (
    set "DRY_RUN_FLAG="
) else (
    set "DRY_RUN_FLAG=--dry-run"
)

set "VIDEO_ARG="
set "CAMERA_ARG="
set "RTSP_ARG="
set "OUTPUT_ARG="
set "CONFIG_ARG="
set "ENV_ARG="
set "CONDA_ARG="
set "TELEGRAM_ARG="
set "SECRETS_ARG="

if defined VIDEO_FILE set VIDEO_ARG=--video-file "%VIDEO_FILE%"
if defined CAMERA_ID set CAMERA_ARG=--camera-id "%CAMERA_ID%"
if defined RTSP_URL set RTSP_ARG=--rtsp-url "%RTSP_URL%"
if defined OUTPUT_DIR set OUTPUT_ARG=--output-dir "%OUTPUT_DIR%"
if defined CONFIG set CONFIG_ARG=--config "%CONFIG%"
if defined ENV_NAME set ENV_ARG=--env-name "%ENV_NAME%"
if defined CONDA_EXE set CONDA_ARG=--conda-exe "%CONDA_EXE%"
if "%USE_TELEGRAM%"=="1" (
    set "TELEGRAM_ARG=--use-telegram"
    if defined SECRETS_FILE set SECRETS_ARG=--secrets-file "%SECRETS_FILE%"
)

echo.
echo Launching:
echo   "%RUNNER%" %SOURCE_TYPE% %VIDEO_ARG% %CAMERA_ARG% %RTSP_ARG% %OUTPUT_ARG% %CONFIG_ARG% %ENV_ARG% %CONDA_ARG% %TELEGRAM_ARG% %SECRETS_ARG% %PREVIEW_FLAG% %PREVIEW_FPS_ARG% %SAVE_FLAG% %CPU_FLAG% %DRY_RUN_FLAG%
echo.
call "%RUNNER%" %SOURCE_TYPE% %VIDEO_ARG% %CAMERA_ARG% %RTSP_ARG% %OUTPUT_ARG% %CONFIG_ARG% %ENV_ARG% %CONDA_ARG% %TELEGRAM_ARG% %SECRETS_ARG% %PREVIEW_FLAG% %PREVIEW_FPS_ARG% %SAVE_FLAG% %CPU_FLAG% %DRY_RUN_FLAG%
exit /b %ERRORLEVEL%

:prompt_rtsp
echo.
set /p "RTSP_URL=RTSP URL: "
if not defined RTSP_URL (
    echo RTSP URL is required.
    goto prompt_rtsp
)
exit /b 0
