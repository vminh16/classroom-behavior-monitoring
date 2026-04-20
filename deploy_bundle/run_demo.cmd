@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

set "SOURCE_TYPE=video"
set "VIDEO_FILE="
set "CAMERA_ID=0"
set "RTSP_URL="
set "OUTPUT_DIR="
set "ENV_NAME=paddle_det"
set "CONFIG="
set "SECRETS_FILE="
set "RUN_MODE=paddle"
set "CONDA_EXE_OVERRIDE="
set "PREVIEW_LOCAL=0"
set "NO_SAVE_VIDEO=0"
set "PREVIEW_MAX_FPS=0"
set "USE_TELEGRAM=0"
set "USE_CPU=0"
set "DRY_RUN=0"

:parse_args
if "%~1"=="" goto after_parse

if /I "%~1"=="video" (
    set "SOURCE_TYPE=video"
    shift
    goto parse_args
)
if /I "%~1"=="camera" (
    set "SOURCE_TYPE=camera"
    shift
    goto parse_args
)
if /I "%~1"=="rtsp" (
    set "SOURCE_TYPE=rtsp"
    shift
    goto parse_args
)
if /I "%~1"=="--help" goto usage
if /I "%~1"=="-h" goto usage
if /I "%~1"=="/?" goto usage

if /I "%~1"=="--source-type" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "SOURCE_TYPE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="-SourceType" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "SOURCE_TYPE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--video-file" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "VIDEO_FILE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="-VideoFile" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "VIDEO_FILE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--camera-id" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "CAMERA_ID=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="-CameraId" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "CAMERA_ID=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--rtsp-url" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "RTSP_URL=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="-RtspUrl" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "RTSP_URL=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--output-dir" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "OUTPUT_DIR=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="-OutputDir" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "OUTPUT_DIR=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--env-name" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "ENV_NAME=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="-EnvName" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "ENV_NAME=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--config" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "CONFIG=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="-Config" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "CONFIG=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--secrets-file" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "SECRETS_FILE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="-SecretsFile" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "SECRETS_FILE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--run-mode" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "RUN_MODE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="-RunMode" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "RUN_MODE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--conda-exe" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "CONDA_EXE_OVERRIDE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="-CondaExe" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "CONDA_EXE_OVERRIDE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--use-telegram" (
    set "USE_TELEGRAM=1"
    shift
    goto parse_args
)
if /I "%~1"=="-UseTelegram" (
    set "USE_TELEGRAM=1"
    shift
    goto parse_args
)
if /I "%~1"=="--preview-local" (
    set "PREVIEW_LOCAL=1"
    shift
    goto parse_args
)
if /I "%~1"=="-PreviewLocal" (
    set "PREVIEW_LOCAL=1"
    shift
    goto parse_args
)
if /I "%~1"=="--no-save-video" (
    set "NO_SAVE_VIDEO=1"
    shift
    goto parse_args
)
if /I "%~1"=="-NoSaveVideo" (
    set "NO_SAVE_VIDEO=1"
    shift
    goto parse_args
)
if /I "%~1"=="--preview-max-fps" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "PREVIEW_MAX_FPS=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="-PreviewMaxFps" (
    if "%~2"=="" call :missing_value "%~1" & exit /b 2
    set "PREVIEW_MAX_FPS=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--cpu" (
    set "USE_CPU=1"
    shift
    goto parse_args
)
if /I "%~1"=="-Cpu" (
    set "USE_CPU=1"
    shift
    goto parse_args
)
if /I "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
    goto parse_args
)
if /I "%~1"=="-DryRun" (
    set "DRY_RUN=1"
    shift
    goto parse_args
)

echo ERROR: Unknown argument: %~1
echo.
goto usage_error

:after_parse
if not defined CONFIG set "CONFIG=%SCRIPT_DIR%config\demo_final.yml"
if not defined SECRETS_FILE set "SECRETS_FILE=%SCRIPT_DIR%secrets\telegram.env"
if not defined OUTPUT_DIR call :build_timestamp STAMP
if not defined OUTPUT_DIR set "OUTPUT_DIR=%PROJECT_ROOT%\output\demo_%STAMP%"
if not defined VIDEO_FILE set "VIDEO_FILE=%PROJECT_ROOT%\test_data\data.mp4"

call :resolve_conda
if errorlevel 1 exit /b %ERRORLEVEL%

if /I "%SOURCE_TYPE%"=="video" goto source_ok
if /I "%SOURCE_TYPE%"=="camera" goto source_ok
if /I "%SOURCE_TYPE%"=="rtsp" goto source_ok
echo ERROR: Source type must be one of: video, camera, rtsp
exit /b 2

:source_ok
if not exist "%PROJECT_ROOT%\output_inference\picodet_m_416_classroom" (
    echo ERROR: Required path not found: %PROJECT_ROOT%\output_inference\picodet_m_416_classroom
    echo Restore runtime artifacts first. See docs\ARTIFACTS.md.
    exit /b 1
)
if not exist "%PROJECT_ROOT%\output_inference\pplcnet_behavior" (
    echo ERROR: Required path not found: %PROJECT_ROOT%\output_inference\pplcnet_behavior
    echo Restore runtime artifacts first. See docs\ARTIFACTS.md.
    exit /b 1
)
if not exist "%CONFIG%" (
    echo ERROR: Config not found: %CONFIG%
    exit /b 1
)

if /I "%SOURCE_TYPE%"=="video" (
    if not exist "%VIDEO_FILE%" (
        echo ERROR: Video file not found: %VIDEO_FILE%
        echo Restore sample inputs first. See docs\ARTIFACTS.md.
        exit /b 1
    )
)
if /I "%SOURCE_TYPE%"=="rtsp" (
    if not defined RTSP_URL (
        echo ERROR: SourceType=rtsp requires --rtsp-url
        exit /b 2
    )
)

if "%USE_CPU%"=="1" (
    set "DEVICE=CPU"
) else (
    set "DEVICE=GPU"
)

if "%USE_TELEGRAM%"=="1" (
    call :load_dotenv "%SECRETS_FILE%"
    if errorlevel 1 exit /b %ERRORLEVEL%
    if not defined TELEGRAM_BOT_TOKEN (
        echo ERROR: Telegram was requested but TELEGRAM_BOT_TOKEN is missing in %SECRETS_FILE%
        exit /b 1
    )
    if not defined TELEGRAM_CHAT_ID (
        echo ERROR: Telegram was requested but TELEGRAM_CHAT_ID is missing in %SECRETS_FILE%
        exit /b 1
    )
    set "TELEGRAM_ENABLE_TEXT=True"
) else (
    set "TELEGRAM_ENABLE_TEXT=False"
)

echo === Demo Deploy Bundle ===
echo Project root : %PROJECT_ROOT%
echo Config       : %CONFIG%
echo Environment  : %ENV_NAME%
echo Conda        : %CONDA_EXE_PATH%
echo Device       : %DEVICE%
echo Run mode     : %RUN_MODE%
echo Source type  : %SOURCE_TYPE%
if /I "%SOURCE_TYPE%"=="video" echo Video file   : %VIDEO_FILE%
if /I "%SOURCE_TYPE%"=="camera" echo Camera ID    : %CAMERA_ID%
if /I "%SOURCE_TYPE%"=="rtsp" echo RTSP URL     : %RTSP_URL%
echo Output dir   : %OUTPUT_DIR%
if "%USE_TELEGRAM%"=="1" (
    echo Telegram     : enabled via env file
) else (
    echo Telegram     : disabled
)
if "%PREVIEW_LOCAL%"=="1" (
    echo Preview      : forced local preview
) else (
    echo Preview      : config/default
)
if "%NO_SAVE_VIDEO%"=="1" (
    echo Save video   : disabled
) else (
    echo Save video   : config/default
)
echo.
echo Command:
call :print_command

if "%DRY_RUN%"=="1" exit /b 0

pushd "%PROJECT_ROOT%" >nul
if /I "%SOURCE_TYPE%"=="video" (
    if "%USE_TELEGRAM%"=="1" (
        call :run_pipeline --video_file "%VIDEO_FILE%" "TELEGRAM_ALERT.enable=True" "TELEGRAM_ALERT.bot_token=%TELEGRAM_BOT_TOKEN%" "TELEGRAM_ALERT.chat_id=%TELEGRAM_CHAT_ID%"
    ) else (
        call :run_pipeline --video_file "%VIDEO_FILE%" "TELEGRAM_ALERT.enable=False"
    )
) else if /I "%SOURCE_TYPE%"=="camera" (
    if "%USE_TELEGRAM%"=="1" (
        call :run_pipeline --camera_id "%CAMERA_ID%" "TELEGRAM_ALERT.enable=True" "TELEGRAM_ALERT.bot_token=%TELEGRAM_BOT_TOKEN%" "TELEGRAM_ALERT.chat_id=%TELEGRAM_CHAT_ID%"
    ) else (
        call :run_pipeline --camera_id "%CAMERA_ID%" "TELEGRAM_ALERT.enable=False"
    )
) else (
    if "%USE_TELEGRAM%"=="1" (
        call :run_pipeline --rtsp "%RTSP_URL%" "TELEGRAM_ALERT.enable=True" "TELEGRAM_ALERT.bot_token=%TELEGRAM_BOT_TOKEN%" "TELEGRAM_ALERT.chat_id=%TELEGRAM_CHAT_ID%"
    ) else (
        call :run_pipeline --rtsp "%RTSP_URL%" "TELEGRAM_ALERT.enable=False"
    )
)
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%

:print_command
if /I "%SOURCE_TYPE%"=="video" (
    call :print_pipeline_command --video_file "%VIDEO_FILE%" "TELEGRAM_ALERT.enable=%TELEGRAM_ENABLE_TEXT%"
    exit /b 0
)
if /I "%SOURCE_TYPE%"=="camera" (
    call :print_pipeline_command --camera_id "%CAMERA_ID%" "TELEGRAM_ALERT.enable=%TELEGRAM_ENABLE_TEXT%"
    exit /b 0
)
call :print_pipeline_command --rtsp "%RTSP_URL%" "TELEGRAM_ALERT.enable=%TELEGRAM_ENABLE_TEXT%"
exit /b 0

:run_pipeline
set "SOURCE_ARG_NAME=%~1"
set "SOURCE_ARG_VALUE=%~2"
set "TELEGRAM_ARG=%~3"
set "TELEGRAM_TOKEN_ARG=%~4"
set "TELEGRAM_CHAT_ARG=%~5"
set "OPTIONAL_PREVIEW_ARG="
set "OPTIONAL_SAVE_ARG="
set "OPTIONAL_FPS_ARG="
if "%PREVIEW_LOCAL%"=="1" set "OPTIONAL_PREVIEW_ARG=preview_local=True"
if "%NO_SAVE_VIDEO%"=="1" set "OPTIONAL_SAVE_ARG=save_visual_output=False"
if not "%PREVIEW_MAX_FPS%"=="0" set "OPTIONAL_FPS_ARG=preview_max_fps=%PREVIEW_MAX_FPS%"
"%CONDA_EXE_PATH%" run -n "%ENV_NAME%" python deploy/pipeline/pipeline_product.py --config "%CONFIG%" --device %DEVICE% --run_mode "%RUN_MODE%" --output_dir "%OUTPUT_DIR%" %SOURCE_ARG_NAME% "%SOURCE_ARG_VALUE%" --opt "visual=True" "visual_style=minimal" %OPTIONAL_PREVIEW_ARG% %OPTIONAL_SAVE_ARG% %OPTIONAL_FPS_ARG% %TELEGRAM_ARG% %TELEGRAM_TOKEN_ARG% %TELEGRAM_CHAT_ARG%
exit /b %ERRORLEVEL%

:print_pipeline_command
set "SOURCE_ARG_NAME=%~1"
set "SOURCE_ARG_VALUE=%~2"
set "TELEGRAM_ARG=%~3"
set "TELEGRAM_TOKEN_ARG=%~4"
set "TELEGRAM_CHAT_ARG=%~5"
set "OPTIONAL_PREVIEW_ARG="
set "OPTIONAL_SAVE_ARG="
set "OPTIONAL_FPS_ARG="
if "%PREVIEW_LOCAL%"=="1" set "OPTIONAL_PREVIEW_ARG=preview_local=True"
if "%NO_SAVE_VIDEO%"=="1" set "OPTIONAL_SAVE_ARG=save_visual_output=False"
if not "%PREVIEW_MAX_FPS%"=="0" set "OPTIONAL_FPS_ARG=preview_max_fps=%PREVIEW_MAX_FPS%"
echo "%CONDA_EXE_PATH%" run -n "%ENV_NAME%" python deploy/pipeline/pipeline_product.py --config "%CONFIG%" --device %DEVICE% --run_mode "%RUN_MODE%" --output_dir "%OUTPUT_DIR%" %SOURCE_ARG_NAME% "%SOURCE_ARG_VALUE%" --opt "visual=True" "visual_style=minimal" %OPTIONAL_PREVIEW_ARG% %OPTIONAL_SAVE_ARG% %OPTIONAL_FPS_ARG% %TELEGRAM_ARG% %TELEGRAM_TOKEN_ARG% %TELEGRAM_CHAT_ARG%
exit /b 0

:resolve_conda
if defined CONDA_EXE_OVERRIDE (
    if exist "%CONDA_EXE_OVERRIDE%" (
        for %%I in ("%CONDA_EXE_OVERRIDE%") do set "CONDA_EXE_PATH=%%~fI"
        exit /b 0
    )
    echo ERROR: Cannot find conda.exe at: %CONDA_EXE_OVERRIDE%
    exit /b 1
)
if defined CONDA_EXE (
    if exist "%CONDA_EXE%" (
        for %%I in ("%CONDA_EXE%") do set "CONDA_EXE_PATH=%%~fI"
        exit /b 0
    )
)
if defined USERPROFILE (
    if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" (
        for %%I in ("%USERPROFILE%\anaconda3\Scripts\conda.exe") do set "CONDA_EXE_PATH=%%~fI"
        exit /b 0
    )
    if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
        for %%I in ("%USERPROFILE%\miniconda3\Scripts\conda.exe") do set "CONDA_EXE_PATH=%%~fI"
        exit /b 0
    )
)
if exist "C:\Users\USER\anaconda3\Scripts\conda.exe" (
    for %%I in ("C:\Users\USER\anaconda3\Scripts\conda.exe") do set "CONDA_EXE_PATH=%%~fI"
    exit /b 0
)
echo ERROR: Cannot find conda.exe. Pass --conda-exe "C:\full\path\to\conda.exe".
exit /b 1

:load_dotenv
if not exist "%~1" exit /b 0
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%~1") do call :assign_env "%%~A" "%%~B"
exit /b 0

:assign_env
set "DOTENV_KEY=%~1"
set "DOTENV_VAL=%~2"
if not defined DOTENV_KEY exit /b 0
if "%DOTENV_VAL:~0,1%"=="^"" if "%DOTENV_VAL:~-1%"=="^"" set "DOTENV_VAL=%DOTENV_VAL:~1,-1%"
if "%DOTENV_VAL:~0,1%"=="'" if "%DOTENV_VAL:~-1%"=="'" set "DOTENV_VAL=%DOTENV_VAL:~1,-1%"
set "%DOTENV_KEY%=%DOTENV_VAL%"
exit /b 0

:build_timestamp
set "STAMP=%DATE%_%TIME%"
set "STAMP=%STAMP: =0%"
set "STAMP=%STAMP:/=%"
set "STAMP=%STAMP::=%"
set "STAMP=%STAMP:,=%"
set "STAMP=%STAMP:.=%"
set "STAMP=%STAMP:-=%"
set "%~1=%STAMP%"
exit /b 0

:missing_value
echo ERROR: Missing value for %~1
exit /b 0

:usage_error
exit /b 2

:usage
echo Usage:
echo   run_demo.cmd [video^|camera^|rtsp] [options]
echo.
echo Examples:
echo   run_demo.cmd video --dry-run
echo   run_demo.cmd camera --camera-id 0
echo   run_demo.cmd rtsp --rtsp-url "rtsp://user:pass@camera/Streaming/Channels/101"
echo   run_demo.cmd video --use-telegram --secrets-file "%SCRIPT_DIR%secrets\telegram.env"
echo.
echo Options:
echo   --source-type ^<video^|camera^|rtsp^>
echo   --video-file ^<path^>
echo   --camera-id ^<int^>
echo   --rtsp-url ^<url^>
echo   --output-dir ^<path^>
echo   --env-name ^<name^>
echo   --config ^<path^>
echo   --secrets-file ^<path^>
echo   --run-mode ^<mode^>
echo   --conda-exe ^<path-to-conda.exe^>
echo   --use-telegram
echo   --preview-local
echo   --no-save-video
echo   --preview-max-fps ^<int^>
echo   --cpu
echo   --dry-run
exit /b 0
