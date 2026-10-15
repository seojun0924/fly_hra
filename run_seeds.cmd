@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set ENV=%~1
if "%ENV%"=="" set ENV=LunarLander-v3
set EPISODES=%~2
if "%EPISODES%"=="" set EPISODES=1500
set LOG_DIR=%~3
if "%LOG_DIR%"=="" set LOG_DIR=logs
set TOTAL_STEPS=%~4
set START_SEED=%~5
if "%START_SEED%"=="" set START_SEED=0
set END_SEED=%~6
if "%END_SEED%"=="" set END_SEED=4
if %END_SEED% LSS %START_SEED% (
    echo END_SEED^(%END_SEED%^) is lesser than START_SEED^(%START_SEED%^).
    exit /b 1
)
set STEP_ARG=
if not "%TOTAL_STEPS%"=="" set STEP_ARG=--total-steps %TOTAL_STEPS%
echo ============================================
echo ENV=%ENV%
echo EPISODES=%EPISODES%
echo LOG_DIR=%LOG_DIR%
echo SEEDS=%START_SEED%~%END_SEED%
if "%TOTAL_STEPS%"=="" (
    echo TOTAL_STEPS=^(undefined, shall run as episode-based mode^)
) else (
    echo TOTAL_STEPS=%TOTAL_STEPS%
)
echo ============================================
echo.
for /L %%S in (%START_SEED%,1,%END_SEED%) do (
    echo === seed %%S : baseline ===
    python train.py --env %ENV% --agent baseline --seed %%S --episodes %EPISODES% --log-dir %LOG_DIR% %STEP_ARG%
    if errorlevel 1 goto :error
    echo === seed %%S : fly_hra ===
    python train.py --env %ENV% --agent fly_hra --seed %%S --episodes %EPISODES% --log-dir %LOG_DIR% %STEP_ARG%
    if errorlevel 1 goto :error
)
echo.
echo === evaluate_metrics.py ===
python evaluate_metrics.py --log-dir %LOG_DIR% --env %ENV%
if errorlevel 1 goto :error
echo.
echo === make_report.py ===
python make_report.py --log-dir %LOG_DIR% --env %ENV%
if errorlevel 1 goto :error
echo.
echo run(seed=%START_SEED%~%END_SEED%, baseline+fly_hra) complete. result=%LOG_DIR%\
echo - report: %LOG_DIR%\REPORT_%ENV%.txt
echo - console: %LOG_DIR%\*.console.log
goto :eof
:error
echo.
echo halted due to error.
echo use flag --overwrite if you are sure.
exit /b 1
