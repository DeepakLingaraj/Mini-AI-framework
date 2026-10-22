@echo off
title Mini AI Framework Launcher
color 0A
cls

echo.
echo  ============================================================
echo   Mini AI Framework  ^|  MAX + Mojo Engine
echo   Built by: DEEPAK LINGARAJ M
echo  ============================================================
echo.
echo  Select an option:
echo.
echo  [1]  Launch Web Dashboard  (http://127.0.0.1:8000)
echo  [2]  Tokenize Text
echo  [3]  Generate Text (Stream Inference)
echo  [4]  Run Benchmark (All Backends)
echo  [5]  Run Unit Tests
echo  [6]  Exit
echo.
echo  ============================================================

set /p choice= Enter your choice (1-6): 

if "%choice%"=="1" goto serve
if "%choice%"=="2" goto tokenize
if "%choice%"=="3" goto generate
if "%choice%"=="4" goto benchmark
if "%choice%"=="5" goto tests
if "%choice%"=="6" goto end

echo  Invalid choice. Please try again.
pause
goto start

:: -------------------------------------------------------
:serve
cls
echo.
echo  Starting Web Dashboard...
echo  Open your browser at: http://127.0.0.1:8000
echo  Press Ctrl+C to stop the server.
echo.
start "" http://127.0.0.1:8000
.venv\Scripts\python.exe cli.py serve --port 8000
pause
goto start

:: -------------------------------------------------------
:tokenize
cls
echo.
set /p text= Enter text to tokenize (or press Enter for default): 
if "%text%"=="" set text=Building lightweight AI framework with MAX and Mojo
echo.
.venv\Scripts\python.exe cli.py tokenize --text "%text%"
echo.
pause
goto start

:: -------------------------------------------------------
:generate
cls
echo.
set /p prompt= Enter prompt (or press Enter for default): 
if "%prompt%"=="" set prompt=The future of AI framework is

echo.
echo  Select backend:
echo  [1] numpy  (default)
echo  [2] mojo_simd
echo  [3] max_engine
set /p bk= Backend choice (1-3): 

if "%bk%"=="2" set backend=mojo_simd
if "%bk%"=="3" set backend=max_engine
if not defined backend set backend=numpy

echo.
.venv\Scripts\python.exe cli.py generate --prompt "%prompt%" --backend %backend%
set backend=
echo.
pause
goto start

:: -------------------------------------------------------
:benchmark
cls
echo.
echo  Running Multi-Backend Benchmark Comparison...
echo  (NumPy SIMD  vs  MAX Engine  vs  Mojo SIMD)
echo.
.venv\Scripts\python.exe cli.py benchmark
echo.
pause
goto start

:: -------------------------------------------------------
:tests
cls
echo.
echo  Running Unit Tests...
echo.
.venv\Scripts\python.exe -m unittest discover tests -v
echo.
pause
goto start

:: -------------------------------------------------------
:end
cls
echo.
echo  Goodbye! Exiting Mini AI Framework Launcher.
echo.
timeout /t 2 >nul
exit
