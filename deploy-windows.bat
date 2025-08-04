@echo off
echo ========================================
echo Manga Image Translator - DeepSeek Setup
echo ========================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not installed!
    echo Please install Docker Desktop and make sure it's running.
    echo Download from: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

echo Docker is running ✓
echo.

REM Check if docker-compose file exists
if not exist "docker-compose-deepseek.yml" (
    echo ERROR: docker-compose-deepseek.yml not found!
    echo Please make sure you're running this script from the project directory.
    pause
    exit /b 1
)

echo Docker Compose file found ✓
echo.

REM Check if API key is configured
findstr /C:"YOUR_DEEPSEEK_API_KEY_HERE" docker-compose-deepseek.yml >nul
if %errorlevel% equ 0 (
    echo WARNING: DeepSeek API key not configured!
    echo Please edit docker-compose-deepseek.yml and replace YOUR_DEEPSEEK_API_KEY_HERE
    echo with your actual DeepSeek API key.
    echo.
    echo Get your API key from: https://platform.deepseek.com/
    echo.
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" (
        echo Setup cancelled.
        pause
        exit /b 1
    )
)

echo.
echo Pulling Docker image (this may take 10-30 minutes - image is ~15GB)...
echo Please be patient, this is a one-time download.
docker pull zyddnys/manga-image-translator:main

if %errorlevel% neq 0 (
    echo ERROR: Failed to pull Docker image!
    echo Please check your internet connection.
    pause
    exit /b 1
)

echo.
echo Starting Manga Image Translator...
docker-compose -f docker-compose-deepseek.yml up -d

if %errorlevel% neq 0 (
    echo ERROR: Failed to start the service!
    echo Check the logs with: docker-compose -f docker-compose-deepseek.yml logs
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS! Manga Image Translator is now running!
echo ========================================
echo.
echo Web Interface: http://localhost:8000
echo.
echo Waiting for service to be ready...
timeout /t 10 /nobreak >nul

REM Try to open the web interface
echo Opening web interface...
start http://localhost:8000

echo.
echo Useful commands:
echo - View logs: docker-compose -f docker-compose-deepseek.yml logs
echo - Stop service: docker-compose -f docker-compose-deepseek.yml down
echo - Restart service: docker-compose -f docker-compose-deepseek.yml restart
echo.
echo Results will be saved in the 'result' folder.
echo.
pause