@echo off
echo ========================================
echo Manga Image Translator - Setup Checker
echo ========================================
echo.

REM Check Docker
echo [1/5] Checking Docker...
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running or not installed
    echo    Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/
) else (
    echo ✅ Docker is running
)
echo.

REM Check Docker Compose
echo [2/5] Checking Docker Compose...
docker-compose version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose not available
) else (
    echo ✅ Docker Compose is available
)
echo.

REM Check configuration file
echo [3/5] Checking configuration file...
if not exist "docker-compose-deepseek.yml" (
    echo ❌ docker-compose-deepseek.yml not found
) else (
    echo ✅ Configuration file exists
)
echo.

REM Check API key configuration
echo [4/5] Checking API key configuration...
findstr /C:"YOUR_DEEPSEEK_API_KEY_HERE" docker-compose-deepseek.yml >nul 2>&1
if %errorlevel% equ 0 (
    echo ⚠️  DeepSeek API key not configured
    echo    Please edit docker-compose-deepseek.yml and add your API key
    echo    Get your key from: https://platform.deepseek.com/
) else (
    echo ✅ API key appears to be configured
)
echo.

REM Check if service is running
echo [5/5] Checking service status...
docker-compose -f docker-compose-deepseek.yml ps >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Service is not running
    echo    Run deploy-windows.bat to start the service
) else (
    docker-compose -f docker-compose-deepseek.yml ps | findstr "Up" >nul
    if %errorlevel% equ 0 (
        echo ✅ Service is running
        echo    Web interface: http://localhost:8000
    ) else (
        echo ⚠️  Service containers exist but may not be running
        echo    Check with: docker-compose -f docker-compose-deepseek.yml logs
    )
)
echo.

REM Check result directory
if not exist "result" (
    echo Creating result directory...
    mkdir result
    echo ✅ Result directory created
) else (
    echo ✅ Result directory exists
)

echo.
echo ========================================
echo Setup check complete!
echo ========================================
echo.
pause