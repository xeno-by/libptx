@echo off

if not "%2"=="" (
    echo [Fatal error] Too many arguments! Please, provide a single argument to this batch file: desired build configuration ^(it can be either "Debug" or "Release"^).
    pause
    exit /B 1
)

SET BUILDCONFIG=%1
if "%1"=="" SET /P BUILDCONFIG=Specify build configuration ^(it can be either "Debug" or "Release"^): 

if "%BUILDCONFIG%"=="Debug" (
    SET BUILDCONFIG=Debug
    echo Building Libcuda using Debug configuration...
) else ( if "%BUILDCONFIG%"=="Release" (
    SET BUILDCONFIG=Release
    echo Building Libcuda using Release configuration...
) else (
    echo [Fatal error] Unsupported build configuration "%BUILDCONFIG%". Only "Debug" or "Release" are supported.
    pause
    exit /B 1
))

if exist Sources rmdir Sources /s /q
mkdir Sources
cd Sources

echo.
echo ^>^>^>^>^> Downloading Libcuda sources from https://libcuda.googlecode.com/hg/
hg clone https://libcuda.googlecode.com/hg/ Libcuda -r 6fb7d04de2dc
if not exist Libcuda (
    echo [Fatal error] Failed to get Libcuda sources.
    pause
    cd ..\
    exit /B 1
)

echo.
echo ^>^>^>^>^> Building Libcuda...
cd Libcuda\Libcuda
"%WINDIR%\Microsoft.NET\Framework\v4.0.30319\msbuild" /t:Rebuild /p:Configuration=%BUILDCONFIG%
if not exist bin\%BUILDCONFIG%\Libcuda.dll (
    echo [Fatal error] Failed to build Libcuda.
    pause
    cd ..\..\..\
    exit /B 1
)


echo.
echo ^>^>^>^>^> Provisioning Libcuda...
cd ..\..\..\
if not exist Binaries mkdir Binaries
cd Binaries
if exist %BUILDCONFIG% rmdir %BUILDCONFIG% /s /q
mkdir %BUILDCONFIG%
cd %BUILDCONFIG%
copy ..\..\Sources\Libcuda\Libcuda\bin\%BUILDCONFIG%\Libcuda.* Libcuda.* /Y
if not exist Libcuda.dll (
    echo [Fatal error] Failed to provision Libcuda.
    pause
    cd ..\..
    exit /B 1
)

cd ..\..
rmdir Sources /s /q
