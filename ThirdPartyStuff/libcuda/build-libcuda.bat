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
hg clone https://libcuda.googlecode.com/hg/ Libcuda -r a899aee51751

echo.
echo ^>^>^>^>^> Downloading XenoGears sources from https://xenogears.googlecode.com/hg/
hg clone https://xenogears.googlecode.com/hg/ XenoGears -r b1bd7ae9d315

echo.
echo ^>^>^>^>^> Building Libcuda...
cd Libcuda\Libcuda
"%WINDIR%\Microsoft.NET\Framework\v3.5\msbuild" /t:Rebuild /p:Configuration=%BUILDCONFIG%

echo.
echo ^>^>^>^>^> Ilmerging Libcuda...
cd ..\..\..\
if not exist Binaries mkdir Binaries
cd Binaries
if exist %BUILDCONFIG% rmdir %BUILDCONFIG% /s /q
mkdir %BUILDCONFIG%
cd %BUILDCONFIG%
ilmerge /t:library /out:Libcuda.dll ..\..\Sources\Libcuda\Libcuda\bin\%BUILDCONFIG%\Libcuda.dll ..\..\Sources\Libcuda\Libcuda\bin\%BUILDCONFIG%\XenoGears.dll /internalize /log

cd ..\..
rmdir Sources /s /q