@echo off

if not "%2"=="" (
    echo [Fatal error] Too many arguments! Please, provide a single argument to this batch file: desired build configuration ^(it can be either "Debug" or "Release"^).
    pause
    exit /B 1
)

if "%1"=="" (
    echo [Fatal error] Too little arguments! Please, provide a single argument to this batch file: desired build configuration ^(it can be either "Debug" or "Release"^).
    pause
    exit /B 1
) else ( 

if "%1"=="Debug" (
    echo Building Libcuda using Debug configuration...
) else ( if "%1"=="Release" (
    echo Building Libcuda using Release configuration...
) else (
    echo [Fatal error] Unsupported build configuration "%1". Only "Debug" or "Release" are supported.
    @pause
    exit /B 1
)))

if exist Sources rmdir Sources /s /q
mkdir Sources
cd Sources

echo.
echo ^>^>^>^>^> Downloading Libcuda sources from https://libcuda.googlecode.com/hg/
hg clone https://libcuda.googlecode.com/hg/ Libcuda -r ebc3af4a8a12

echo.
echo ^>^>^>^>^> Downloading XenoGears sources from https://xenogears.googlecode.com/hg/
hg clone https://xenogears.googlecode.com/hg/ XenoGears -r e9e9791dc334

echo.
echo ^>^>^>^>^> Building Libcuda...
cd Libcuda\Libcuda
"%WINDIR%\Microsoft.NET\Framework\v3.5\msbuild" /t:Rebuild /p:Configuration=%1

echo.
echo ^>^>^>^>^> Ilmerging Libcuda...
cd ..\..\..\
if not exist Binaries mkdir Binaries
cd Binaries
if exist %1 rmdir %1 /s /q
mkdir %1
cd %1
ilmerge /t:library /out:Libcuda.dll ..\..\Sources\Libcuda\Libcuda\bin\%1\Libcuda.dll ..\..\Sources\Libcuda\Libcuda\bin\%1\XenoGears.dll /internalize /log

cd ..\..
rmdir Sources /s /q
@pause