@echo off
echo.
echo This batch file will build a custom MKL dynamic link library for usage by CNTK.
echo.
echo Requirements:
echo  - Intel MKL SDK installed on the machine
echo  - MKLROOT environment variable is set to the MKL directory inside the Intel MKL SDK
echo  - Visual Studio 2013 installed and included in the path
echo.

setlocal enableextensions enabledelayedexpansion

pushd "%~dp0"
if errorlevel 1 (
  echo Could not change directory to script location.
  exit /b 1
)

if not defined MKLROOT (
  echo Error: Environment variable MKLROOT is undefined.
  exit /b 1
)

if not exist "%MKLROOT%" (
  echo Error: Directory doesn't exist: "%MKLROOT%".
  exit /b 1
)

set MKLBUILDERROOT=%MKLROOT%\tools\builder

if not exist "%MKLBUILDERROOT%" (
  echo Error: Directory doesn't exist: "%MKLBUILDERROOT%".
  exit /b 1
)

where /q nmake.exe
if errorlevel 1 (
  echo Error: NMAKE.EXE not in path.
  exit /b 1
)

where /q link.exe
if errorlevel 1 (
  echo Error: LINK.EXE not in path.
  exit /b 1
)

set /p CNTKCUSTOMMKLVERSION=<version.txt
if not defined CNTKCUSTOMMKLVERSION (
  echo Cannot determine CNTK custom MKL version.
  exit /b 1
)

if exist lib rmdir /s /q lib
if errorlevel 1 exit /b 1

if exist Publish rmdir /s /q Publish
if errorlevel 1 exit /b 1

mkdir -p Publish\%CNTKCUSTOMMKLVERSION%\x64

echo.
echo Copying "%MKLBUILDERROOT%\lib".

xcopy /s /e /y /i "%MKLBUILDERROOT%\lib" lib
if errorlevel 1 (
  exit /b 1
)

echo.
echo Compiling and copying libraries.

for %%t in (
  parallel
  sequential
) do (

  set TFIRSTCHAR=%%t
  set TFIRSTCHAR=!TFIRSTCHAR:~0,1!
  set LIBBASENAME=mkl_cntk_!TFIRSTCHAR!

  echo.
  echo Calling NMAKE libintel64 export=cntklist.txt threading=%%t name=!LIBBASENAME! MKLROOT="%MKLROOT%".
  NMAKE /f "%MKLBUILDERROOT%\makefile" ^
    libintel64 ^
    export=cntklist.txt ^
    threading=%%t ^
    name=!LIBBASENAME! ^
    MKLROOT="%MKLROOT%"

  if errorlevel 1 (
    echo Error: NMAKE.exe for threading=%%t failed.
    exit /b 1
  )

  mkdir Publish\%CNTKCUSTOMMKLVERSION%\x64\%%t
  if errorlevel 1 exit /b 1

  move !LIBBASENAME!.dll Publish\%CNTKCUSTOMMKLVERSION%\x64\%%t
  if errorlevel 1 exit /b 1

  move !LIBBASENAME!.lib Publish\%CNTKCUSTOMMKLVERSION%\x64\%%t
  if errorlevel 1 exit /b 1

  del !LIBBASENAME!*
  if errorlevel 1 exit /b 1
  @REM TODO manifest?
)

echo.
echo Copying libiomp5md.dll.

copy "%MKLROOT%\..\redist\intel64_win\compiler\libiomp5md.dll" Publish\%CNTKCUSTOMMKLVERSION%\x64\parallel
if errorlevel 1 (
  exit /b 1
)

echo.
echo Removing LIB directory.

rmdir /s /q lib
if errorlevel 1 exit /b 1

echo.
echo Copying include files to Publish\%CNTKCUSTOMMKLVERSION%\include.

mkdir Publish\%CNTKCUSTOMMKLVERSION%\include

for /f %%h in (cntkheaderlist.txt) do (
  copy "%MKLROOT%\include\%%h" Publish\%CNTKCUSTOMMKLVERSION%\include
  if errorlevel 1 (
    echo Failed to copy "%MKLROOT%\include\%%h".
    exit /b 1
  )
)

copy license.txt Publish\%CNTKCUSTOMMKLVERSION%
if errorlevel 1 (
  echo Failed to copy license.txt.
  exit /b 1
)

popd
