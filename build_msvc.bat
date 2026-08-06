@echo off
REM Set up MSVC environment and run CMake.
REM ASCII only: cmd parses this under the console codepage, so non-ASCII
REM comments corrupt parsing. This is the one file in the repo without
REM Korean comments, deliberately.
REM
REM Paths contain "(x86)"; parentheses inside if-blocks break cmd parsing,
REM so control flow uses goto rather than parenthesised blocks.
REM
REM Usage:  build_msvc.bat [cmake args...]
REM   build_msvc.bat                      -> print toolchain versions
REM   build_msvc.bat -S . -B build/msvc   -> configure
REM   build_msvc.bat --build build/msvc   -> build

setlocal
set "VSROOT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "VCVARS=%VSROOT%\VC\Auxiliary\Build\vcvars64.bat"
set "VSEXT=%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake"

if not exist "%VCVARS%" goto :missing

call "%VCVARS%" >nul
set "PATH=%VSEXT%\CMake\bin;%VSEXT%\Ninja;%PATH%"

if "%~1"=="" goto :versions

cmake %*
exit /b %ERRORLEVEL%

:versions
echo === toolchain ===
REM cl prints its banner in the system language, so match the version
REM number rather than the word "Version".
cl 2>&1 | findstr /C:"19."
cmake --version
ninja --version
exit /b 0

:missing
echo ERROR: vcvars64.bat not found
echo   expected under %VSROOT%
exit /b 1
