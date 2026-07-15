@echo off
setlocal
call "D:\ProgramFiles\VSBuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
if not exist build mkdir build

set CPP_EXTRACT_SRC=..\cpp simulator\src

"D:\ProgramFiles\Cuda\bin\nvcc.exe" ^
    -std=c++17 -O3 ^
    -arch=sm_52 ^
    --extended-lambda ^
    -Isrc\kernel ^
    -I"%CPP_EXTRACT_SRC%" ^
    src\kernel\block_registry.cu ^
    src\kernel\gpu_kernel.cu ^
    src\host\main_gpu.cu ^
    "%CPP_EXTRACT_SRC%\json_stream.cpp" ^
    -o build\gpu_kernel_stream.exe

if %errorlevel% neq 0 (
    echo Build failed.
    exit /b 1
)
echo Build succeeded: build\gpu_kernel_stream.exe
