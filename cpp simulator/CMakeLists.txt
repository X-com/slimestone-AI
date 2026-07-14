cmake_minimum_required(VERSION 3.16)
project(mcp1122_cpp_extract LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(mcp1122_cpp_stream
    src/main.cpp
    src/json_stream.cpp
    src/block_registry.cpp
    src/piston.cpp
    src/simulator.cpp
    src/trace.cpp
    src/world.cpp
)

target_include_directories(mcp1122_cpp_stream PRIVATE src)

if (MSVC)
    target_compile_options(mcp1122_cpp_stream PRIVATE /W4 /permissive-)
else()
    target_compile_options(mcp1122_cpp_stream PRIVATE -Wall -Wextra -Wpedantic)
endif()
