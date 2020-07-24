cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(melty2 LANGUAGES C CXX)

option(MELTY2_ENABLE_TEST "Enable test target" OFF)

add_library(melty2 melty2.c)
set_property(TARGET melty2 PROPERTY C_STANDARD 99)
set_property(TARGET melty2 PROPERTY C_STANDARD_REQUIRED ON)
target_include_directories(melty2 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(melty2 PRIVATE -O3 -fno-stack-protector -Wall -Wextra -Wpedantic -Werror)
endif()

if(MELTY2_ENABLE_TEST)
    enable_testing()
    add_subdirectory(test)
endif()
