cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(melty2 LANGUAGES C CXX)

option(MELTY2_ENABLE_TEST "Enable test targets" ON)
option(MELTY2_ENABLE_BIGCRUSH "Enable bigcrush targets" OFF)

add_subdirectory(src)

if(MELTY2_ENABLE_TEST)
    enable_testing()
    add_subdirectory(test)
endif()
