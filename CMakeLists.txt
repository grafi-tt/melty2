cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(melty2 LANGUAGES C CXX)

option(MELTY2_ENABLE_CPU_FEATURES "Enable runtime CPU detection by cpu_features" ON)
option(MELTY2_ENABLE_TEST "Enable test targets" ON)
option(MELTY2_ENABLE_BIGCRUSH "Enable bigcrush targets" OFF)

if(MELTY2_ENABLE_CPU_FEATURES)
    # Build cpu_featurs as static library
    set(${PROJECT_NAME}_BUILD_SHARED_LIBS_BAK ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "" FORCE)
    add_subdirectory(third_party/cpu_features EXCLUDE_FROM_ALL)
    set(BUILD_SHARED_LIBS ${${PROJECT_NAME}_BUILD_SHARED_LIBS_BAK} CACHE INTERNAL "" FORCE)
    unset(${PROJECT_NAME}_BUILD_SHARED_LIBS_BAK)
endif()

add_subdirectory(src)

if(MELTY2_ENABLE_TEST)
    enable_testing()
    add_subdirectory(test)
endif()
if(MELTY2_ENABLE_BIGCRUSH)
    add_subdirectory(bigcrush)
endif()
