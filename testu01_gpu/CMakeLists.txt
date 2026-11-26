# testu01_gpu: the tests of BigCrush on an AMD GPU (HIP), for counter-based generators.
#
#   cmake -B build -DCMAKE_HIP_ARCHITECTURES=gfx1030 [-DTestU01_ROOT=/usr/local] [-DTESTU01_GPU_HOST_FLAGS=-march=native]
#   cmake --build build && cmake --install build --prefix /some/prefix
#
# Options:
#   CMAKE_HIP_ARCHITECTURES   the GPU architectures of the kernels of rocThrust (the kernels of the tests are
#                             compiled at run time for the device); default gfx1030
#   TestU01_ROOT              the installation prefix of TestU01 (its headers, probdist and mylib)
#   TESTU01_GPU_HOST_FLAGS    flags of the host code only (default none, portable), e.g. -march=native for the
#                             SIMD of the host in the tests on the host (LinearComp)
#   TESTU01_GPU_EXAMPLES      build the example splitmix_bigcrush (default ON for the top-level project)
#   BUILD_SHARED_LIBS         a shared library in place of the static one
cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES AND NOT DEFINED ENV{HIPARCHS})
  set(CMAKE_HIP_ARCHITECTURES gfx1030 CACHE STRING "the GPU architectures of the kernels of rocThrust")
endif()

project(testu01_gpu VERSION 0.1.0 LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "the build type" FORCE)
endif()

set(TESTU01_GPU_HOST_FLAGS "" CACHE STRING "flags of the host code only, e.g. -march=native")
option(TESTU01_GPU_EXAMPLES "build the example splitmix_bigcrush" ${PROJECT_IS_TOP_LEVEL})

include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(hip REQUIRED)
find_package(hiprtc REQUIRED)
find_package(rocfft REQUIRED)
find_package(rocthrust REQUIRED)
find_package(TestU01 REQUIRED)

# the device code and the parameters header, embedded into the library as raw string literals
set(gen ${CMAKE_CURRENT_BINARY_DIR}/generated)
foreach(pair "src/device.hpp;device.inc" "include/testu01_gpu_params.hpp;params.inc")
  list(GET pair 0 in)
  list(GET pair 1 out)
  add_custom_command(
    OUTPUT ${gen}/${out}
    COMMAND ${CMAKE_COMMAND} -DIN=${CMAKE_CURRENT_SOURCE_DIR}/${in} -DOUT=${gen}/${out}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${in} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    COMMENT "Embedding ${in}")
endforeach()

add_library(testu01_gpu src/testu01_gpu.hip ${gen}/device.inc ${gen}/params.inc)
add_library(testu01_gpu::testu01_gpu ALIAS testu01_gpu)
target_compile_features(testu01_gpu PUBLIC cxx_std_17)
set_target_properties(testu01_gpu PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON HIP_EXTENSIONS OFF
                                             POSITION_INDEPENDENT_CODE ON
                                             VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
target_include_directories(testu01_gpu
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE src ${gen})
target_compile_options(testu01_gpu PRIVATE -Wall)
foreach(f IN LISTS TESTU01_GPU_HOST_FLAGS)
  target_compile_options(testu01_gpu PRIVATE "SHELL:-Xarch_host ${f}")
endforeach()
target_link_libraries(testu01_gpu PRIVATE hip::host roc::rocthrust hiprtc::hiprtc roc::rocfft
                                          TestU01::probdist TestU01::mylib)

if(TESTU01_GPU_EXAMPLES)
  set_source_files_properties(examples/splitmix_bigcrush.cpp PROPERTIES LANGUAGE HIP)
  add_executable(splitmix_bigcrush examples/splitmix_bigcrush.cpp)
  set_target_properties(splitmix_bigcrush PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON HIP_EXTENSIONS OFF)
  target_compile_options(splitmix_bigcrush PRIVATE -Wall)
  target_link_libraries(splitmix_bigcrush PRIVATE testu01_gpu::testu01_gpu hip::host)
endif()

# installation: the headers, the library and the package testu01_gpu (find_package(testu01_gpu))
include(CMakePackageConfigHelpers)
set(cmakedir ${CMAKE_INSTALL_LIBDIR}/cmake/testu01_gpu)
install(TARGETS testu01_gpu EXPORT testu01_gpu-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/testu01_gpu.hpp include/testu01_gpu_params.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT testu01_gpu-targets NAMESPACE testu01_gpu:: DESTINATION ${cmakedir})
configure_package_config_file(cmake/testu01_gpu-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/testu01_gpu-config.cmake
                              INSTALL_DESTINATION ${cmakedir})
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/testu01_gpu-config-version.cmake
                                 COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/testu01_gpu-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/testu01_gpu-config-version.cmake cmake/FindTestU01.cmake
        DESTINATION ${cmakedir})
install(FILES LICENSE LICENSE.TestU01 NOTICE DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/testu01_gpu)
