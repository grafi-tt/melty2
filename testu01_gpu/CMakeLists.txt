# testu01_gpu: the tests of BigCrush on a GPU, by HIP (ROCm) or CUDA, for counter-based generators.
#
#   cmake -B build [-DTESTU01_GPU_BACKEND=HIP|CUDA] [-DCMAKE_HIP_ARCHITECTURES=gfx1030 | -DCMAKE_CUDA_ARCHITECTURES=89]
#         [-DTestU01_ROOT=/usr/local] [-DTESTU01_GPU_HOST_FLAGS=-march=native]
#   cmake --build build && cmake --install build --prefix /some/prefix
#
# Options:
#   TESTU01_GPU_BACKEND       HIP (ROCm: hiprtc, rocFFT, rocThrust) or CUDA (NVRTC, cuFFT, Thrust); default: HIP if a
#                             HIP compiler is found, else CUDA
#   CMAKE_HIP_ARCHITECTURES   (HIP) the GPU architectures of the kernels of rocThrust (the kernels of the tests are
#                             compiled at run time for the device); default gfx1030
#   CMAKE_CUDA_ARCHITECTURES  (CUDA) the GPU architectures of the kernels of Thrust; default native (the GPUs of
#                             the build machine), or those of nvcc if it finds none
#   TestU01_ROOT              the installation prefix of TestU01 (its headers, probdist and mylib)
#   TESTU01_GPU_HOST_FLAGS    flags of the host code only (default none, portable), e.g. -march=native for the
#                             SIMD of the host in the tests on the host (LinearComp)
#   TESTU01_GPU_EXAMPLES      build the example splitmix_bigcrush (default ON for the top-level project)
#   BUILD_SHARED_LIBS         a shared library in place of the static one
cmake_minimum_required(VERSION 3.21)

project(testu01_gpu VERSION 0.1.0 LANGUAGES CXX)

set(TESTU01_GPU_BACKEND "" CACHE STRING "the GPU backend: HIP or CUDA (empty: HIP if a HIP compiler is found, else CUDA)")
set_property(CACHE TESTU01_GPU_BACKEND PROPERTY STRINGS "" HIP CUDA)
if(NOT TESTU01_GPU_BACKEND)
  include(CheckLanguage)
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    set(TESTU01_GPU_BACKEND HIP CACHE STRING "the GPU backend: HIP or CUDA" FORCE)
  else()
    set(TESTU01_GPU_BACKEND CUDA CACHE STRING "the GPU backend: HIP or CUDA" FORCE)
  endif()
endif()
string(TOUPPER "${TESTU01_GPU_BACKEND}" TESTU01_GPU_BACKEND_UPPER)
if(TESTU01_GPU_BACKEND_UPPER STREQUAL "HIP")
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES AND NOT DEFINED ENV{HIPARCHS})
    set(CMAKE_HIP_ARCHITECTURES gfx1030 CACHE STRING "the GPU architectures of the kernels of rocThrust")
  endif()
  enable_language(HIP)
  set(gpu_lang HIP)
elseif(TESTU01_GPU_BACKEND_UPPER STREQUAL "CUDA")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS} AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "the GPU architectures of the kernels of Thrust")
  endif()
  enable_language(CUDA)
  set(gpu_lang CUDA)
else()
  message(FATAL_ERROR "TESTU01_GPU_BACKEND must be HIP or CUDA, not '${TESTU01_GPU_BACKEND}'")
endif()
message(STATUS "testu01_gpu: the backend ${gpu_lang}")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "the build type" FORCE)
endif()

set(TESTU01_GPU_HOST_FLAGS "" CACHE STRING "flags of the host code only, e.g. -march=native")
option(TESTU01_GPU_EXAMPLES "build the example splitmix_bigcrush" ${PROJECT_IS_TOP_LEVEL})

include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
if(gpu_lang STREQUAL "HIP")
  find_package(hip REQUIRED)
  find_package(hiprtc REQUIRED)
  find_package(rocfft REQUIRED)
  find_package(rocthrust REQUIRED)
  set(gpu_runtime hip::host)
  set(gpu_libs hip::host roc::rocthrust hiprtc::hiprtc roc::rocfft)
else()
  find_package(CUDAToolkit REQUIRED)
  set(gpu_runtime CUDA::cudart)
  set(gpu_libs CUDA::cudart CUDA::cuda_driver CUDA::nvrtc CUDA::cufft)
endif()
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

# the host code, compiled as HIP or CUDA (for the kernels of rocThrust or Thrust)
set_source_files_properties(src/testu01_gpu.hip PROPERTIES LANGUAGE ${gpu_lang})
add_library(testu01_gpu src/testu01_gpu.hip ${gen}/device.inc ${gen}/params.inc)
add_library(testu01_gpu::testu01_gpu ALIAS testu01_gpu)
target_compile_features(testu01_gpu PUBLIC cxx_std_17)
# the host code is C++20 (std::format); the header needs only C++17 of its users
set_target_properties(testu01_gpu PROPERTIES ${gpu_lang}_STANDARD 20 ${gpu_lang}_STANDARD_REQUIRED ON ${gpu_lang}_EXTENSIONS OFF
                                             POSITION_INDEPENDENT_CODE ON
                                             VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
if(gpu_lang STREQUAL "CUDA")
  target_compile_definitions(testu01_gpu PRIVATE TESTU01_GPU_CUDA)
endif()
target_include_directories(testu01_gpu
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE src ${gen})
if(gpu_lang STREQUAL "HIP")
  target_compile_options(testu01_gpu PRIVATE -Wall)
  foreach(f IN LISTS TESTU01_GPU_HOST_FLAGS)
    target_compile_options(testu01_gpu PRIVATE "SHELL:-Xarch_host ${f}")
  endforeach()
else()  # nvcc passes the flags of the host compiler by -Xcompiler
  target_compile_options(testu01_gpu PRIVATE -Xcompiler=-Wall)
  foreach(f IN LISTS TESTU01_GPU_HOST_FLAGS)
    target_compile_options(testu01_gpu PRIVATE "-Xcompiler=${f}")
  endforeach()
endif()
target_link_libraries(testu01_gpu PRIVATE ${gpu_libs} TestU01::probdist TestU01::mylib)

if(TESTU01_GPU_EXAMPLES)
  # abseil (its LTS releases from 20230125, with absl/log): the logging of the example (the library doesn't use it)
  find_package(absl REQUIRED)
  # (HIP: compiled as HIP for hip_runtime.h; CUDA: a C++ program with the runtime of CUDA)
  if(gpu_lang STREQUAL "HIP")
    set_source_files_properties(examples/splitmix_bigcrush.cpp PROPERTIES LANGUAGE HIP)
  endif()
  add_executable(splitmix_bigcrush examples/splitmix_bigcrush.cpp)
  set_target_properties(splitmix_bigcrush PROPERTIES ${gpu_lang}_STANDARD 20 ${gpu_lang}_STANDARD_REQUIRED ON ${gpu_lang}_EXTENSIONS OFF)
  target_compile_features(splitmix_bigcrush PRIVATE cxx_std_20)
  target_compile_options(splitmix_bigcrush PRIVATE -Wall)
  if(gpu_lang STREQUAL "CUDA")
    target_compile_definitions(splitmix_bigcrush PRIVATE TESTU01_GPU_CUDA)
  endif()
  target_link_libraries(splitmix_bigcrush PRIVATE testu01_gpu::testu01_gpu ${gpu_runtime} absl::log absl::log_initialize absl::log_globals)
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
