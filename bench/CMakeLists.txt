add_executable(bench_cpu bench_cpu.cc)
target_compile_features(bench_cpu PRIVATE cxx_std_17)
target_link_libraries(bench_cpu PRIVATE melty2)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # the other generators are plain loops, vectorized by the compiler for the host
  target_compile_options(bench_cpu PRIVATE -march=native)
endif()
