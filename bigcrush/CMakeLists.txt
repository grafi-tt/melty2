add_executable(run_bigcrush run_bigcrush.cc)
target_compile_features(run_bigcrush PRIVATE cxx_std_20)

find_library(TESTU01_TESTU01_LIB NAMES testu01 REQUIRED)
find_library(TESTU01_PROBDIST_LIB NAMES probdist REQUIRED)
find_library(TESTU01_MYLIB_LIB NAMES mylib REQUIRED)
find_path(TESTU01_INCLUDE_DIR NAMES unif01.h bbattery.h REQUIRED)

target_link_libraries(run_bigcrush PRIVATE melty2
  ${TESTU01_TESTU01_LIB}
  ${TESTU01_PROBDIST_LIB}
  ${TESTU01_MYLIB_LIB}
)
target_include_directories(run_bigcrush PRIVATE ${TESTU01_INCLUDE_DIR})
