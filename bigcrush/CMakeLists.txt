add_executable(run_bigcrush run_bigcrush.c)

set_property(TARGET run_bigcrush PROPERTY C_STANDARD 99)
set_property(TARGET run_bigcrush PROPERTY C_STANDARD_REQUIRED ON)

find_library(${PROJECT_NAME}_TESTU01_LIBRARIES NAMES testu01 probdist mylib REQUIRED)
find_path(${PROJECT_NAME}_TESTU01_INCLUDE_DIR NAMES unif01.h bbattery.h REQUIRED)

target_link_libraries(run_bigcrush PRIVATE melty2 ${${PROJECT_NAME}_TESTU01_LIBRARIES})
target_include_directories(run_bigcrush PRIVATE ${${PROJECT_NAME}_TESTU01_INCLUDE_DIR})
