add_executable(test_melty2 test_melty2.c)
add_test(NAME test_melty2 COMMAND test_melty2)

add_executable(test_melty2_hpp test_melty2_hpp.cc)
add_test(NAME test_melty2_hpp COMMAND test_melty2_hpp)

set(EXES_C test_melty2)
set_property(TARGET ${EXES_C} PROPERTY C_STANDARD 99)
set_property(TARGET ${EXES_C} PROPERTY C_STANDARD_REQUIRED ON)
if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    set_property(TARGET ${EXES_C} APPEND PROPERTY COMPILE_OPTIONS
        -std=c99 -Wall -Wextra -Wpedantic -Werror)
endif()

set(EXES_CXX test_melty2_hpp)
set_property(TARGET ${EXES_CXX} PROPERTY CXX_STANDARD_REQUIRED ON)
if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_property(TARGET ${EXES_CXX} APPEND PROPERTY COMPILE_OPTIONS
        -std=c++11 -Wall -Wextra -Wpedantic -Werror)
endif()

set_property(TARGET ${EXES_C} ${EXES_CXX} APPEND PROPERTY LINK_LIBRARIES melty2)
