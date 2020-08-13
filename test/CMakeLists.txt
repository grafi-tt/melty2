add_executable(test_melty2_gen test_melty2_gen.c)
add_test(NAME test_melty2_gen COMMAND test_melty2_gen)

add_executable(test_melty2_seeder test_melty2_seeder.c)
add_test(NAME test_melty2_seeder COMMAND test_melty2_seeder)

set(EXES test_melty2_gen test_melty2_seeder)

set_property(TARGET ${EXES} PROPERTY C_STANDARD 99)
set_property(TARGET ${EXES} PROPERTY C_STANDARD_REQUIRED ON)
if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    set_property(TARGET ${EXES} APPEND PROPERTY COMPILE_OPTIONS
        -std=c99 -Wall -Wextra -Wpedantic -Werror)
endif()

set_property(TARGET ${EXES} APPEND PROPERTY LINK_LIBRARIES melty2)
