add_executable(test_gen test_gen.c)
add_test(NAME test_gen COMMAND test_gen)

add_executable(test_seeder test_seeder.c)
add_test(NAME test_seeder COMMAND test_seeder)

set(EXES test_gen test_seeder)

set_property(TARGET ${EXES} PROPERTY C_STANDARD 99)
set_property(TARGET ${EXES} PROPERTY C_STANDARD_REQUIRED ON)
if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    set_property(TARGET ${EXES} APPEND PROPERTY COMPILE_OPTIONS
        -std=c99 -Wall -Wextra -Wpedantic -Werror)
endif()

set_property(TARGET ${EXES} APPEND PROPERTY LINK_LIBRARIES melty2)
