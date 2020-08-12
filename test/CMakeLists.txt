add_executable(test_seeder test_seeder.c)
add_test(NAME test_seeder COMMAND test_seeder)

#add_executable(test_melty2 test_melty2.c)
#add_test(NAME test_melty2 COMMAND test_melty2)

set(EXES test_seeder)

set_property(TARGET ${EXES} PROPERTY C_STANDARD 99)
set_property(TARGET ${EXES} PROPERTY C_STANDARD_REQUIRED ON)
if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    set_property(TARGET ${EXES} APPEND PROPERTY COMPILE_OPTIONS
        -std=c99 -Wall -Wextra -Wpedantic -Werror)
endif()

set_property(TARGET ${EXES} APPEND PROPERTY LINK_LIBRARIES melty2)
