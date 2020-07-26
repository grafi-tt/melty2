add_executable(test_blake3 test_blake3.c)
target_link_libraries(test_blake3 melty2)
add_test(NAME test_blake3 COMMAND test_blake3)

if(MELTY2_ENABLE_BIGCRUSH)
    add_executable(bigcrush bigcrush.c)
    add_executable(bigcrush_bitrev bigcrush.c)
    target_compile_definitions(bigcrush_bitrev PRIVATE BITREV)

    set_property(TARGET bigcrush bigcrush_bitrev PROPERTY C_STANDARD 99)
    set_property(TARGET bigcrush bigcrush_bitrev PROPERTY C_STANDARD_REQUIRED ON)
    set_property(TARGET bigcrush bigcrush_bitrev PROPERTY LINK_LIBRARIES melty2
        testu01 probdist mylib)
endif()
