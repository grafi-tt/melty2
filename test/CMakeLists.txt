add_executable(test_blake3 test_blake3.c)
add_test(NAME test_blake3 COMMAND test_blake3)

set_property(TARGET test_blake3 PROPERTY C_STANDARD 99)
set_property(TARGET test_blake3 PROPERTY C_STANDARD_REQUIRED ON)

target_link_libraries(test_blake3 PRIVATE melty2)
