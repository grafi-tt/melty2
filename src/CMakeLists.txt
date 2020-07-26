add_library(melty2 melty2.c blake3.c)

set_property(TARGET melty2 PROPERTY C_STANDARD 99)
set_property(TARGET melty2 PROPERTY C_STANDARD_REQUIRED ON)
if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(melty2 PRIVATE -O3 -fno-stack-protector -Wall -Wextra -Wpedantic -Werror)
endif()

target_include_directories(melty2 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>)
