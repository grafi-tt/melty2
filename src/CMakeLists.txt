add_library(melty2 blake3.c melty2.c melty2_bulk_dispatch.c)

add_library(melty2_bulk_generic_ STATIC melty2_bulk_impl.c)
target_compile_definitions(melty2_bulk_generic_ PRIVATE MELTY2_BULK_SUFFIX=generic)
list(APPEND BULK_LIBS melty2_bulk_generic_)

if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86$)")
        target_compile_definitions(melty2 PRIVATE MELTY2_ENABLE_CPU_FEATURES)

        add_library(melty2_bulk_avx512_ STATIC melty2_bulk_impl.c)
        target_compile_options(melty2_bulk_avx512_ PRIVATE -mavx512f -mavx512dq)
        target_compile_definitions(melty2_bulk_avx512_ PRIVATE MELTY2_BULK_SUFFIX=avx512)
        list(APPEND BULK_LIBS melty2_bulk_avx512_)

        add_library(melty2_bulk_avx2_ STATIC melty2_bulk_impl.c)
        target_compile_options(melty2_bulk_avx2_ PRIVATE -mavx2)
        target_compile_definitions(melty2_bulk_avx2_ PRIVATE MELTY2_BULK_SUFFIX=avx2)
        list(APPEND BULK_LIBS melty2_bulk_avx2_)
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
        target_compile_definitions(melty2 PRIVATE MELTY2_ENABLE_CPU_FEATURES)

        add_library(melty2_bulk_neon_ STATIC melty2_bulk_impl.c)
        target_compile_options(melty2_bulk_neon_ PRIVATE -mfpu=neon)
        target_compile_definitions(melty2_bulk_neon_ PRIVATE MELTY2_BULK_SUFFIX=neon)
        list(APPEND BULK_LIBS melty2_bulk_neon_)
    endif()

    set_property(TARGET ${BULK_LIBS} APPEND PROPERTY COMPILE_OPTIONS -O3)
endif()

set_property(TARGET melty2 ${BULK_LIBS} PROPERTY C_STANDARD 99)
set_property(TARGET melty2 ${BULK_LIBS} PROPERTY C_STANDARD_REQUIRED ON)
if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    set_property(TARGET ${BULK_LIBS} APPEND PROPERTY COMPILE_OPTIONS
        -std=c99 -Wall -Wextra -Wpedantic -Werror)
endif()

target_link_libraries(melty2 PRIVATE ${BULK_LIBS} cpu_features)
target_include_directories(melty2
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cpu_features/include)
