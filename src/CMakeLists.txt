add_library(melty2_bulk_generic_ STATIC EXCLUDE_FROM_ALL melty2_bulk_impl.c)
list(APPEND BULK_LIBS melty2_bulk_generic_)

if(MELTY2_ENABLE_CPU_FEATURES)
    add_library(melty2_bulk_dispatch_ STATIC EXCLUDE_FROM_ALL melty2_bulk_dispatch.cc)
    target_include_directories(melty2_bulk_dispatch_
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cpu_features/include)
    target_link_libraries(melty2_bulk_dispatch_ PRIVATE cpu_features)

    set_property(TARGET melty2_bulk_dispatch_ PROPERTY CXX_STANDARD 11)
    set_property(TARGET melty2_bulk_dispatch_ PROPERTY CXX_STANDARD_REQUIRED ON)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
        set_property(TARGET melty2_bulk_dispatch_ APPEND PROPERTY COMPILE_OPTIONS
            -std=c++11 -Wall -Wextra -Wpedantic -Werror)
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86$)")
        if(MSVC)
            add_library(melty2_bulk_avx512_ STATIC EXCLUDE_FROM_ALL melty2_bulk_impl.c)
            target_compile_options(melty2_bulk_avx512_ PRIVATE /arch:AVX512)
            target_compile_definitions(melty2_bulk_avx512_ PRIVATE MELTY2_BULK_SUFFIX=avx512)
            list(APPEND BULK_LIBS melty2_bulk_avx512_)

            add_library(melty2_bulk_avx2_ STATIC EXCLUDE_FROM_ALL melty2_bulk_impl.c)
            target_compile_options(melty2_bulk_avx2_ PRIVATE /arch:AVX2)
            target_compile_definitions(melty2_bulk_avx2_ PRIVATE MELTY2_BULK_SUFFIX=avx2)
            list(APPEND BULK_LIBS melty2_bulk_avx2_)
        else()
            add_library(melty2_bulk_avx512_ STATIC EXCLUDE_FROM_ALL melty2_bulk_impl.c)
            target_compile_options(melty2_bulk_avx512_ PRIVATE -mavx512f -mavx512dq)
            target_compile_definitions(melty2_bulk_avx512_ PRIVATE MELTY2_BULK_SUFFIX=avx512)
            list(APPEND BULK_LIBS melty2_bulk_avx512_)

            add_library(melty2_bulk_avx2_ STATIC EXCLUDE_FROM_ALL melty2_bulk_impl.c)
            target_compile_options(melty2_bulk_avx2_ PRIVATE -mavx2)
            target_compile_definitions(melty2_bulk_avx2_ PRIVATE MELTY2_BULK_SUFFIX=avx2)
            list(APPEND BULK_LIBS melty2_bulk_avx2_)

            add_library(melty2_bulk_sse2_ STATIC EXCLUDE_FROM_ALL melty2_bulk_impl.c)
            target_compile_options(melty2_bulk_sse2_ PRIVATE -msse2)
            target_compile_definitions(melty2_bulk_sse2_ PRIVATE MELTY2_BULK_SUFFIX=sse2)
            list(APPEND BULK_LIBS melty2_bulk_sse2_)
        endif()
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm" AND NOT MSVC)
        add_library(melty2_bulk_neon_ STATIC EXCLUDE_FROM_ALL melty2_bulk_impl.c)
        target_compile_options(melty2_bulk_neon_ PRIVATE -mfpu=neon)
        target_compile_definitions(melty2_bulk_neon_ PRIVATE MELTY2_BULK_SUFFIX=neon)
        list(APPEND BULK_LIBS melty2_bulk_neon_)
    endif()

    target_compile_definitions(melty2_bulk_generic_ PRIVATE MELTY2_BULK_SUFFIX=generic)
endif()

add_library(melty2 blake3.c melty2.c)
target_include_directories(melty2
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>)

if(MELTY2_ENABLE_CPU_FEATURES)
    set(INTERNAL_LIBS melty2_bulk_dispatch_ ${BULK_LIBS})
else()
    set(INTERNAL_LIBS ${BULK_LIBS})
endif()
get_property(USE_PIC TARGET melty2 PROPERTY POSITION_INDEPENDENT_CODE)
set_property(TARGET ${INTERTNAL_LIBS} PROPERTY POSITION_INDEPENDENT_CODE ${USE_PIC})
target_link_libraries(melty2 ${INTERNAL_LIBS})

set_property(TARGET melty2 ${BULK_LIBS} PROPERTY C_STANDARD 99)
set_property(TARGET melty2 ${BULK_LIBS} PROPERTY C_STANDARD_REQUIRED ON)
if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    set_property(TARGET melty2 ${BULK_LIBS} APPEND PROPERTY COMPILE_OPTIONS
        -std=c99 -Wall -Wextra -Wpedantic -Werror)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    set_property(TARGET ${BULK_LIBS} APPEND PROPERTY COMPILE_OPTIONS -O3)
endif()
