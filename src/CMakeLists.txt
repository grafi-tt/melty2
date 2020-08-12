add_library(melty2_gen_generic_ STATIC EXCLUDE_FROM_ALL melty2_gen.c)
list(APPEND MELTY2_GEN_LIBS melty2_gen_generic_)

if(MELTY2_ENABLE_CPU_FEATURES)
    add_library(melty2_gen_dispatch_ STATIC EXCLUDE_FROM_ALL melty2_gen_dispatch.cc)
    target_include_directories(melty2_gen_dispatch_
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cpu_features/include)
    target_link_libraries(melty2_gen_dispatch_ PRIVATE cpu_features)

    set_property(TARGET melty2_gen_dispatch_ PROPERTY CXX_STANDARD 11)
    set_property(TARGET melty2_gen_dispatch_ PROPERTY CXX_STANDARD_REQUIRED ON)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set_property(TARGET melty2_gen_dispatch_ APPEND PROPERTY COMPILE_OPTIONS
            -std=c++11 -Wall -Wextra -Wpedantic -Werror)
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86$)")
        if(MSVC)
            add_library(melty2_gen_avx512_ STATIC EXCLUDE_FROM_ALL melty2_gen.c)
            target_compile_options(melty2_gen_avx512_ PRIVATE /arch:AVX512)
            target_compile_definitions(melty2_gen_avx512_ PRIVATE MELTY2_GEN_SUFFIX=avx512)
            list(APPEND MELTY2_GEN_LIBS melty2_gen_avx512_)

            add_library(melty2_gen_avx2_ STATIC EXCLUDE_FROM_ALL melty2_gen.c)
            target_compile_options(melty2_gen_avx2_ PRIVATE /arch:AVX2)
            target_compile_definitions(melty2_gen_avx2_ PRIVATE MELTY2_GEN_SUFFIX=avx2)
            list(APPEND MELTY2_GEN_LIBS melty2_gen_avx2_)
        else()
            add_library(melty2_gen_avx512_ STATIC EXCLUDE_FROM_ALL melty2_gen.c)
            target_compile_options(melty2_gen_avx512_ PRIVATE -mavx512f -mavx512dq)
            target_compile_definitions(melty2_gen_avx512_ PRIVATE MELTY2_GEN_SUFFIX=avx512)
            list(APPEND MELTY2_GEN_LIBS melty2_gen_avx512_)

            add_library(melty2_gen_avx2_ STATIC EXCLUDE_FROM_ALL melty2_gen.c)
            target_compile_options(melty2_gen_avx2_ PRIVATE -mavx2)
            target_compile_definitions(melty2_gen_avx2_ PRIVATE MELTY2_GEN_SUFFIX=avx2)
            list(APPEND MELTY2_GEN_LIBS melty2_gen_avx2_)

            add_library(melty2_gen_sse2_ STATIC EXCLUDE_FROM_ALL melty2_gen.c)
            target_compile_options(melty2_gen_sse2_ PRIVATE -msse2)
            target_compile_definitions(melty2_gen_sse2_ PRIVATE MELTY2_GEN_SUFFIX=sse2)
            list(APPEND MELTY2_GEN_LIBS melty2_gen_sse2_)
        endif()
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm" AND NOT MSVC)
        add_library(melty2_gen_neon_ STATIC EXCLUDE_FROM_ALL melty2_gen.c)
        target_compile_options(melty2_gen_neon_ PRIVATE -mfpu=neon)
        target_compile_definitions(melty2_gen_neon_ PRIVATE MELTY2_GEN_SUFFIX=neon)
        list(APPEND MELTY2_GEN_LIBS melty2_gen_neon_)
    endif()

    target_compile_definitions(melty2_gen_generic_ PRIVATE MELTY2_GEN_SUFFIX=generic)
endif()

set_property(TARGET ${MELTY2_GEN_LIBS} PROPERTY INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}/../include)

if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    set_property(TARGET ${MELTY2_GEN_LIBS} APPEND PROPERTY COMPILE_OPTIONS $<$<NOT:$<CONFIG:Debug>>:-O3>)
endif()

add_library(melty2 melty2_seeder.c)
target_include_directories(melty2
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>)

set_property(TARGET melty2 ${MELTY2_GEN_LIBS} PROPERTY C_STANDARD 99)
set_property(TARGET melty2 ${MELTY2_GEN_LIBS} PROPERTY C_STANDARD_REQUIRED ON)
if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    set_property(TARGET melty2 ${MELTY2_GEN_LIBS} APPEND PROPERTY COMPILE_OPTIONS
        -std=c99 -Wall -Wextra -Wpedantic -Werror)
endif()

if(MELTY2_ENABLE_CPU_FEATURES)
    set(MELTY2_INTERNAL_LIBS melty2_gen_dispatch_ ${MELTY2_GEN_LIBS})
else()
    set(MELTY2_INTERNAL_LIBS ${MELTY2_GEN_LIBS})
endif()
get_property(USE_PIC TARGET melty2 PROPERTY POSITION_INDEPENDENT_CODE)
set_property(TARGET ${MELTY2_INTERTNAL_LIBS} PROPERTY POSITION_INDEPENDENT_CODE ${USE_PIC})
target_link_libraries(melty2 ${MELTY2_INTERNAL_LIBS})
