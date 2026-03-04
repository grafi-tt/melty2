macro(melty2_add_gen_lib_ suffix)
  add_library(melty2_gen_${suffix}_ STATIC EXCLUDE_FROM_ALL melty2_gen.c)
  target_include_directories(melty2_gen_${suffix}_ PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
  target_compile_features(melty2_gen_${suffix}_ PRIVATE c_std_99)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(melty2_gen_${suffix}_ PRIVATE -std=c99 -Wall -Wextra -Wpedantic -Werror $<$<NOT:$<CONFIG:Debug>>:-O3>)
  endif()
  if(MELTY2_ENABLE_CPU_FEATURES)
    target_compile_definitions(melty2_gen_${suffix}_ PRIVATE MELTY2_GEN_SUFFIX=${suffix})
  endif()
endmacro()

melty2_add_gen_lib_(avx512)
melty2_add_gen_lib_(avx2)
melty2_add_gen_lib_(sse2)
melty2_add_gen_lib_(neon)
melty2_add_gen_lib_(generic)

list(APPEND MELTY2_GEN_LIBS melty2_gen_generic_)

if(MELTY2_ENABLE_CPU_FEATURES)
  add_library(melty2_gen_dispatch_ STATIC EXCLUDE_FROM_ALL melty2_gen_dispatch.cc)
  target_include_directories(melty2_gen_dispatch_ PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cpu_features/include
  )
  target_link_libraries(melty2_gen_dispatch_ PRIVATE cpu_features)
  target_compile_features(melty2_gen_dispatch_ PRIVATE cxx_std_11)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(melty2_gen_dispatch_ PRIVATE -std=c++11 -Wall -Wextra -Wpedantic -Werror)
  endif()
  list(APPEND MELTY2_GEN_LIBS melty2_gen_dispatch_)

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86$)")
    if(MSVC)
      target_compile_options(melty2_gen_avx512_ PRIVATE /arch:AVX512)
      list(APPEND MELTY2_GEN_LIBS melty2_gen_avx512_)
      target_compile_options(melty2_gen_avx2_ PRIVATE /arch:AVX2)
      list(APPEND MELTY2_GEN_LIBS melty2_gen_avx2_)
    else()
      target_compile_options(melty2_gen_avx512_ PRIVATE -mavx512f -mavx512dq)
      list(APPEND MELTY2_GEN_LIBS melty2_gen_avx512_)
      target_compile_definitions(melty2_gen_avx2_ PRIVATE MELTY2_GEN_SUFFIX=avx2)
      list(APPEND MELTY2_GEN_LIBS melty2_gen_avx2_)
      target_compile_options(melty2_gen_sse2_ PRIVATE -msse2)
      list(APPEND MELTY2_GEN_LIBS melty2_gen_sse2_)
    endif()
  endif()

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm" AND NOT MSVC)
    target_compile_options(melty2_gen_neon_ PRIVATE -mfpu=neon)
    list(APPEND MELTY2_GEN_LIBS melty2_gen_neon_)
  endif()
endif()

add_library(melty2 melty2_init.c)
target_include_directories(melty2 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(melty2 PUBLIC c_std_99)
if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
  target_compile_options(melty2 PRIVATE -std=c99 -Wall -Wextra -Wpedantic -Werror)
endif()

if(MELTY2_ENABLE_CPU_FEATURES)
  set(MELTY2_INTERNAL_LIBS melty2_gen_dispatch_ ${MELTY2_GEN_LIBS})
else()
  set(MELTY2_INTERNAL_LIBS ${MELTY2_GEN_LIBS})
endif()
get_property(melty2_use_pic_ TARGET melty2 PROPERTY POSITION_INDEPENDENT_CODE)
set_property(TARGET ${MELTY2_INTERTNAL_LIBS} PROPERTY POSITION_INDEPENDENT_CODE ${melty2_use_pic_})
target_link_libraries(melty2 ${MELTY2_INTERNAL_LIBS})
