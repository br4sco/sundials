install(CODE "MESSAGE(\"\nInstall DD\n\")")

if(MSVC)
  # warning level 4
  add_compile_options(/W4)
else()
  # additional warnings
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(${CMAKE_SOURCE_DIR}/src)

set(dd_SOURCES
    dd_staged_pivot_matrix.c
    static_info.c
    dynamic_info.c
    dd_staged_pivot.c
    dd_state_pivot.c
    dd_err.c
    dd.c)

set(dd_HEADERS
    macros.h
    alg.h
    darr.h
    dd_math.h
    dd_staged_pivot_matrix.h
    static_info.h
    dynamic_info.h
    dd_staged_pivot.h
    dd_staged_pivot_impl.h
    dd_state_pivot.h
    dd_err.h
    dd.h)

sundials_add_library(
  dd
  SOURCES ${dd_SOURCES}
  HEADERS ${dd_HEADERS}
  INCLUDE_SUBDIR dd
  LINK_LIBRARIES PUBLIC sundials_core sundials_idas
  OUTPUT_NAME dd)

message(STATUS "Added DD module")
