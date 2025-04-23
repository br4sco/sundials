install(CODE "MESSAGE(\"\nInstall DD\n\")")

if (MSVC)
    # warning level 4
    add_compile_options(/W4)
else()
    # additional warnings
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(${CMAKE_SOURCE_DIR}/src)

set(dd_SOURCES matrix.c structure.c pivot.c dd_err.c dd.c)

set(dd_HEADERS alg.h darr.h dd_math.h matrix.h structure.h pivot.h dd_err.h dd.h)

sundials_add_library(
  dd
  SOURCES ${dd_SOURCES}
  HEADERS ${dd_HEADERS}
  INCLUDE_SUBDIR dd
  LINK_LIBRARIES PUBLIC sundials_core sundials_idas
  OUTPUT_NAME dd)

message(STATUS "Added DD module")
