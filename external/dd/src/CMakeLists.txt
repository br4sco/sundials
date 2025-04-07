install(CODE "MESSAGE(\"\nInstall DD\n\")")

include_directories(${CMAKE_SOURCE_DIR}/src)

set(dd_SOURCES dd_chkpt.c)

set(dd_HEADERS dd_chkpt.h)

sundials_add_library(
  dd
  SOURCES ${dd_SOURCES}
  HEADERS ${dd_HEADERS}
  INCLUDE_SUBDIR dd
  LINK_LIBRARIES PUBLIC sundials_core sundials_idas
  OUTPUT_NAME dd)

message(STATUS "Added DD module")
